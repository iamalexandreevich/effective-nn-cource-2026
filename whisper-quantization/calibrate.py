import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ggml_model import F16, Q4_0, Model, replace_q4_payloads
from q4 import Q4Blocks, gptq, reference_q4


class Moments:
    def __init__(self, features: int) -> None:
        self.gram = np.zeros((features, features), dtype=np.float32)
        self.frames = 0

    def update(self, inputs: NDArray[np.float32]) -> None:
        self.gram += inputs.T @ inputs
        self.frames += len(inputs)

    def average(self) -> NDArray[np.float32]:
        if self.frames == 0:
            raise ValueError("No audio frames were collected")
        return self.gram / self.frames


def train_rows(manifest: Path) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or any(row["split"].split(".")[0].split("-")[0] != "train" for row in rows):
        raise ValueError("Calibration requires a nonempty manifest containing only train rows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate calibration IDs")
    return rows


def input_source(module_name: str) -> str:
    if module_name.endswith((".attn.key", ".attn.value")):
        return module_name.rsplit(".", 1)[0] + ".query"
    return module_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate encoder GPTQ and export native Q4_0 checkpoints")
    parser.add_argument("--f16", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--counts", required=True, type=int, nargs="+")
    parser.add_argument("--tensor", action="append", help="Exact GGML tensor name. Default: all encoder linears.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--damp", type=float, default=0.01)
    args = parser.parse_args()
    rows = train_rows(args.manifest)
    counts = sorted(set(args.counts))
    if counts[0] <= 0 or counts[-1] > len(rows):
        parser.error("--counts must be between one and the number of train rows")

    source = Model.read(args.f16)
    control = Model.read(args.control)
    if source.hparams[-1] != F16 or control.hparams[-1] % 1000 != Q4_0:
        raise ValueError("Expected a mostly-F16 source and a stock Q4_0 control")
    if source.hparams[:-1] != control.hparams[:-1] or source.tensors.keys() != control.tensors.keys():
        raise ValueError("The source and control describe different models")

    encoder = _load_encoder(source, args.threads)
    names = [f"encoder.{name}.weight" for name, module in encoder.named_modules()
             if _is_linear(module)]
    if args.tensor:
        invalid = set(args.tensor) - set(names)
        if invalid:
            raise ValueError(f"Not encoder linear weights: {sorted(invalid)}")
        names = list(dict.fromkeys(args.tensor))
    tensor_sources = {name: input_source(name.removeprefix("encoder.").removesuffix(".weight"))
                      for name in names}
    measurements = _register_hooks(encoder, tensor_sources.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"f16_sha256": _sha256(args.f16), "control_sha256": _sha256(args.control)}
    for index, row in enumerate(rows[:counts[-1]], start=1):
        _encode(encoder, args.manifest.parent / row["audio"])
        if index == 1 or index % 8 == 0:
            print(f"Encoded {index}/{counts[-1]} train utterances", flush=True)
        if index in counts:
            _export(args, source, control, tensor_sources, measurements, rows[:index], hashes)


def _load_encoder(source: Model, threads: int):
    import torch
    from whisper.model import AudioEncoder

    _, context, state, heads, layers, _, _, _, _, mel_bands, _ = source.hparams
    torch.set_num_threads(threads)
    encoder = AudioEncoder(mel_bands, context, state, heads, layers).eval()
    tensors = {}
    for name, parameter in encoder.state_dict().items():
        weights = source.weights("encoder." + name).reshape(tuple(parameter.shape))
        tensors[name] = torch.from_numpy(weights)
    encoder.load_state_dict(tensors)
    return encoder


def _is_linear(module) -> bool:
    from torch.nn import Linear

    return isinstance(module, Linear)


def _register_hooks(encoder, module_names) -> dict[str, Moments]:
    modules = dict(encoder.named_modules())
    moments = {}
    for name in dict.fromkeys(module_names):
        module = modules[name]
        measurement = Moments(module.in_features)
        module.register_forward_pre_hook(_input_hook(encoder, measurement))
        moments[name] = measurement
    return moments


def _input_hook(encoder, measurement: Moments):
    def collect(_module, inputs) -> None:
        valid_frames = encoder.valid_calibration_frames
        measurement.update(inputs[0][0, :valid_frames].detach().float().numpy())

    return collect


def _encode(encoder, path: Path) -> None:
    import torch
    from whisper.audio import HOP_LENGTH, load_audio, log_mel_spectrogram

    mel_frames = 2 * len(encoder.positional_embedding)
    sample_limit = mel_frames * HOP_LENGTH
    audio = load_audio(str(path))[:sample_limit]
    if not len(audio):
        raise ValueError(f"No audio samples at {path}")
    encoder.valid_calibration_frames = (len(audio) + 2 * HOP_LENGTH - 1) // (2 * HOP_LENGTH)
    with torch.inference_mode():
        mel = log_mel_spectrogram(audio, n_mels=encoder.conv1.in_channels, padding=sample_limit)
        encoder(mel[None, :, :mel_frames])


def _export(args, source, control, tensor_sources, measurements, rows, hashes) -> None:
    name = f"gptq-q4_0-{len(rows)}"
    destination = args.output_dir / f"{name}.bin"
    if destination.exists() and destination.samefile(source.path):
        raise ValueError("Source and destination must be distinct files")
    replacements = {}
    diagnostics = []
    for tensor, module in tensor_sources.items():
        weights = source.weights(tensor)
        stock = control.payload(tensor)
        if stock != reference_q4(weights).pack():
            raise ValueError(f"{tensor} does not match the reference Q4_0 of the chosen F16 weights")
        moment = measurements[module].average()
        optimized = gptq(weights, moment, args.damp)
        replacements[tensor] = optimized.pack()
        diagnostics.append({
            "tensor": tensor,
            "shape": list(weights.shape),
            "activation_frames": measurements[module].frames,
            "calibration_output_mse_stock": _output_mse(weights, Q4Blocks.read(stock, weights.shape), moment),
            "calibration_output_mse_gptq": _output_mse(weights, optimized, moment),
        })
    replace_q4_payloads(control, replacements, destination)
    metadata = {
        **hashes,
        "model_sha256": _sha256(destination),
        "damp": args.damp,
        "calibration": rows,
        "activation_mask": "encoder frames up to ceil(min(samples, 480000) / 320)",
        "activation_weights": "original F16 checkpoint in OpenAI Whisper encoder, float32 CPU",
        "tensors": diagnostics,
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"{destination}: {len(replacements)} encoder linear tensors", flush=True)


def _output_mse(original: NDArray, quantized: Q4Blocks, second_moment: NDArray) -> float:
    residual = original - quantized.dequantize()
    return float(np.einsum("ij,ij->", residual @ second_moment, residual) / len(residual))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
