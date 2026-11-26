import argparse
import json
import os
import platform
import re
import subprocess
import wave
from pathlib import Path
from time import perf_counter

NORMALIZATION = {
    "en": "OpenAI Whisper EnglishTextNormalizer",
    "ru": "OpenAI Whisper BasicTextNormalizer with ё→е",
}


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
            raise ValueError(f"Expected mono 16-bit 16 kHz WAV: {path}")
        return stream.getnframes() / stream.getframerate()


def score(pairs: list[tuple[str, str]], language: str) -> dict:
    from jiwer import process_characters, process_words

    refs, hyps = zip(*pairs)
    normalize = _normalizer(language)
    references = [normalize(text) for text in refs]
    hypotheses = [normalize(text) for text in hyps]
    word = process_words(references, hypotheses)
    char = process_characters(references, hypotheses)
    return {
        "wer": word.wer,
        "cer": char.cer,
        "reference_words": word.hits + word.substitutions + word.deletions,
        "word_errors": word.substitutions + word.insertions + word.deletions,
        "reference_chars": char.hits + char.substitutions + char.deletions,
        "char_errors": char.substitutions + char.insertions + char.deletions,
    }


def inference_flags(args: argparse.Namespace) -> list[str]:
    return ["-l", args.language, "-t", str(args.threads), "-ng", "-nf", "-nt",
            "-bs", str(args.beam), "-bo", "1", "-tp", "0", "-oj", "-ac", str(args.audio_context),
            "-nfa" if args.no_flash_attention else "-fa"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure matched whisper.cpp CPU runs on one ASR manifest")
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--language", required=True, choices=NORMALIZATION)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--audio-context", type=int, default=0, help="0 uses the full model context")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.threads, args.beam, args.chunk_size) <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("threads, beam, batch size and limit must be positive")
    if args.audio_context < 0:
        parser.error("audio context must be nonnegative")

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = rows[:args.limit]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        parser.error("A nonempty manifest with unique IDs is required")
    audio = [(args.manifest.parent / row["audio"]).resolve() for row in rows]
    audio_seconds = sum(map(duration, audio))
    if audio_seconds <= 0:
        parser.error("Manifest has no audio duration")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    labels = set()
    for specification in args.model:
        label, separator, value = specification.partition("=")
        if not separator or not value or not re.fullmatch(r"[a-zA-Z0-9_-]+", label) or label in labels:
            parser.error(f"Expected a unique LABEL=PATH: {specification}")
        labels.add(label)
        model = Path(value).resolve(strict=True)
        result = _measure(args, label, model, rows, audio, audio_seconds)
        results.append(result)
        print(f"{label}: WER {100 * result['wer']:.2f}% CER {100 * result['cer']:.2f}% "
              f"RTF {result['rtf']:.3f}  ({len(rows)} utterances)", flush=True)
    if "f16" in labels:
        baseline = next(result["wer"] for result in results if result["model"] == "f16")
        for result in results:
            result["delta_wer_pp_vs_f16"] = 100 * (result["wer"] - baseline)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


def _measure(args, label: str, model: Path, rows: list[dict], audio: list[Path], audio_seconds: float) -> dict:
    run = args.output / label
    run.mkdir()
    prefixes = [run / f"prediction-{index:05}" for index in range(len(audio))]
    chunks = [_run_chunk(args, model, audio, prefixes, start)
              for start in range(0, len(audio), args.chunk_size)]
    predictions = []
    for row, prefix in zip(rows, prefixes, strict=True):
        native = json.loads(prefix.with_suffix(".json").read_text())
        text = "".join(segment["text"] for segment in native["transcription"]).strip()
        predictions.append({**row, "prediction": text})
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    metrics = score([(row["text"], row["prediction"]) for row in predictions], args.language)
    wall_seconds = sum(chunk["wall_seconds"] for chunk in chunks)
    result = {
        "model": label,
        "model_path": str(model),
        "model_bytes": model.stat().st_size,
        "language": args.language,
        "manifest": str(args.manifest),
        "clips": len(rows),
        "threads": args.threads,
        "beam": args.beam,
        "audio_context": args.audio_context,
        "flash_attention": not args.no_flash_attention,
        "audio_seconds": audio_seconds,
        "wall_seconds_including_load": wall_seconds,
        "rtf": wall_seconds / audio_seconds,
        "peak_rss_bytes": max(chunk["peak_rss_bytes"] for chunk in chunks),
        "energy_joules": None,
        "platform": platform.platform(),
        "normalization": NORMALIZATION[args.language],
        "cer_includes_spaces": True,
        "chunks": chunks,
        **metrics,
    }
    (run / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _run_chunk(args, model: Path, audio: list[Path], prefixes: list[Path], start: int) -> dict:
    end = start + args.chunk_size
    command = [str(args.cli), "-m", str(model), *inference_flags(args)]
    for path, prefix in zip(audio[start:end], prefixes[start:end], strict=True):
        command.extend(["-f", str(path), "-of", str(prefix)])
    log = prefixes[0].parent / f"run-{start:05}.log"
    with log.open("w") as stream:
        start_time = perf_counter()
        with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT) as process:
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        elapsed = perf_counter() - start_time
    if process.returncode:
        raise RuntimeError(f"whisper-cli failed with status {process.returncode}. See {log}")
    rss_bytes = usage.ru_maxrss if platform.system() == "Darwin" else usage.ru_maxrss * 1024
    return {"wall_seconds": elapsed, "peak_rss_bytes": rss_bytes, "log": str(log), "command": command}


def _normalizer(language: str):
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    if language == "en":
        return EnglishTextNormalizer()
    normalize = BasicTextNormalizer()
    return lambda text: normalize(text).replace("ё", "е")


if __name__ == "__main__":
    main()
