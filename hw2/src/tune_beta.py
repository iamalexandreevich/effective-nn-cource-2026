import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from positional import BASE_MODEL, TRAIN_CONTEXT, drope_temperature, load_drope_model, set_attention_temperature
from ppl import load_eval_sequences, perplexity

HW2_DIR = Path(__file__).resolve().parents[1]
# 0.103 и 0.412 — значения авторов для SmolLM-DroPE и для модели с нуля
DEFAULT_COEFFICIENTS = (0.0, 0.103, 0.2, 0.412, 0.8)
# Итоговая перплексия считается на первых токенах потока отложенных данных, для подбора берём токены дальше
TUNE_SKIP_TOKENS = 200_000


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_drope_model(args.checkpoint, beta=1.0, device=device)
    sweep = {}
    for scale in args.scales:
        context_length = int(TRAIN_CONTEXT * scale)
        sequences = load_eval_sequences(
            tokenizer, context_length, args.ppl_sequences,
            HW2_DIR / "artifacts" / "eval" / f"fineweb_edu_{context_length}_x{args.ppl_sequences}_tune_disjoint.pt",
            skip_tokens=TUNE_SKIP_TOKENS,
        )
        sweep[str(scale)] = perplexity_by_coefficient(model, sequences, scale, args.coefficients, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "ppl_sequences": args.ppl_sequences,
        "perplexity": sweep,
        "best_coefficient": {scale: min(values, key=values.get) for scale, values in sweep.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["best_coefficient"]))


def perplexity_by_coefficient(model, sequences, scale, coefficients, device):
    values = {}
    for coefficient in coefficients:
        set_attention_temperature(model, drope_temperature(scale, coefficient))
        values[str(coefficient)] = perplexity(model, sequences, device)
        print(f"s={scale:g} c={coefficient}: ppl={values[str(coefficient)]:.3f}", flush=True)
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=(2.0, 4.0))
    parser.add_argument("--coefficients", type=float, nargs="+", default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--ppl-sequences", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
