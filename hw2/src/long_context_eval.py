import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from niah import evaluate_niah, load_haystack_ids, make_trials
from positional import BASE_MODEL, ROPE_METHODS, TRAIN_CONTEXT, drope_temperature, load_drope_model, load_rope_model
from ppl import load_eval_sequences, perplexity

HW2_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = HW2_DIR / "artifacts"
DEPTHS = tuple(index / 9 for index in range(10))
NIAH_SEED = 2026


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    haystack_ids = load_haystack_ids(tokenizer, ARTIFACTS_DIR / "haystack" / "paul_graham.txt")
    for scale in args.scales:
        result = evaluate_scale(args, scale, tokenizer, haystack_ids, device)
        write_result(args.output_dir / f"{args.name}_s{scale:g}.json", result)


def evaluate_scale(args, scale, tokenizer, haystack_ids, device):
    context_length = int(TRAIN_CONTEXT * scale)
    window = TRAIN_CONTEXT if args.method == "crop" else None
    beta = drope_temperature(scale, args.beta_coefficient) if args.method == "drope" else None
    model = load_model(args, scale, beta, device)
    started = time.time()
    sequences = load_eval_sequences(
        tokenizer, context_length, args.ppl_sequences, ARTIFACTS_DIR / "eval" / f"fineweb_edu_{context_length}.pt"
    )
    trials = make_trials(DEPTHS, args.niah_trials_per_depth, len(haystack_ids), context_length, seed=NIAH_SEED)
    niah_results = evaluate_niah(model, tokenizer, haystack_ids, trials, context_length, device, window=window)
    return {
        "name": args.name,
        "method": args.method,
        "scale": scale,
        "context_length": context_length,
        "checkpoint": str(args.checkpoint) if args.checkpoint else BASE_MODEL,
        "beta": beta,
        "beta_coefficient": args.beta_coefficient if args.method == "drope" else None,
        "dtype": "float32",
        "device": device,
        "perplexity": perplexity(model, sequences, device, window=window),
        "ppl_sequences": args.ppl_sequences,
        "niah_accuracy": sum(result["correct"] for result in niah_results) / len(niah_results),
        "niah_accuracy_by_depth": accuracy_by_depth(niah_results),
        "niah_trials": niah_results,
        "elapsed_minutes": (time.time() - started) / 60,
    }


def load_model(args, scale, beta, device):
    if args.method == "drope":
        return load_drope_model(args.checkpoint, beta, device)
    rope_method = "base" if args.method == "crop" else args.method
    return load_rope_model(rope_method, scale, device)


def accuracy_by_depth(niah_results):
    outcomes = defaultdict(list)
    for result in niah_results:
        outcomes[round(result["depth"], 3)].append(result["correct"])
    return {str(depth): sum(values) / len(values) for depth, values in sorted(outcomes.items())}


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{path.name}: ppl={result['perplexity']:.3f} niah={result['niah_accuracy']:.3f} ({result['elapsed_minutes']:.1f} min)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=(*ROPE_METHODS, "crop", "drope"), required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--name")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--beta-coefficient", type=float, default=0.0)
    parser.add_argument("--niah-trials-per-depth", type=int, default=10)
    parser.add_argument("--ppl-sequences", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=HW2_DIR / "results" / "eval")
    args = parser.parse_args()
    if args.method == "drope" and args.checkpoint is None:
        parser.error("--checkpoint обязателен для drope")
    args.name = args.name or args.method
    return args


if __name__ == "__main__":
    main()
