import argparse
import json
from pathlib import Path

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from positional import BASE_MODEL, load_drope_model, load_rope_model

TASKS = ("arc_easy", "arc_challenge", "hellaswag", "openbookqa", "piqa", "winogrande")
# Как в отчётах SmolLM: нормированная точность там, где она есть, для winogrande обычная
REPORTED_METRIC = {task: "acc_norm,none" for task in TASKS} | {"winogrande": "acc,none"}


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_drope_model(args.checkpoint, beta=1.0, device=device) if args.checkpoint else load_rope_model("base", 1.0, device)
    harness_model = HFLM(pretrained=model, tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL), batch_size=args.batch_size)
    output = simple_evaluate(model=harness_model, tasks=list(TASKS), num_fewshot=0, limit=args.limit, random_seed=0)
    scores = {task: 100 * output["results"][task][REPORTED_METRIC[task]] for task in TASKS}
    result = {
        "name": args.name,
        "checkpoint": str(args.checkpoint) if args.checkpoint else BASE_MODEL,
        "limit": args.limit,
        "scores": scores,
        "average": sum(scores.values()) / len(scores),
        "raw_results": output["results"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({"scores": scores, "average": result["average"]}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
