import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from longbench import DATASETS, build_prompt_ids, load_samples, load_task_config, mean_score
from positional import BASE_MODEL, ROPE_METHODS, TRAIN_CONTEXT, drope_temperature, load_drope_model, load_rope_model

HW2_DIR = Path(__file__).resolve().parents[1]
SAMPLING_SEED = 2026


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    max_length = int(TRAIN_CONTEXT * args.scale)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = load_model(args, device, dtype)
    prompts, max_generation = load_task_config("dataset2prompt"), load_task_config("dataset2maxlen")
    result = load_existing(args.output)
    result.update(describe_run(args, max_length, dtype, device))
    for dataset in args.datasets:
        if dataset in result["datasets"]:
            continue
        result["datasets"][dataset] = evaluate_dataset(
            model, tokenizer, device, dataset, prompts[dataset], max_generation[dataset], max_length, args.samples
        )
        write_result(args.output, result)
        print(f"{args.name} {dataset}: {result['datasets'][dataset]['score']}", flush=True)


def evaluate_dataset(model, tokenizer, device, dataset, prompt_format, max_generation, max_length, samples_count):
    started = time.time()
    samples = load_samples(dataset, samples_count, SAMPLING_SEED)
    predictions = [generate_answer(model, tokenizer, device, sample, prompt_format, max_generation, max_length) for sample in samples]
    answers = [sample["answers"] for sample in samples]
    return {
        "score": mean_score(dataset, predictions, answers),
        "samples": len(samples),
        "max_generation": max_generation,
        "elapsed_minutes": (time.time() - started) / 60,
        "predictions": [{"prediction": prediction, "answers": reference} for prediction, reference in zip(predictions, answers)],
    }


def generate_answer(model, tokenizer, device, sample, prompt_format, max_generation, max_length):
    prompt_ids = build_prompt_ids(tokenizer, prompt_format, sample, max_length)
    input_ids = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_generation,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, len(prompt_ids) :], skip_special_tokens=True)


def load_model(args, device, dtype):
    if args.method == "drope":
        return load_drope_model(args.checkpoint, drope_temperature(args.scale, args.beta_coefficient), device, dtype)
    return load_rope_model(args.method, args.scale, device, dtype)


def describe_run(args, max_length, dtype, device):
    return {
        "name": args.name,
        "method": args.method,
        "scale": args.scale,
        "max_length": max_length,
        "beta_coefficient": args.beta_coefficient if args.method == "drope" else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else BASE_MODEL,
        "dtype": str(dtype),
        "device": device,
        "sampling_seed": SAMPLING_SEED,
        "samples_per_dataset": args.samples,
    }


def load_existing(path):
    if path.exists():
        return json.loads(path.read_text())
    return {"datasets": {}}


def write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=(*ROPE_METHODS, "drope"), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--beta-coefficient", type=float, default=0.0)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=HW2_DIR / "results" / "longbench")
    args = parser.parse_args()
    if args.method == "drope" and args.checkpoint is None:
        parser.error("--checkpoint обязателен для drope")
    args.output = args.output_dir / f"{args.name}.json"
    return args


if __name__ == "__main__":
    main()
