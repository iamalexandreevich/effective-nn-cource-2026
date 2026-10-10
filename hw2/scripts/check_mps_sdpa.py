import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

HW2_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HW2_DIR / "src"), str(HW2_DIR / "third_party" / "DroPE")]

from longbench import load_samples, load_task_config  # noqa: E402
from longbench_eval import generate_answer  # noqa: E402
from positional import BASE_MODEL, load_rope_model  # noqa: E402

DATASET = "gov_report"
SAMPLES = 6
MAX_GENERATION = 40
MAX_LENGTH = 2048
SAMPLING_SEED = 2026
OUTPUT = HW2_DIR / "results" / "longbench" / "mps_sdpa_check.json"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    prompt_format = load_task_config("dataset2prompt")[DATASET]
    samples = load_samples(DATASET, 60, SAMPLING_SEED)[:SAMPLES]
    runs = {}
    for implementation in ("sdpa", "eager"):
        model = load_rope_model("base", 1.0, "mps", torch.bfloat16)
        model.config._attn_implementation = implementation
        answers = [generate_answer(model, tokenizer, "mps", sample, prompt_format, MAX_GENERATION, MAX_LENGTH) for sample in samples]
        runs[implementation] = {"empty_answers": sum(1 for answer in answers if not answer.strip()), "answers": answers}
        del model
    result = {"dataset": DATASET, "samples": SAMPLES, "dtype": "bfloat16", "device": "mps", "max_length": MAX_LENGTH, "runs": runs}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print({name: run["empty_answers"] for name, run in runs.items()})


if __name__ == "__main__":
    main()
