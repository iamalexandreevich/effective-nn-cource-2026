import json
import sys
import time
from pathlib import Path

import torch

HW2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW2_DIR / "third_party" / "DroPE"))

from custom_data.pretraining_data import load_pretraining_dataset  # noqa: E402
from custom_models.drope import DroPELlamaConfig, DroPELlamaForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM-360M"
SEQUENCE_LENGTH = 2048
OUTPUT = Path("/kaggle/working/benchmark.json")
GPU_CASES = (
    dict(batch_size=2, checkpointing=True, optimizer="adamw"),
    dict(batch_size=2, checkpointing=False, optimizer="adamw"),
    dict(batch_size=4, checkpointing=True, optimizer="adamw_8bit"),
    dict(batch_size=8, checkpointing=True, optimizer="adamw_8bit"),
    dict(batch_size=4, checkpointing=False, optimizer="adamw_8bit"),
)


def main():
    results = {"gpu": []}
    for case in GPU_CASES:
        results["gpu"].append({**case, **measure_gpu_case(**case)})
        report(results)


def measure_data_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    datasets = load_pretraining_dataset(
        tokenizer=tokenizer,
        dataset_id_or_path="PrimeIntellect/fineweb-edu",
        dataset_local_directory="PrimeIntellect/fineweb-edu",
        max_seq_length=SEQUENCE_LENGTH,
        manually_add_eos=True,
        streaming=True,
        do_train=True,
        do_eval=False,
        load_dataset_kwargs={"data_files": "data0/*.parquet"},
        trust_remote_code=True,
    )
    results = {}
    for workers in (0, 2):
        loader = torch.utils.data.DataLoader(
            datasets["train_dataset"], batch_size=2, num_workers=workers, collate_fn=datasets["data_collator"]
        )
        iterator = iter(loader)
        next(iterator)
        started = time.time()
        batches = 32
        for _ in range(batches):
            next(iterator)
        elapsed = time.time() - started
        results[f"workers_{workers}_tokens_per_second"] = batches * 2 * SEQUENCE_LENGTH / elapsed
    return results


def measure_gpu_case(batch_size, checkpointing, optimizer, steps=6):
    model = load_model()
    if checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    torch.cuda.reset_peak_memory_stats()
    try:
        seconds = time_steps(model, build_optimizer(model, optimizer), batch_size, steps)
    except torch.OutOfMemoryError:
        return {"out_of_memory": True}
    finally:
        del model
        torch.cuda.empty_cache()
    return {
        "seconds_per_micro_batch": seconds,
        "tokens_per_second_one_gpu": batch_size * SEQUENCE_LENGTH / seconds,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def time_steps(model, optimizer, batch_size, steps):
    scaler = torch.amp.GradScaler()
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, SEQUENCE_LENGTH), device="cuda")
    for step in range(steps + 2):
        if step == 2:
            torch.cuda.synchronize()
            started = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids, labels=input_ids).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.time() - started) / steps


def load_model():
    config = DroPELlamaConfig.from_pretrained(MODEL, attention_type="qk_norm_nope")
    model = DroPELlamaForCausalLM.from_pretrained(MODEL, config=config, attn_implementation="sdpa", dtype=torch.float32)
    return model.cuda().train()


def build_optimizer(model, name):
    if name == "adamw_8bit":
        import bitsandbytes

        return bitsandbytes.optim.AdamW8bit(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    return torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)


def report(results):
    OUTPUT.write_text(json.dumps(results, indent=2))
    print("BENCH " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
