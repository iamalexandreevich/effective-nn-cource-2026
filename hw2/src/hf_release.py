import json
import shutil
import sys
from pathlib import Path

HW2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW2_DIR / "kaggle"))

from launch_recalibration import MAIN_MAX_STEPS, TOKENS_PER_SEQUENCE, TRAIN_BATCH_SIZE  # noqa: E402

BASE_MODEL = "HuggingFaceTB/SmolLM-360M"
FINAL_STEP = 1000
PAPER_URL = "https://arxiv.org/abs/2512.12167"
AUTHORS_CODE_URL = "https://github.com/SakanaAI/DroPE"
COURSE_REPO_URL = "https://github.com/iamalexandreevich/effective-nn-cource-2026/tree/main/hw2"


def build_release(snapshot_dir, results_dir, release_dir):
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(snapshot_dir) / "model.safetensors", release_dir / "model.safetensors")
    config = json.loads((Path(snapshot_dir) / "config.json").read_text())
    config.pop("auto_map", None)
    (release_dir / "config.json").write_text(json.dumps(config, indent=2))
    (release_dir / "README.md").write_text(render_model_card(results_dir))
    return release_dir


def render_model_card(results_dir):
    results_dir = Path(results_dir)
    drope = {scale: load_json(results_dir / "eval" / f"drope_step{FINAL_STEP}_s{scale}.json") for scale in (1, 2, 4)}
    base_perplexity = load_json(results_dir / "eval" / "base_s1.json")["perplexity"]
    base_niah = load_json(results_dir / "eval" / "base_s1.json")["niah_accuracy"]
    quality = load_json(results_dir / "quality" / "drope_final.json")["average"]
    base_quality = load_json(results_dir / "quality" / "smollm360m_base.json")["average"]
    tokens_million = MAIN_MAX_STEPS * TRAIN_BATCH_SIZE * TOKENS_PER_SEQUENCE / 1e6
    rows = [
        ("1× (2048)", drope[1]["perplexity"], None, drope[1]["niah_accuracy"]),
        ("2× (4096)", drope[2]["perplexity"], drope[2]["beta_coefficient"], drope[2]["niah_accuracy"]),
        ("4× (8192)", drope[4]["perplexity"], drope[4]["beta_coefficient"], drope[4]["niah_accuracy"]),
    ]
    table = "\n".join(
        f"| {length} | {perplexity:.2f} | {'—' if coefficient is None else coefficient} | {100 * niah:.0f}% |"
        for length, perplexity, coefficient, niah in rows
    )
    return f"""---
license: apache-2.0
base_model: {BASE_MODEL}
language: en
tags:
- drope
- long-context
- positional-embeddings
- reproduction
inference: false
---

# SmolLM-360M-DroPE (short recalibration, coursework reproduction)

[SmolLM-360M]({"https://huggingface.co/" + BASE_MODEL}) with rotary position embeddings (RoPE) switched off in every layer and recalibrated for {tokens_million:.0f}M tokens, as proposed in *Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings* ([arXiv]({PAPER_URL})).

**This is not an official release and not a faithful reproduction of the paper's numbers.** It comes from a university coursework reproduction. The recalibration budget is about 0.2% of the paper's, and the model is clearly under-trained. Treat it as an intermediate checkpoint for studying DroPE, not as a drop-in replacement for SmolLM-360M.

## Results (this checkpoint)

Perplexity on held-out fineweb-edu text and needle-in-a-haystack (NIAH) accuracy, 100 trials per row. `c` is the attention-temperature coefficient (logits are scaled by β = 1 + c·ln(s), s is the context multiple), tuned on separate held-out text.

| Context | Perplexity | c | NIAH |
|---|---|---|---|
{table}

For reference, the original SmolLM-360M has perplexity {base_perplexity:.2f} and NIAH {100 * base_niah:.0f}% at 2048 tokens. Average over ARC-E, ARC-C, HellaSwag, OpenBookQA, PIQA and Winogrande (lm-eval-harness, zero-shot): **{quality:.1f}** for this checkpoint versus **{base_quality:.1f}** for the original model.

In short: perplexity does not blow up at 2× context once the temperature is tuned, but the model has not yet recovered the base model's quality or its ability to retrieve a number from context. The paper's claim that DroPE beats YaRN on NIAH could **not** be confirmed at this budget.

## Training

- Start from {BASE_MODEL}; every attention layer without RoPE and with QKNorm (RMSNorm on queries and keys), as in the authors' code.
- {MAIN_MAX_STEPS} steps × {TRAIN_BATCH_SIZE * TOKENS_PER_SEQUENCE} tokens, sequence length {TOKENS_PER_SEQUENCE}, AdamW (lr 1e-3, cosine, 4% warmup, weight decay 0.1, betas 0.9/0.95), fp16 on 2×T4 with fp32 master weights.
- Data: first four files of `PrimeIntellect/fineweb-edu` (`data0`).

## How to load

The weights use the authors' `DroPELlamaForCausalLM` class with `attention_type="qk_norm_nope"`. The model code is not included here; use the loader from the coursework repository ({COURSE_REPO_URL}), which wraps the authors' code ({AUTHORS_CODE_URL}) and fixes a KV-cache incompatibility with `transformers` 4.56.2:

```python
import math
from positional import load_drope_model, set_attention_temperature

model = load_drope_model("path/to/this/checkpoint", beta=1.0, device="cuda")
set_attention_temperature(model, 1 + 0.2 * math.log(2))  # 2x context
```

## Citation

```
@article{{gelberg2025drope,
  title   = {{Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings}},
  author  = {{Gelberg, Yoav and Eguchi, Koshi and Akiba, Takuya and Cetin, Edoardo}},
  journal = {{arXiv preprint arXiv:2512.12167}},
  year    = {{2025}}
}}
```
"""


def load_json(path):
    return json.loads(path.read_text())
