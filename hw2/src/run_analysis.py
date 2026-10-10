import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from custom_models.drope import DroPEQwen2Config, DroPEQwen2ForCausalLM
from transformers import AutoConfig, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from analysis import bias_gradient_norms, diagonal_weights, previous_token_weights
from positional import BASE_MODEL, frequency_multipliers
from ppl import load_eval_sequences

HW2_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = HW2_DIR / "results" / "analysis"
FIG6_SCALE = 2.0
FIG4_CONTEXT = 1024
FIG4_BATCH = 2
FIG4_SEEDS = (0, 1, 2)
# Модель «с нуля» из Table 4 статьи; словарь SmolLM вместо Qwen2, чтобы взять готовые токены fineweb-edu
FIG4_MODEL_CONFIG = dict(
    vocab_size=49152,
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    rope_theta=1_000_000.0,
    max_position_embeddings=FIG4_CONTEXT,
    attn_implementation="eager",
)
BIAS_TARGETS = {"diagonal": diagonal_weights, "previous_token": previous_token_weights}


def main():
    figure = parse_figure()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if figure == "fig6":
        make_fig6()
    else:
        make_fig4()


def make_fig6():
    config = AutoConfig.from_pretrained(BASE_MODEL)
    multipliers = {method: frequency_multipliers(config, method, FIG6_SCALE).tolist() for method in ("pi", "ntk", "yarn")}
    write_json(RESULTS_DIR / "fig6_frequency_scaling.json", {"scale": FIG6_SCALE, "model": BASE_MODEL, "gamma": multipliers})
    figure, axis = plt.subplots(figsize=(6, 4))
    for method, values in multipliers.items():
        axis.plot(range(len(values)), values, marker="o", markersize=3, label=method.upper())
    axis.set_xlabel("индекс частоты m")
    axis.set_ylabel("γ_m")
    axis.set_title(f"Масштабирование частот RoPE при s = {FIG6_SCALE:g} ({BASE_MODEL.split('/')[-1]})")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "fig6_frequency_scaling.png", dpi=150)


def make_fig4():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    input_ids = load_eval_sequences(
        tokenizer, FIG4_CONTEXT, FIG4_BATCH, HW2_DIR / "artifacts" / "eval" / f"fineweb_edu_{FIG4_CONTEXT}_x{FIG4_BATCH}.pt"
    )
    norms = {target: {"rope": [], "nope": []} for target in BIAS_TARGETS}
    for seed in FIG4_SEEDS:
        for name, model in initialized_models(seed).items():
            for target, build_weights in BIAS_TARGETS.items():
                norms[target][name].append(bias_gradient_norms(model, input_ids, build_weights(FIG4_CONTEXT)))
    write_json(
        RESULTS_DIR / "fig4_bias_gradients.json",
        {"config": FIG4_MODEL_CONFIG, "context": FIG4_CONTEXT, "batch": FIG4_BATCH, "seeds": FIG4_SEEDS, "norms": norms},
    )
    plot_fig4(norms)


def initialized_models(seed):
    torch.manual_seed(seed)
    rope = Qwen2ForCausalLM(Qwen2Config(**FIG4_MODEL_CONFIG))
    nope = DroPEQwen2ForCausalLM(DroPEQwen2Config(attention_type="nope", **FIG4_MODEL_CONFIG))
    nope.load_state_dict(rope.state_dict())
    return {"rope": rope, "nope": nope}


def plot_fig4(norms):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    titles = {"diagonal": "(a) диагональная голова", "previous_token": "(b) голова предыдущего токена"}
    for axis, (target, by_model) in zip(axes, norms.items()):
        for name, runs in by_model.items():
            values = torch.tensor(runs)
            layers = range(1, values.shape[1] + 1)
            axis.plot(layers, values.mean(0), marker="o", markersize=3, label=name.upper())
            axis.fill_between(layers, values.min(0).values, values.max(0).values, alpha=0.2)
        axis.set_yscale("log")
        axis.set_xlabel("слой")
        axis.set_ylabel("‖∂A^c / ∂(W_Q, W_K)‖")
        axis.set_title(titles[target])
        axis.legend()
    figure.suptitle("Градиент позиционного смещения на инициализации")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "fig4_bias_gradients.png", dpi=150)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_figure():
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=("fig4", "fig6"))
    return parser.parse_args().figure


if __name__ == "__main__":
    main()
