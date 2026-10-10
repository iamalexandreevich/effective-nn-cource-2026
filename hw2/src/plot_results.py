import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HW2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW2_DIR / "kaggle"))

from launch_recalibration import TOKENS_PER_SEQUENCE, TRAIN_BATCH_SIZE  # noqa: E402

RESULTS_DIR = HW2_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "analysis"
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * TOKENS_PER_SEQUENCE
SMOOTHING_WINDOW = 20
REFERENCE_METHODS = {"base": "без масштабирования", "crop": "обрезка до 2048", "ntk_dynamic": "NTK (динамический)", "yarn": "YaRN"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_loss()
    plot_recovery_curve()


def plot_training_loss():
    log = json.loads((RESULTS_DIR / "kaggle" / "main_train_log.json").read_text())
    steps = [entry["step"] for entry in log]
    losses = [entry["loss"] for entry in log]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([step * TOKENS_PER_STEP / 1e6 for step in steps], losses, alpha=0.3, label="loss на шаге")
    smoothed = [sum(losses[max(0, i - SMOOTHING_WINDOW + 1) : i + 1]) / len(losses[max(0, i - SMOOTHING_WINDOW + 1) : i + 1]) for i in range(len(losses))]
    axis.plot([step * TOKENS_PER_STEP / 1e6 for step in steps], smoothed, label=f"скользящее среднее ({SMOOTHING_WINDOW} шагов)")
    axis.set_xlabel("токенов рекалибровки, млн")
    axis.set_ylabel("loss")
    axis.set_title("Обучение DroPE (SmolLM-360M)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "train_loss.png", dpi=150)


def plot_recovery_curve():
    steps = sorted(int(re.fullmatch(r"drope_step(\d+)_s1", path.stem).group(1)) for path in (RESULTS_DIR / "eval").glob("drope_step*_s1.json"))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, scale in zip(axes, (2, 4)):
        tokens = [step * TOKENS_PER_STEP / 1e6 for step in steps]
        values = [json.loads((RESULTS_DIR / "eval" / f"drope_step{step}_s{scale}.json").read_text())["perplexity"] for step in steps]
        axis.plot(tokens, values, marker="o", color="black", linewidth=2, label="DroPE, c подобран")
        for method, label in REFERENCE_METHODS.items():
            perplexity = json.loads((RESULTS_DIR / "eval" / f"{method}_s{scale}.json").read_text())["perplexity"]
            axis.axhline(perplexity, linestyle="--", linewidth=1, label=label, color=reference_color(method))
        axis.set_yscale("log")
        axis.set_xlabel("токенов рекалибровки, млн")
        axis.set_title(f"Контекст {scale}× ({scale * 2048} токенов)")
    axes[0].set_ylabel("перплексия (log)")
    axes[1].legend(fontsize=8)
    figure.suptitle("Восстановление DroPE по числу токенов и базовые методы")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "recovery_curve.png", dpi=150)


def reference_color(method):
    return {"base": "tab:red", "crop": "tab:gray", "ntk_dynamic": "tab:orange", "yarn": "tab:blue"}[method]


if __name__ == "__main__":
    main()
