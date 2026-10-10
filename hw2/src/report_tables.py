import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HW2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW2_DIR / "kaggle"))

from launch_recalibration import MAIN_MAX_STEPS, TOKENS_PER_SEQUENCE, TRAIN_BATCH_SIZE  # noqa: E402

PENDING = "_Результаты рекалибровки ещё не посчитаны._"
PAPER_TOKENS = 31.46e9
# Table 10 статьи (NIAH, SmolLM-360M)
PAPER_NIAH = {"ntk": {2: 29.84, 4: 14.37}, "yarn": {2: 48.25, 4: 25.62}, "drope": {2: 74.92, 4: 55.0}}
# Table 5 статьи (lighteval, SmolLM-360M и SmolLM-360M-DroPE)
PAPER_QUALITY = {
    "SmolLM-360M": [65.6, 36.0, 53.8, 37.2, 72.0, 53.7, 53.1],
    "SmolLM-360M-DroPE": [67.3, 37.6, 53.9, 38.0, 71.5, 52.3, 53.4],
}
QUALITY_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "openbookqa", "piqa", "winogrande")
QUALITY_HEADERS = ("ARC-E", "ARC-C", "HellaSwag", "OpenBookQA", "PIQA", "Winogrande", "Среднее")
METHOD_LABELS = {
    "base": "без масштабирования",
    "crop": "обрезка до 2048",
    "pi": "PI",
    "ntk": "NTK (статический, формула статьи)",
    "ntk_dynamic": "NTK (динамический, HF)",
    "yarn": "YaRN",
}
FIG4_LAYERS = (1, 12, 24)
LONGBENCH_DATASETS = ("multifieldqa_en", "musique", "gov_report", "lcc")
LONGBENCH_HEADERS = ("MultiFieldQA", "MuSiQue", "GovReport", "LCC")
# Table 2 статьи (SmolLM-360M, длина до 80×)
PAPER_LONGBENCH = {
    "SmolLM": [4.03, 0.4, 4.48, 5.99],
    "+ PI": [13.68, 2.45, 5.67, 11.52],
    "+ RoPE-NTK": [18.87, 4.89, 23.71, 8.26],
    "+ YaRN": [20.78, 4.77, 15.03, 10.87],
    "SmolLM-DroPE": [29.33, 7.93, 21.87, 18.56],
}
LONGBENCH_RUNS = (
    ("base_s1", "без масштабирования, обрезка до 2048"),
    ("ntk_s4", "NTK (статический), 4×"),
    ("ntk_dynamic_s4", "NTK (динамический), 4×"),
    ("yarn_s4", "YaRN, 4×"),
    ("drope_step1000_s4", "DroPE (step-1000), 4×"),
)


def main():
    args = parse_args()
    template = (args.report_dir / "report_template.md").read_text()
    report = fill_markers(template, build_tables(args.results_dir))
    (args.report_dir / "report.md").write_text(report)


def build_tables(results_dir):
    return {
        "baselines": baselines_table(results_dir),
        "niah_vs_paper": niah_vs_paper_table(results_dir),
        "niah_by_depth": niah_by_depth_table(results_dir),
        "quality": quality_table(results_dir),
        "fig4": fig4_table(results_dir),
        "budget": budget_table(results_dir),
        "drope_curve": drope_curve_table(results_dir),
        "beta_sweep": beta_sweep_table(results_dir),
        "longbench": longbench_table(results_dir),
        "longbench_empty": longbench_empty_table(results_dir),
    }


def fill_markers(template, tables):
    return re.sub(r"<!-- table:(\w+) -->", lambda match: tables[match.group(1)], template)


def baselines_table(results_dir):
    rows = []
    for method, label in METHOD_LABELS.items():
        cells = [label]
        for scale in (1, 2, 4):
            path = results_dir / "eval" / f"{method}_s{scale}.json"
            if path.exists():
                result = load_json(path)
                cells += [f"{result['perplexity']:.2f}", percent(result["niah_accuracy"])]
            else:
                cells += ["", ""]
        rows.append(cells)
    return markdown_table(("Метод", "PPL 1×", "NIAH 1×, %", "PPL 2×", "NIAH 2×, %", "PPL 4×", "NIAH 4×, %"), rows)


def niah_vs_paper_table(results_dir):
    rows = []
    for method in ("ntk", "yarn"):
        cells = [METHOD_LABELS[method]]
        for scale in (2, 4):
            ours = percent(load_eval(results_dir, method, scale)["niah_accuracy"])
            cells += [f"{PAPER_NIAH[method][scale]:.1f}", ours]
        rows.append(cells)
    return markdown_table(("Метод", "2×, статья", "2×, у нас", "4×, статья", "4×, у нас"), rows)


def niah_by_depth_table(results_dir):
    rows = []
    for scale in (2, 4):
        for method in ("crop", "yarn"):
            by_depth = load_eval(results_dir, method, scale)["niah_accuracy_by_depth"]
            rows.append([f"{scale}×", METHOD_LABELS[method], *[f"{100 * value:.0f}" for value in by_depth.values()]])
    depths = [f"{float(depth):.2f}" for depth in load_eval(results_dir, "crop", 2)["niah_accuracy_by_depth"]]
    return markdown_table(("Контекст", "Метод", *depths), rows)


def quality_table(results_dir):
    rows = [[f"{name} (статья)", *[f"{value:.1f}" for value in values]] for name, values in PAPER_QUALITY.items()]
    path = results_dir / "quality" / "smollm360m_base.json"
    if path.exists():
        rows.append(quality_row("SmolLM-360M (у нас)", json.loads(path.read_text())))
    drope_path = results_dir / "quality" / "drope_final.json"
    if drope_path.exists():
        rows.append(quality_row("DroPE (у нас)", json.loads(drope_path.read_text())))
    return markdown_table(("Модель", *QUALITY_HEADERS), rows)


def quality_row(label, result):
    return [label, *[f"{result['scores'][task]:.1f}" for task in QUALITY_TASKS], f"{result['average']:.1f}"]


def fig4_table(results_dir):
    norms = json.loads((results_dir / "analysis" / "fig4_bias_gradients.json").read_text())["norms"]
    titles = {"diagonal": "диагональная голова", "previous_token": "голова предыдущего токена"}
    rows = []
    for target, title in titles.items():
        for name in ("rope", "nope"):
            mean_by_layer = [statistics.mean(values) for values in zip(*norms[target][name])]
            rows.append([title, name.upper(), *[f"{mean_by_layer[layer - 1] * 1000:.2f}" for layer in FIG4_LAYERS]])
    return markdown_table(("Голова", "Модель", *[f"слой {layer}" for layer in FIG4_LAYERS]), rows)


def budget_table(results_dir):
    smoke = json.loads((results_dir / "kaggle" / "smoke_summary_ddp.json").read_text())["attempts"][0]
    benchmark = json.loads((results_dir / "kaggle" / "gpu_benchmark.json").read_text())["gpu"][0]
    tokens = MAIN_MAX_STEPS * TRAIN_BATCH_SIZE * TOKENS_PER_SEQUENCE
    rows = [
        ["Токенов рекалибровки у нас", f"{tokens / 1e6:.0f} млн"],
        ["Токенов рекалибровки в статье", f"{PAPER_TOKENS / 1e9:.2f} млрд"],
        ["Доля бюджета статьи", f"{100 * tokens / PAPER_TOKENS:.2f}%"],
        ["Шагов оптимизатора", str(MAIN_MAX_STEPS)],
        ["Токенов на шаг", f"{TRAIN_BATCH_SIZE * TOKENS_PER_SEQUENCE}"],
        ["Trainer, 2×T4 (DDP), токенов/с суммарно", f"{smoke['tokens_per_second']:.0f}"],
        ["Trainer, 2×T4 (DDP), токенов/с на карту", f"{smoke['tokens_per_second'] / smoke['process_count']:.0f}"],
        ["Ручной цикл, одна T4, токенов/с", f"{benchmark['tokens_per_second_one_gpu']:.0f}"],
    ]
    return markdown_table(("Показатель", "Значение"), rows)


def drope_curve_table(results_dir):
    steps = sorted(int(re.fullmatch(r"drope_step(\d+)_s1", path.stem).group(1)) for path in (results_dir / "eval").glob("drope_step*_s1.json"))
    if not steps:
        return PENDING
    rows = []
    for step in steps:
        cells = [str(step), f"{step * TRAIN_BATCH_SIZE * TOKENS_PER_SEQUENCE / 1e6:.0f}"]
        cells.append(f"{load_json(results_dir / 'eval' / f'drope_step{step}_s1.json')['perplexity']:.2f}")
        for scale in (2, 4):
            result = load_json(results_dir / "eval" / f"drope_step{step}_s{scale}.json")
            cells += [f"{result['beta_coefficient']:g}", f"{result['perplexity']:.2f}", percent(result["niah_accuracy"])]
        rows.append(cells)
    headers = ("Шаг", "Токенов, млн", "PPL 1×", "c (2×)", "PPL 2×", "NIAH 2×, %", "c (4×)", "PPL 4×", "NIAH 4×, %")
    return markdown_table(headers, rows)


def beta_sweep_table(results_dir):
    paths = sorted((results_dir / "eval").glob("beta_step*.json"), key=lambda path: int(re.fullmatch(r"beta_step(\d+)", path.stem).group(1)))
    if not paths:
        return PENDING
    sweep = load_json(paths[-1])["perplexity"]
    rows = [[coefficient, f"{sweep['2.0'][coefficient]:.2f}", f"{sweep['4.0'][coefficient]:.2f}"] for coefficient in sweep["2.0"]]
    return markdown_table(("c", "PPL 2×", "PPL 4×"), rows)


def longbench_table(results_dir):
    rows = [[f"{name} (статья)", *[f"{score:.2f}" for score in scores], f"{statistics.mean(scores):.1f}"] for name, scores in PAPER_LONGBENCH.items()]
    ours = []
    for name, label in LONGBENCH_RUNS:
        path = results_dir / "longbench" / f"{name}.json"
        if not path.exists():
            continue
        datasets = load_json(path)["datasets"]
        if not all(dataset in datasets for dataset in LONGBENCH_DATASETS):
            continue
        scores = [datasets[dataset]["score"] for dataset in LONGBENCH_DATASETS]
        ours.append([f"{label} (у нас)", *[f"{score:.2f}" for score in scores], f"{statistics.mean(scores):.1f}"])
    if not ours:
        return PENDING
    return markdown_table(("Метод", *LONGBENCH_HEADERS, "Среднее"), rows + ours)


def longbench_empty_table(results_dir):
    rows = []
    for name, label in LONGBENCH_RUNS:
        path = results_dir / "longbench" / f"{name}.json"
        if not path.exists():
            continue
        datasets = load_json(path)["datasets"]
        if not all(dataset in datasets for dataset in LONGBENCH_DATASETS):
            continue
        cells = []
        for dataset in LONGBENCH_DATASETS:
            predictions = datasets[dataset]["predictions"]
            empty = sum(1 for entry in predictions if not entry["prediction"].strip())
            cells.append(f"{empty}/{len(predictions)}")
        rows.append([label, *cells])
    if not rows:
        return PENDING
    return markdown_table(("Метод", *LONGBENCH_HEADERS), rows)


def load_eval(results_dir, method, scale):
    return load_json(results_dir / "eval" / f"{method}_s{scale}.json")


def load_json(path):
    return json.loads(path.read_text())


def percent(fraction):
    return f"{100 * fraction:.1f}"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HW2_DIR / "results")
    parser.add_argument("--report-dir", type=Path, default=HW2_DIR / "report")
    return parser.parse_args()


if __name__ == "__main__":
    main()
