import json

from report_tables import PENDING, beta_sweep_table, drope_curve_table, fill_markers, longbench_empty_table, longbench_table, markdown_table, percent


def write_drope_result(results_dir, step, scale, perplexity, niah_accuracy, coefficient):
    eval_dir = results_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    result = {"perplexity": perplexity, "niah_accuracy": niah_accuracy, "beta_coefficient": coefficient}
    (eval_dir / f"drope_step{step}_s{scale}.json").write_text(json.dumps(result))


def test_curve_is_pending_without_drope_results(tmp_path):
    (tmp_path / "eval").mkdir()
    assert drope_curve_table(tmp_path) == PENDING


def test_curve_lists_steps_in_numeric_order(tmp_path):
    for step in (1000, 200):
        for scale in (1, 2, 4):
            write_drope_result(tmp_path, step, scale, perplexity=12.5, niah_accuracy=0.5, coefficient=0.1)
    steps = [line.split("|")[1].strip() for line in drope_curve_table(tmp_path).splitlines()[2:]]
    assert steps == ["200", "1000"]


def test_curve_converts_steps_to_million_tokens(tmp_path):
    for scale in (1, 2, 4):
        write_drope_result(tmp_path, 200, scale, perplexity=12.5, niah_accuracy=0.5, coefficient=0.1)
    assert drope_curve_table(tmp_path).splitlines()[2].split("|")[2].strip() == "13"


def test_markers_are_replaced_by_tables():
    assert fill_markers("до\n<!-- table:a -->\nпосле", {"a": "ТАБЛИЦА"}) == "до\nТАБЛИЦА\nпосле"


def test_percent_has_one_decimal():
    assert percent(0.7599) == "76.0"


def test_markdown_table_has_header_separator_and_rows():
    assert markdown_table(("a", "b"), [["1", "2"]]).splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_beta_sweep_uses_latest_step(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for step, offset in ((200, 100.0), (1000, 0.0)):
        sweep = {"2.0": {"0.0": 60.0 + offset, "0.2": 30.0 + offset}, "4.0": {"0.0": 200.0 + offset, "0.2": 50.0 + offset}}
        (eval_dir / f"beta_step{step}.json").write_text(json.dumps({"perplexity": sweep}))
    assert beta_sweep_table(tmp_path).splitlines()[3] == "| 0.2 | 30.00 | 50.00 |"


def write_longbench_result(results_dir, name, scores):
    longbench_dir = results_dir / "longbench"
    longbench_dir.mkdir(parents=True, exist_ok=True)
    datasets = {dataset: {"score": score, "samples": 10} for dataset, score in scores.items()}
    (longbench_dir / f"{name}.json").write_text(json.dumps({"datasets": datasets}))


def test_longbench_table_is_pending_without_results(tmp_path):
    assert longbench_table(tmp_path) == PENDING


def test_longbench_table_averages_four_datasets(tmp_path):
    write_longbench_result(tmp_path, "yarn_s4", {"multifieldqa_en": 10.0, "musique": 20.0, "gov_report": 30.0, "lcc": 40.0})
    row = [line for line in longbench_table(tmp_path).splitlines() if "(у нас)" in line][0]
    assert row.split("|")[-2].strip() == "25.0"


def test_longbench_table_skips_methods_without_results(tmp_path):
    write_longbench_result(tmp_path, "yarn_s4", {"multifieldqa_en": 1.0, "musique": 1.0, "gov_report": 1.0, "lcc": 1.0})
    assert sum("(у нас)" in line for line in longbench_table(tmp_path).splitlines()) == 1


def test_longbench_empty_table_counts_blank_predictions(tmp_path):
    longbench_dir = tmp_path / "longbench"
    longbench_dir.mkdir()
    predictions = [{"prediction": "", "answers": ["x"]}, {"prediction": " \n", "answers": ["x"]}, {"prediction": "text", "answers": ["x"]}]
    datasets = {dataset: {"score": 0.0, "samples": 3, "predictions": predictions} for dataset in ("multifieldqa_en", "musique", "gov_report", "lcc")}
    (longbench_dir / "yarn_s4.json").write_text(json.dumps({"datasets": datasets}))
    row = [line for line in longbench_empty_table(tmp_path).splitlines() if "YaRN" in line][0]
    assert row.split("|")[2].strip() == "2/3"


def test_longbench_empty_table_is_pending_without_results(tmp_path):
    assert longbench_empty_table(tmp_path) == PENDING
