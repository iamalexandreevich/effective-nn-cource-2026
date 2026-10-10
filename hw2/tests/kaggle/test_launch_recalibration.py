import json

import pytest

from launch_recalibration import find_previous_run_dir, is_out_of_memory, summarize_smoke_attempt


def write_smoke_outputs(output_dir, samples_per_second, losses):
    output_dir.mkdir(parents=True)
    (output_dir / "train_results.json").write_text(json.dumps({"train_samples_per_second": samples_per_second}))
    log_history = [{"loss": loss, "step": index} for index, loss in enumerate(losses)]
    (output_dir / "trainer_state.json").write_text(json.dumps({"log_history": log_history}))


def test_smoke_summary_converts_samples_to_tokens(tmp_path):
    write_smoke_outputs(tmp_path / "smoke", samples_per_second=5.0, losses=[3.1, 2.9])
    assert summarize_smoke_attempt(tmp_path / "smoke", per_device_batch=4)["tokens_per_second"] == 10240.0


def test_smoke_summary_flags_nonfinite_loss(tmp_path):
    write_smoke_outputs(tmp_path / "smoke", samples_per_second=5.0, losses=[3.1, float("nan")])
    assert summarize_smoke_attempt(tmp_path / "smoke", per_device_batch=4)["has_nonfinite_loss"]


def test_finds_run_dir_holding_checkpoint(tmp_path):
    (tmp_path / "drope-main" / "run" / "checkpoint-1500").mkdir(parents=True)
    assert find_previous_run_dir(tmp_path) == tmp_path / "drope-main" / "run"


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_previous_run_dir(tmp_path)


@pytest.mark.parametrize(
    "log_text, expected",
    [("torch.OutOfMemoryError: CUDA out of memory.", True), ("KeyError: 'loss'", False)],
    ids=["cuda_oom", "other_error"],
)
def test_detects_out_of_memory(log_text, expected):
    assert is_out_of_memory(log_text) is expected
