from types import SimpleNamespace

from transformers import TrainerState

from snapshot_callback import SnapshotCallback


def make_checkpoint(output_dir, step):
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "config.json").write_text("{}")
    (checkpoint_dir / "optimizer.pt").write_bytes(b"optimizer")


def save_event(tmp_path, is_world_process_zero):
    output_dir = tmp_path / "run"
    make_checkpoint(output_dir, step=40)
    state = TrainerState(global_step=40, is_world_process_zero=is_world_process_zero)
    SnapshotCallback(snapshot_dir=str(tmp_path / "snapshots")).on_save(
        SimpleNamespace(output_dir=str(output_dir)), state, control=None
    )
    return tmp_path / "snapshots" / "step-40"


def test_copies_weights_and_config_on_main_process(tmp_path):
    snapshot = save_event(tmp_path, is_world_process_zero=True)
    assert sorted(path.name for path in snapshot.iterdir()) == ["config.json", "model.safetensors"]


def test_skips_copy_on_other_processes(tmp_path):
    snapshot = save_event(tmp_path, is_world_process_zero=False)
    assert not snapshot.exists()
