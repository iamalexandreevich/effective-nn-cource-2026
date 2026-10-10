import shutil
from pathlib import Path

from transformers import TrainerCallback

SNAPSHOT_FILES = ("model.safetensors", "config.json")


class SnapshotCallback(TrainerCallback):
    def __init__(self, snapshot_dir: str) -> None:
        self._snapshot_dir = Path(snapshot_dir)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        target_dir = self._snapshot_dir / f"step-{state.global_step}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in SNAPSHOT_FILES:
            shutil.copy2(checkpoint_dir / name, target_dir / name)
