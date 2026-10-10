import os
import runpy
import sys
import traceback
from pathlib import Path

AUTHORS_TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "third_party" / "DroPE" / "train.py"


def main():
    sys.argv[0] = str(AUTHORS_TRAIN_SCRIPT)
    sys.path.insert(0, str(AUTHORS_TRAIN_SCRIPT.parent))
    exit_code = 0
    try:
        runpy.run_path(str(AUTHORS_TRAIN_SCRIPT), run_name="__main__")
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # После обучения процесс не завершается сам: его держат фоновые потоки стримингового датасета
    os._exit(exit_code)


if __name__ == "__main__":
    main()
