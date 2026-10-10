import glob
import shutil
import subprocess
import sys

PROFILE = "main"
CODE_DIR = "/tmp/course"
LAUNCHER_SUFFIX = "/hw2/kaggle/launch_recalibration.py"

# Kaggle сам распаковывает архив датасета, а /kaggle/input только для чтения
launcher = glob.glob(f"/kaggle/input/**{LAUNCHER_SUFFIX}", recursive=True)[0]
shutil.copytree(launcher.removesuffix(LAUNCHER_SUFFIX), CODE_DIR, dirs_exist_ok=True)
subprocess.run([sys.executable, f"{CODE_DIR}{LAUNCHER_SUFFIX}", "--profile", PROFILE], check=True)
