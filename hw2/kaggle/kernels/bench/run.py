import glob
import shutil
import subprocess
import sys

CODE_DIR = "/tmp/course"
BENCHMARK_SUFFIX = "/hw2/kaggle/benchmark_throughput.py"

# Kaggle сам распаковывает архив датасета, а /kaggle/input только для чтения
benchmark = glob.glob(f"/kaggle/input/**{BENCHMARK_SUFFIX}", recursive=True)[0]
shutil.copytree(benchmark.removesuffix(BENCHMARK_SUFFIX), CODE_DIR, dirs_exist_ok=True)
requirements = f"{CODE_DIR}/hw2/kaggle/requirements-kaggle.txt"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", requirements, "bitsandbytes"], check=True)
subprocess.run([sys.executable, "-u", f"{CODE_DIR}{BENCHMARK_SUFFIX}"], check=True)
