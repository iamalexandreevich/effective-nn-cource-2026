import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HW2_DIR = Path(__file__).resolve().parents[1]
TRAIN_ENTRY = HW2_DIR / "kaggle" / "run_authors_train.py"
CONFIG_DIR = HW2_DIR / "configs"
REQUIREMENTS = HW2_DIR / "kaggle" / "requirements-kaggle.txt"
CODE_VERSION_FILE = HW2_DIR / "kaggle" / "code_version.txt"
WORKING_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")

RUN_CONFIG = "drope_smollm360m_t4"
TOKENS_PER_SEQUENCE = 2048
TRAIN_BATCH_SIZE = 32
PER_DEVICE_BATCH = 2
# Сравниваем DDP на двух T4 и одну T4: в DDP шаг оказался медленнее, чем на одной карте
SMOKE_PROCESS_COUNTS = (2,)

# Через Trainer: ~0.9K токенов/с на одной T4 и ~1.9K в DDP на 2xT4, 1000 шагов по 65K токенов ≈ 9.4 ч
MAIN_MAX_STEPS = 1000
MAIN_PROCESS_COUNT = 2

PROFILES = {
    "smoke": dict(max_steps=8, save_steps=100_000, timeout_hours=0.4),
    "main": dict(max_steps=MAIN_MAX_STEPS, save_steps=MAIN_MAX_STEPS // 5, timeout_hours=11.0),
    "resume": dict(max_steps=MAIN_MAX_STEPS, save_steps=MAIN_MAX_STEPS // 5, timeout_hours=11.0),
}


def main():
    profile_name = parse_profile_name()
    install_requirements()
    if profile_name == "smoke":
        run_smoke()
    else:
        run_long(profile_name)


def run_smoke():
    attempts = []
    for process_count in SMOKE_PROCESS_COUNTS:
        output_dir = WORKING_DIR / f"smoke-p{process_count}"
        log_path = WORKING_DIR / "logs" / f"smoke-p{process_count}.log"
        returncode = run_training(PROFILES["smoke"], process_count, output_dir, log_path, resume_dir=None)
        if returncode == 0:
            attempts.append({"process_count": process_count, **summarize_smoke_attempt(output_dir, PER_DEVICE_BATCH)})
        else:
            attempts.append({"process_count": process_count, "returncode": returncode})
    write_json(WORKING_DIR / "smoke_summary.json", {"code_version": code_version(), "gpu_count": gpu_count(), "attempts": attempts})


def run_long(profile_name):
    if MAIN_MAX_STEPS <= 0 or MAIN_PROCESS_COUNT <= 0:
        raise ValueError("MAIN_MAX_STEPS и MAIN_PROCESS_COUNT не заданы, см. Task 7 плана")
    profile = PROFILES[profile_name]
    resume_dir = find_previous_run_dir(INPUT_DIR) if profile_name == "resume" else None
    started = time.time()
    returncode = run_training(
        profile, MAIN_PROCESS_COUNT, WORKING_DIR / "run", WORKING_DIR / "logs" / f"{profile_name}.log", resume_dir
    )
    write_json(
        WORKING_DIR / f"{profile_name}_info.json",
        {
            "code_version": code_version(),
            "profile": profile,
            "process_count": MAIN_PROCESS_COUNT,
            "resume_dir": str(resume_dir) if resume_dir else None,
            "returncode": returncode,
            "timed_out": returncode is None,
            "elapsed_hours": (time.time() - started) / 3600,
        },
    )


def run_training(profile, process_count, output_dir, log_path, resume_dir):
    distributed = ["--multi_gpu"] if process_count > 1 else []
    command = [
        sys.executable, "-m", "accelerate.commands.launch",
        *distributed, "--num_processes", str(process_count), "--main_process_port", "29500",
        str(TRAIN_ENTRY), "--config-dir", str(CONFIG_DIR),
        *build_overrides(profile, output_dir, resume_dir),
    ]
    env = {**os.environ, "PYTHONPATH": str(HW2_DIR / "src"), "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
    if process_count == 1:
        # Иначе Trainer увидит две карты в одном процессе и включит DataParallel
        env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return run_with_timeout(command, env, profile["timeout_hours"] * 3600, log_path)


def build_overrides(profile, output_dir, resume_dir):
    overrides = [
        f"run_cfg@_global_={RUN_CONFIG}",
        f"max_steps={profile['max_steps']}",
        f"save_steps={profile['save_steps']}",
        f"train_batch_size={TRAIN_BATCH_SIZE}",
        f"per_device_train_batch_size={PER_DEVICE_BATCH}",
        "dataloader_num_workers=0",
        "dataloader_prefetch_factor=null",
        "logging_steps=1",
        f"tokens_path={find_tokens_path(INPUT_DIR)}",
        f"output_dir={output_dir}",
        f"snapshot_dir={WORKING_DIR / 'snapshots'}",
    ]
    if resume_dir is not None:
        overrides.append(f"resume_from={resume_dir}")
    return overrides


def run_with_timeout(command, env, timeout_seconds, log_path):
    with open(log_path, "a") as log_file:
        process = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
        )
        copier = threading.Thread(target=copy_lines, args=(process.stdout, log_file), daemon=True)
        copier.start()
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            returncode = None
        copier.join(timeout=10)
        return returncode


def copy_lines(stream, log_file):
    for line in stream:
        log_file.write(line)
        log_file.flush()
        sys.stdout.write(line)
        sys.stdout.flush()


def summarize_smoke_attempt(output_dir, per_device_batch):
    train_results = json.loads((output_dir / "train_results.json").read_text())
    log_history = json.loads((output_dir / "trainer_state.json").read_text())["log_history"]
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    return {
        "per_device_batch": per_device_batch,
        "tokens_per_second": train_results["train_samples_per_second"] * TOKENS_PER_SEQUENCE,
        "losses": losses,
        "has_nonfinite_loss": not all(math.isfinite(loss) for loss in losses),
    }


def find_previous_run_dir(input_root):
    checkpoints = list(input_root.glob("**/checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(f"нет чекпоинта в {input_root}")
    return checkpoints[0].parent


def find_tokens_path(input_root):
    paths = list(input_root.glob("**/fineweb_edu_tokens.npy"))
    if not paths:
        raise FileNotFoundError(f"нет fineweb_edu_tokens.npy в {input_root}")
    return paths[0]


def is_out_of_memory(log_text):
    return "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text


def install_requirements():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)], check=True)


def gpu_count():
    import torch

    return torch.cuda.device_count()


def code_version():
    return CODE_VERSION_FILE.read_text().strip()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def parse_profile_name():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    return parser.parse_args().profile


if __name__ == "__main__":
    main()
