import argparse
import csv
import json
import math
import os
import platform
import random
import threading
import time

import numpy as np
import pynvml
import torch
from torch.utils.flop_counter import FlopCounterMode

from models import build_model

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]

FIELDS = ["S", "B", "is_validation", "status", "latency_s", "latency_p10_s", "latency_p90_s",
          "n_latency", "memory_bytes", "memory_before_bytes", "reserved_bytes", "energy_j",
          "avg_power_w", "flops_counted"]


def make_grid(seed):
    rng = random.Random(seed)
    extra_s = rng.sample([s for s in range(32, 513, 16) if s not in BASE_S], 4)
    extra_b = rng.sample([b for b in range(1, 257) if b & (b - 1) != 0], 3)
    sizes = sorted(BASE_S + extra_s)
    batches = sorted(BASE_B + extra_b)
    grid = []
    for S in sizes:
        for B in batches:
            grid.append((S, B, S in extra_s or B in extra_b))
    return grid


class EnergyMeter:
    def __init__(self, handle):
        self.handle = handle
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            self.has_counter = True
        except pynvml.NVMLError:
            self.has_counter = False

    def start(self):
        self.t0 = time.perf_counter()
        if self.has_counter:
            self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample)
        self.thread.start()

    def stop(self):
        dt = time.perf_counter() - self.t0
        if self.has_counter:
            return (pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - self.e0) / 1000, dt
        self.running = False
        self.thread.join()
        t, p = np.array(self.samples).T
        return np.sum(np.diff(t) * (p[1:] + p[:-1]) / 2), dt

    def _sample(self):
        while self.running:
            self.samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000))
            time.sleep(0.005)


def count_flops(model, S, B):
    counter = FlopCounterMode(display=False)
    with counter, torch.inference_mode():
        model.to("meta")(torch.empty(B, 3, S, S, device="meta"))
    return counter.get_total_flops()


def measure_memory(model, x):
    model(x)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = model(x)
    torch.cuda.synchronize()
    del out
    return torch.cuda.max_memory_allocated(), before, torch.cuda.max_memory_reserved()


def measure_latency(model, x, min_time, max_reps):
    for _ in range(3):
        model(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    model(x)
    torch.cuda.synchronize()
    est = time.perf_counter() - t
    reps = int(min(max_reps, max(10, math.ceil(min_time / est))))

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    return np.median(times), np.percentile(times, 10), np.percentile(times, 90), reps


def measure_energy(model, x, meter, min_time, est):
    # back-to-back forwards, sync only at the end, so the gpu is busy the whole window
    reps = max(5, math.ceil(min_time / est))
    torch.cuda.synchronize()
    meter.start()
    for _ in range(reps):
        model(x)
    torch.cuda.synchronize()
    e, dt = meter.stop()
    return e / reps, e / dt


def run_config(model, S, B, meter, args):
    row = {"S": S, "B": B}
    x = torch.randn(B, 3, S, S, device="cuda")
    mem, before, reserved = measure_memory(model, x)
    row.update(memory_bytes=mem, memory_before_bytes=before, reserved_bytes=reserved)
    lat, p10, p90, n = measure_latency(model, x, args.latency_time, args.max_reps)
    row.update(latency_s=lat, latency_p10_s=p10, latency_p90_s=p90, n_latency=n)
    e, p = measure_energy(model, x, meter, args.energy_time, lat)
    row.update(energy_j=e, avg_power_w=p, status="ok")
    return row


def env_info(args, handle):
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "memory_limit_bytes": args.memory_limit_gb * 2**30 if args.memory_limit_gb > 0 else props.total_memory,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "energy_counter": EnergyMeter(handle).has_counter,
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/measurements.csv")
    parser.add_argument("--memory-limit-gb", type=float, default=2.0, help="0 = no limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-time", type=float, default=0.5)
    parser.add_argument("--energy-time", type=float, default=1.5)
    parser.add_argument("--max-reps", type=int, default=200)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meter = EnergyMeter(handle)

    if args.memory_limit_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gb * 2**30 / total)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    info = env_info(args, handle)
    with open(os.path.join(os.path.dirname(args.out), "env.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(info)

    model = build_model().cuda().eval()
    meta_model = build_model()
    grid = make_grid(args.seed)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, (S, B, is_val) in enumerate(grid):
            torch.cuda.empty_cache()
            try:
                with torch.inference_mode():
                    row = run_config(model, S, B, meter, args)
            except torch.cuda.OutOfMemoryError:
                row = {"S": S, "B": B, "status": "OOM"}
            torch.cuda.empty_cache()
            row["is_validation"] = int(is_val)
            row["flops_counted"] = count_flops(meta_model, S, B)
            writer.writerow(row)
            f.flush()
            print(f"[{i + 1}/{len(grid)}] S={S} B={B} {row['status']} "
                  f"lat={row.get('latency_s', float('nan')) * 1e3:.3f}ms "
                  f"mem={row.get('memory_bytes', 0) / 2**20:.1f}MiB "
                  f"E={row.get('energy_j', float('nan')):.4f}J")


if __name__ == "__main__":
    main()
