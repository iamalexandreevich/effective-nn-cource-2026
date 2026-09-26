import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import equations

S_LINE = np.linspace(32, 512, 200)
B_LINE = np.geomspace(1, 256, 200)


def load_theta(path):
    with open(path) as f:
        t = json.load(f)
    theta = t["latency"]
    return theta, {"latency": theta, **t["energy"]}


def colors(values):
    cmap = plt.get_cmap("viridis")
    return {v: cmap(i / max(1, len(values) - 1)) for i, v in enumerate(values)}


def scatter_split(ax, x, y, val, c, label=None):
    ax.scatter(x[~val], y[~val], color=c, s=18, label=label)
    ax.scatter(x[val], y[val], facecolors="none", edgecolors=c, s=30)


def vs_size_and_batch(df, column, predict, ylabel, title, path, scale=1.0):
    ok = df[df["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    batches = sorted(df["B"].unique())
    cb = colors(batches)
    for B in batches:
        part = ok[ok["B"] == B]
        axes[0].plot(S_LINE, predict(S_LINE, B) * scale, color=cb[B], lw=1)
        scatter_split(axes[0], part["S"].values, part[column].values * scale,
                      part["is_validation"].values == 1, cb[B], f"B={B}")
    axes[0].set_xlabel("image size S, px")
    axes[0].set_xscale("log", base=2)

    sizes = sorted(df["S"].unique())
    cs = colors(sizes)
    for S in sizes:
        part = ok[ok["S"] == S]
        axes[1].plot(B_LINE, predict(S, B_LINE) * scale, color=cs[S], lw=1)
        scatter_split(axes[1], part["B"].values, part[column].values * scale,
                      part["is_validation"].values == 1, cs[S], f"S={S}")
    axes[1].set_xlabel("batch size B")
    axes[1].set_xscale("log", base=2)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title + "\nlines = equation, filled = calibration grid, hollow = validation")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parity(df, theta, theta_e, path):
    ok = df[df["status"] == "ok"]
    S, B = ok["S"].values, ok["B"].values
    val = ok["is_validation"].values == 1
    pairs = [
        ("latency, ms", equations.latency(S, B, theta) * 1e3, ok["latency_s"].values * 1e3),
        ("energy, J", equations.energy(S, B, theta_e), ok["energy_j"].values),
        ("peak memory, MiB", equations.memory(S, B) / 2**20, ok["memory_bytes"].values / 2**20),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, pred, real) in zip(axes, pairs):
        ax.scatter(real[~val], pred[~val], s=14, label="calibration")
        ax.scatter(real[val], pred[val], s=14, marker="^", label="validation")
        lo, hi = min(real.min(), pred.min()), max(real.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("measured " + name)
        ax.set_ylabel("predicted " + name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def regime_map(df, theta, path):
    ok = df[df["status"] == "ok"]
    s = np.linspace(32, 512, 300)
    b = np.geomspace(1, 256, 300)
    SS, BB = np.meshgrid(s, b)
    reg = equations.regime(SS, BB, theta)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#9ecae1", "#fdae6b"])
    ax.pcolormesh(SS, BB, reg, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
    err = (equations.latency(ok["S"].values, ok["B"].values, theta) - ok["latency_s"].values) / ok["latency_s"].values
    sc = ax.scatter(ok["S"], ok["B"], c=err * 100, cmap="coolwarm", vmin=-50, vmax=50, edgecolors="k", s=45)
    oom = df[df["status"] == "OOM"]
    ax.scatter(oom["S"], oom["B"], marker="x", color="k", s=40, label="OOM")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("image size S, px")
    ax.set_ylabel("batch size B")
    handles = [matplotlib.patches.Patch(color=cmap(i), label=n) for i, n in enumerate(["launch-bound", "memory-bound", "compute-bound"])]
    ax.legend(handles=handles + [ax.collections[-1]], loc="upper right")
    fig.colorbar(sc, label="latency error (pred - meas) / meas, %")
    ax.set_title("Predicted regime and latency error at measured points")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def oom_map(df, env, path):
    limit = env["memory_limit_bytes"]
    s = np.linspace(32, 512, 300)
    b = np.geomspace(1, 256, 300)
    SS, BB = np.meshgrid(s, b)

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(SS, BB, equations.memory(SS, BB) / 2**30, levels=20, cmap="Blues")
    ax.contour(SS, BB, equations.memory(SS, BB), levels=[limit], colors="r", linewidths=2)
    ok = df[df["status"] == "ok"]
    oom = df[df["status"] == "OOM"]
    ax.scatter(ok["S"], ok["B"], color="g", s=15, label="fits")
    ax.scatter(oom["S"], oom["B"], marker="x", color="r", s=40, label="OOM measured")
    ax.plot([], [], "r", label=f"predicted limit {limit / 2**30:.1f} GiB")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("image size S, px")
    ax.set_ylabel("batch size B")
    fig.colorbar(cs, label="predicted peak memory, GiB")
    ax.legend(loc="upper right")
    ax.set_title("Memory(S, B) vs measured OOM")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def throughput(df, theta, path):
    ok = df[df["status"] == "ok"]
    sizes = sorted(df["S"].unique())
    cs = colors(sizes)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for S in sizes:
        part = ok[ok["S"] == S]
        ax.plot(B_LINE, equations.flops(S, B_LINE) / equations.latency(S, B_LINE, theta) / 1e12, color=cs[S], lw=1)
        ax.scatter(part["B"], equations.flops(S, part["B"].values) / part["latency_s"] / 1e12, color=cs[S], s=15, label=f"S={S}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("achieved TFLOP/s")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Achieved throughput: points = measured, lines = model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.results, "measurements.csv"))
    theta, theta_e = load_theta(os.path.join(args.results, "theta.json"))
    with open(os.path.join(args.results, "env.json")) as f:
        env = json.load(f)
    fig_dir = os.path.join(args.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    vs_size_and_batch(df, "flops_counted", equations.flops, "GFLOPs", "FLOPs(S, B), points = torch FlopCounterMode",
                      os.path.join(fig_dir, "flops.png"), scale=1e-9)
    vs_size_and_batch(df, "memory_bytes", equations.memory, "peak allocated memory, MiB", "Memory(S, B)",
                      os.path.join(fig_dir, "memory.png"), scale=1 / 2**20)
    vs_size_and_batch(df, "latency_s", lambda s, b: equations.latency(s, b, theta), "latency, ms", "Latency(S, B)",
                      os.path.join(fig_dir, "latency.png"), scale=1e3)
    vs_size_and_batch(df, "energy_j", lambda s, b: equations.energy(s, b, theta_e), "energy, J", "Energy(S, B)",
                      os.path.join(fig_dir, "energy.png"))
    parity(df, theta, theta_e, os.path.join(fig_dir, "parity.png"))
    regime_map(df, theta, os.path.join(fig_dir, "regimes.png"))
    oom_map(df, env, os.path.join(fig_dir, "oom_map.png"))
    throughput(df, theta, os.path.join(fig_dir, "throughput.png"))


if __name__ == "__main__":
    main()
