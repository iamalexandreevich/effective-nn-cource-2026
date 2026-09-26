import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations


def load(path):
    df = pd.read_csv(path)
    df["oom"] = df["status"] == "OOM"
    return df


def fit_latency(df):
    S, B, t = df["S"].values, df["B"].values, df["latency_s"].values

    def residuals(p):
        theta = dict(zip(["t_launch", "peak_flops", "bandwidth"], np.exp(p)))
        return np.log(equations.latency(S, B, theta)) - np.log(t)

    p0 = np.log([t.min(), 4e12, 2e11])
    # latency is a max() of pieces, so try a few starts and keep the best
    best = None
    for scale in [0.3, 1.0, 3.0]:
        res = least_squares(residuals, p0 + np.log([1, scale, scale]))
        if best is None or res.cost < best.cost:
            best = res
    return dict(zip(["t_launch", "peak_flops", "bandwidth"], np.exp(best.x)))


def fit_energy(df, theta):
    S, B, e = df["S"].values, df["B"].values, df["energy_j"].values
    X = np.stack([equations.latency(S, B, theta), equations.flops(S, B), equations.bytes_moved(S, B)], axis=1)
    # relative error weighting, otherwise the big configs dominate
    coef, _ = nnls(X / e[:, None], np.ones_like(e))
    return {"latency": theta, "p_static": coef[0], "e_flop": coef[1], "e_byte": coef[2]}


def rel_errors(pred, real):
    err = np.abs(pred - real) / real
    return {"mape": float(err.mean()), "median": float(np.median(err)), "max": float(err.max())}


def report(df, theta, theta_e):
    ok = df[~df["oom"]]
    out = {}
    for name, part in [("train", ok[ok["is_validation"] == 0]), ("validation", ok[ok["is_validation"] == 1])]:
        S, B = part["S"].values, part["B"].values
        out[name] = {
            "latency": rel_errors(equations.latency(S, B, theta), part["latency_s"].values),
            "energy": rel_errors(equations.energy(S, B, theta_e), part["energy_j"].values),
            "memory": rel_errors(equations.memory(S, B), part["memory_bytes"].values),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/measurements.csv")
    parser.add_argument("--out", default="results/theta.json")
    args = parser.parse_args()

    df = load(args.csv)
    train = df[(~df["oom"]) & (df["is_validation"] == 0)]
    theta = fit_latency(train)
    theta_e = fit_energy(train, theta)
    errors = report(df, theta, theta_e)

    result = {"latency": theta, "energy": {k: v for k, v in theta_e.items() if k != "latency"}, "errors": errors}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
