#!/usr/bin/env python3
"""Plot Bethe Free Energy convergence for the Direct-prior MP planner (chapter 4).

Reads `data/<env>/nuijten/convergence.json` for each environment and produces a
VFE-vs-iteration plot: median across all recorded planning-step traces with an
inter-quartile band, plus faint individual traces.

Usage:
    uv run python scripts/convergence/plot_convergence.py
    uv run python scripts/convergence/plot_convergence.py --environments tmaze stochastic_maze
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ENV_LABELS = {
    "tmaze": "T-Maze",
    "stochastic_maze": "Stochastic Maze",
}


def load_traces(input_dir, env):
    path = Path(input_dir) / env / "nuijten" / "convergence.json"
    if not path.exists():
        print(f"Warning: {path} not found, skipping {env}")
        return None
    with open(path) as f:
        data = json.load(f)
    traces = data.get("traces", [])
    if not traces:
        print(f"Warning: no traces in {path}")
        return None
    max_len = max(len(t) for t in traces)
    arr = np.full((len(traces), max_len), np.nan)
    for i, t in enumerate(traces):
        arr[i, :len(t)] = t
    return arr


def plot_env(ax, arr, env):
    iters = np.arange(arr.shape[1])
    # Faint individual traces.
    for row in arr:
        ax.plot(iters, row, color="0.8", linewidth=0.4, alpha=0.5, zorder=1)
    median = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    ax.fill_between(iters, q25, q75, alpha=0.25, color="C0", zorder=2)
    ax.plot(iters, median, color="C0", linewidth=2.0, zorder=3,
            label=f"median ({arr.shape[0]} traces)")
    ax.set_xlabel("Message-passing iteration")
    ax.set_ylabel("Bethe Free Energy")
    ax.set_title(f"{ENV_LABELS.get(env, env)}: Direct-prior MP convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser(description="Plot Direct-prior MP BFE convergence")
    p.add_argument("--environments", nargs="+", default=["tmaze", "stochastic_maze"])
    p.add_argument("--input-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="data/convergence_plots")
    p.add_argument("--format", type=str, default="both", choices=["pdf", "png", "both"])
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "legend.fontsize": 9,
    })

    exts = ["pdf", "png"] if args.format == "both" else [args.format]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for env in args.environments:
        arr = load_traces(args.input_dir, env)
        if arr is None:
            continue
        loaded.append((env, arr))

        fig, ax = plt.subplots(figsize=(6, 4))
        plot_env(ax, arr, env)
        fig.tight_layout()
        for ext in exts:
            fig.savefig(out_dir / f"{env}_convergence.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {env}_convergence plot to {out_dir}")

    # Combined side-by-side figure.
    if len(loaded) > 1:
        fig, axes = plt.subplots(1, len(loaded), figsize=(6 * len(loaded), 4))
        for ax, (env, arr) in zip(np.atleast_1d(axes), loaded):
            plot_env(ax, arr, env)
        fig.tight_layout()
        for ext in exts:
            fig.savefig(out_dir / f"convergence_all.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined convergence plot to {out_dir}")


if __name__ == "__main__":
    main()
