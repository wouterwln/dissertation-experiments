#!/usr/bin/env python3
"""Scaling study, part 3: merge MP + gradient timings and plot.

Reads `mp_timings.json` (chapter-4 MP side) and `gradient_timings.json`
(chapter-3 gradient side), joins them per config `tag`, and emits:
  - `scaling_horizon.{png,pdf}`   -- per-call time vs planning horizon (T-Maze)
  - `scaling_state_space.{png,pdf}` -- per-call time vs state-space size (Stochastic Maze)
  - `scaling_summary.json`         -- merged table + MP-vs-gradient speedup factors
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_merged(data_dir: Path):
    mp = {r["tag"]: r for r in json.load(open(data_dir / "mp_timings.json"))["records"]}
    grad = {r["tag"]: r for r in json.load(open(data_dir / "gradient_timings.json"))["records"]}
    rows = []
    for tag, m in mp.items():
        g = grad.get(tag)
        if g is None:
            continue
        rows.append({
            "tag": tag, "env": m["env"], "axis": m["axis"], "axis_value": m["axis_value"],
            "n_states": m["n_states"], "horizon": m["horizon"],
            "mp_ms": m["mp_ms"], "grad_ms": g["grad_ms"],
            "speedup": g["grad_ms"] / m["mp_ms"],
            "mp_first_action": m.get("mp_first_action"),
            "grad_first_action": g.get("grad_first_action"),
        })
    return rows


def plot_axis(rows, axis, xlabel, title, out_base):
    sel = sorted([r for r in rows if r["axis"] == axis], key=lambda r: r["axis_value"])
    if not sel:
        return None
    xs = [r["axis_value"] for r in sel]
    mp = [r["mp_ms"] for r in sel]
    gr = [r["grad_ms"] for r in sel]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xs, gr, "o-", color="#c0392b", label="Gradient VFE (chapter 3)")
    ax.plot(xs, mp, "s-", color="#2c7fb8", label="Direct-prior MP")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("planning time per call (ms, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)
    return sel


def main():
    p = argparse.ArgumentParser(description="Merge + plot scaling study")
    p.add_argument("--data-dir", type=str, default="data/scaling")
    args = p.parse_args()
    data_dir = Path(args.data_dir)

    rows = load_merged(data_dir)
    plot_axis(rows, "horizon", "planning horizon",
              "Planning cost vs horizon (T-Maze, S=5)",
              str(data_dir / "scaling_horizon"))
    plot_axis(rows, "n_states", "number of states  S",
              "Planning cost vs state-space size (Stochastic Maze)",
              str(data_dir / "scaling_state_space"))

    with open(data_dir / "scaling_summary.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(f"{'tag':20s} {'S':>4s} {'H':>3s} {'MP ms':>9s} {'grad ms':>10s} {'speedup':>8s}")
    for r in sorted(rows, key=lambda r: (r["axis"], r["axis_value"])):
        print(f"{r['tag']:20s} {r['n_states']:4d} {r['horizon']:3d} "
              f"{r['mp_ms']:9.3f} {r['grad_ms']:10.2f} {r['speedup']:7.0f}x")
    print(f"\nWrote plots + scaling_summary.json to {data_dir}")


if __name__ == "__main__":
    main()
