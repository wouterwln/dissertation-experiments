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


# Gradient-curve label by which planner produced the timing (set per-environment
# in run_gradient.py: T-Maze -> factorized, Stochastic Maze -> temporal).
GRAD_LABELS = {
    "factorized": "Tabular VFE (chapter 3)",
    "temporal": "Factorized VFE (chapter 3)",
}


def load_merged(data_dir: Path):
    mp = {r["tag"]: r for r in json.load(open(data_dir / "mp_timings.json"))["records"]}
    grad = {r["tag"]: r for r in json.load(open(data_dir / "gradient_timings.json"))["records"]}
    rows = []
    # Left-join on MP: keep configs the MP side timed even if the gradient side
    # skipped them (e.g. factorized enumeration capped at its max horizon), so the
    # MP curve still extends past the gradient wall in the plot.
    for tag, m in mp.items():
        g = grad.get(tag)
        rows.append({
            "tag": tag, "env": m["env"], "axis": m["axis"], "axis_value": m["axis_value"],
            "n_states": m["n_states"], "horizon": m["horizon"],
            "mp_ms": m["mp_ms"],
            "grad_ms": g["grad_ms"] if g else None,
            "grad_planner": g.get("grad_planner") if g else None,
            "speedup": (g["grad_ms"] / m["mp_ms"]) if g else None,
            "mp_first_action": m.get("mp_first_action"),
            "grad_first_action": g.get("grad_first_action") if g else None,
        })
    return rows


def plot_axis(rows, axis, xlabel, title, out_base):
    sel = sorted([r for r in rows if r["axis"] == axis], key=lambda r: r["axis_value"])
    if not sel:
        return None
    xs = [r["axis_value"] for r in sel]
    mp = [r["mp_ms"] for r in sel]
    gsel = [r for r in sel if r["grad_ms"] is not None]           # gradient ran here
    gxs = [r["axis_value"] for r in gsel]
    gr = [r["grad_ms"] for r in gsel]
    planner = next((r["grad_planner"] for r in gsel if r.get("grad_planner")), None)
    glabel = GRAD_LABELS.get(planner, "Gradient VFE (chapter 3)")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(gxs, gr, "o-", color="#c0392b", label=glabel)
    ax.plot(xs, mp, "s-", color="#2c7fb8", label="Direct-prior MP")
    ax.set_yscale("log")
    # Mark where the gradient planner stops (enumeration intractable beyond it)
    # while MP keeps scaling.
    if gxs and xs and max(xs) > max(gxs):
        ax.axvline(max(gxs), color="#c0392b", ls="--", alpha=0.35)
        ax.text(max(gxs), gr[-1], "  enumeration\n  intractable →",
                color="#c0392b", fontsize=8, va="center", ha="left", alpha=0.8)
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
        grad_s = f"{r['grad_ms']:10.2f}" if r["grad_ms"] is not None else f"{'--':>10s}"
        spd = f"{r['speedup']:6.0f}x" if r["speedup"] is not None else f"{'--':>7s}"
        print(f"{r['tag']:20s} {r['n_states']:4d} {r['horizon']:3d} "
              f"{r['mp_ms']:9.3f} {grad_s} {spd}")
    print(f"\nWrote plots + scaling_summary.json to {data_dir}")


if __name__ == "__main__":
    main()
