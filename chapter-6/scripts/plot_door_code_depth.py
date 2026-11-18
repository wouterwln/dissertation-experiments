"""Plot the door-code MCTS depth-sensitivity ablation.

Reads output/door_code_depth/depth<D>/<planner>.json for the depths in
params.door_code_depth.depth_sweep and plots, per planner:

  Left:   block-mean terminal reward vs depth
  Middle: block-mean success rate vs depth
  Right:  block-mean probe visits vs depth

`bamcp_depth` bounds the TOTAL simulated lookahead of one MCTS
simulation (tree descent plus leaf rollout), so it is the tree-search
family's counterpart to AIF-MP's `runner.horizon`. The dashed vertical
line marks that horizon, and the sweep runs UP TO it, never past it:
cells left of the line give the tree-search planners strictly less
lookahead than AIF-MP, and there are no cells to the right because there
is nothing there to measure. The planner applies
`max_depth = min(depth, max(horizon, 1))` and `depth` enters nowhere
else, so any `bamcp_depth > runner.horizon` produces a bit-identical run
— same actions, same RNG stream. An above-H cell would have drawn a flat
right tail that was flat *by construction*, which is exactly the reading
("flat => never lookahead-limited") this ablation is supposed to earn.

The corollary is the stronger fairness argument: matched lookahead is the
SATURATION POINT of the depth axis, not a tuned parameter. No setting of
`bamcp_depth` buys the tree-search family more lookahead than AIF-MP's
horizon, so the matched value is the most generous one available to the
baselines.

What the measured cells buy: if the curve is flat below H, the
tree-search arms were never lookahead-limited and the matched-depth
choice is not load-bearing; if it rises towards H, the previously
unmatched depth was a real confound.

Usage:
    uv run python scripts/plot_door_code_depth.py \\
        --input-dir output/door_code_depth \\
        --output output/door_code_depth/figure.pdf \\
        --horizon 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo-root dissertation theme (single source of truth for figure colors).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402


PLANNERS = ("bamcp", "ids")
PLANNER_LABELS = {"bamcp": "BAMCP", "ids": "IDS"}


def _discover_depths(input_dir: Path) -> list[int]:
    """List depth<D> subdirectories under input_dir, sorted ascending."""
    depths: list[int] = []
    for child in input_dir.iterdir():
        if child.is_dir() and child.name.startswith("depth"):
            try:
                depths.append(int(child.name[5:]))
            except ValueError:
                continue
    return sorted(depths)


def _block_mean(summary: dict, metric: str) -> tuple[float, float]:
    """Per-seed block mean of `metric`, aggregated as mean ± 1.96·SEM.

    Collapses episodes within a block first (per-seed scalar), then
    aggregates across seeds — the same convention as the main figures.
    """
    per_ep = np.asarray(summary["per_episode"][metric], dtype=float)
    per_seed = per_ep.mean(axis=1)
    n = max(per_seed.size, 1)
    sem = per_seed.std(ddof=1) / np.sqrt(n) if per_seed.size > 1 else 0.0
    return float(per_seed.mean()), float(1.96 * sem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot door-code MCTS depth-sensitivity ablation.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/door_code_depth"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="AIF-MP planning horizon to mark on the depth axis "
             "(params runner.horizon)",
    )
    args = parser.parse_args()

    depths = _discover_depths(args.input_dir)
    if not depths:
        raise FileNotFoundError(
            f"no depth<D> subdirectories under {args.input_dir}; "
            f"run `dvc repro door_code_depth` first"
        )

    runs: dict[tuple[int, str], dict] = {}
    for depth in depths:
        for planner in PLANNERS:
            path = args.input_dir / f"depth{depth}" / f"{planner}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path}; run `dvc repro door_code_depth` first"
                )
            with path.open() as f:
                runs[(depth, planner)] = json.load(f)

    theme.apply_rcparams()
    cfg = runs[(depths[0], PLANNERS[0])]["config"]
    # Depth is an ordered sweep, so the per-depth markers wear the theme's
    # ordinal teal ramp; planners are told apart by linestyle.
    depth_colors = dict(zip(depths, theme.ordinal_colors(len(depths))))
    linestyles = {"bamcp": "-", "ids": "--"}

    metrics = (
        ("terminal_reward", "terminal reward", "Terminal reward"),
        ("success", "success rate", "Success rate"),
        ("probe_visits", "probe visits / episode", "Probe visits per episode"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for planner in PLANNERS:
            means = np.array(
                [_block_mean(runs[(d, planner)], metric)[0] for d in depths]
            )
            errs = np.array(
                [_block_mean(runs[(d, planner)], metric)[1] for d in depths]
            )
            ax.plot(
                depths, means, color=theme.planner_color(planner),
                linestyle=linestyles[planner], lw=2, zorder=2,
                label=PLANNER_LABELS[planner],
            )
            ax.fill_between(
                depths, means - errs, means + errs,
                color=theme.planner_color(planner), alpha=0.12,
            )
            # Ordinal ramp on the swept axis itself.
            for d, m in zip(depths, means):
                ax.plot(
                    [d], [m], marker="o", ms=6, zorder=3,
                    color=depth_colors[d],
                    markeredgecolor=theme.planner_color(planner),
                )
        if args.horizon is not None:
            ax.axvline(
                args.horizon, color=theme.PALETTE["ink"], ls=":", lw=1.2,
                alpha=0.7,
            )
            ax.annotate(
                f"AIF-MP horizon H={args.horizon}",
                xy=(args.horizon, 0.02), xycoords=("data", "axes fraction"),
                rotation=90, va="bottom", ha="right", fontsize=8,
                color=theme.PALETTE["ink"],
            )
        ax.set_xlabel("MCTS depth (total tree + rollout lookahead)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (mean ± 1.96·SEM)")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
    axes[0].axhline(0, color=theme.PALETTE["ink"], lw=0.5, alpha=0.4)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set_ylim(-0.02, 1.02)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
