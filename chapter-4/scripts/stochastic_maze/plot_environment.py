#!/usr/bin/env python3
"""Render a static figure of the Stochastic Maze environment (chapter 4).

Draws the 5x5 grid-world the Nuijten-MP / BP planners act in: a per-cell
observation-noise heatmap (the ambiguity the EFE term routes around), with the
start, goal (+1), penalty (-1), stochastic-transition, and absorbing-sink cells
overlaid. Saves both PDF and PNG.

Usage:
    uv run python scripts/stochastic_maze/plot_environment.py
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments.stochastic_maze import StochasticMaze, DEFAULT_START_STATE

# Import the shared helpers whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).parent))
from maze_viz import draw_maze_background, legend_handles  # noqa: E402
import dissertation_theme as theme  # noqa: E402  (repo root put on path by maze_viz)


def main():
    p = argparse.ArgumentParser(description="Plot the Stochastic Maze environment")
    p.add_argument("--output-dir", type=str, default="data/stochastic_maze/plots")
    p.add_argument("--format", type=str, default="both", choices=["pdf", "png", "both"])
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    theme.apply_rcparams()

    env = StochasticMaze.create(start_state=DEFAULT_START_STATE)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = draw_maze_background(ax, env)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Stochastic Maze  (5×5)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Observation noise  (1 − p correct)")

    handles, handler_map = legend_handles()
    ax.legend(
        handles=handles, handler_map=handler_map, loc="upper center",
        bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False,
        handletextpad=0.5, columnspacing=1.2,
    )

    fig.tight_layout()

    formats = ("pdf", "png") if args.format == "both" else (args.format,)
    for path in theme.savefig_both(fig, Path(args.output_dir) / "environment", formats=formats):
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
