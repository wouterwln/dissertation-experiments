#!/usr/bin/env python3
"""Plot a representative Stochastic-Maze trajectory as a single overlaid path.

Reads an existing experiment `results.json` (produced by the
`stochastic_maze_nuijten` / `stochastic_maze_bp_marginal` DVC stages -- this
script does NOT recompute trajectories) and draws one planner's representative
episode as a numbered, arrowed path over the maze background.

The path handles revisits (e.g. BP's interior oscillation): edges traversed in
both directions are curved to opposite sides so back-and-forth moves stay
legible. Each step is numbered along the line.

Usage:
    uv run python scripts/stochastic_maze/plot_trajectory.py \
        --results data/stochastic_maze/nuijten/results.json \
        --label "Nuijten-MP (AIF)" --output data/stochastic_maze/plots/trajectory_aif
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments.stochastic_maze import StochasticMaze, DEFAULT_START_STATE, state_to_xy

sys.path.insert(0, str(Path(__file__).parent))
from maze_viz import draw_maze_background, legend_handles  # noqa: E402
import dissertation_theme as theme  # noqa: E402  (repo root put on path by maze_viz)

# Arrow color encodes step order (early light teal -> late dark teal), matching
# the chapter-5 trajectory figures; the numbered badges make the order exact.
PATH_CMAP = theme.path_cmap()
PATH_COLOR = theme.PALETTE["path"]  # single-color uses (legend swatch)
INK = theme.PALETTE["ink"]
CELL_COLOR = theme.PALETTE["cell"]


def outcome(ep):
    if ep["reached_goal"]:
        return "success"
    if ep["hit_penalty"]:
        return "penalty"
    return "timeout"


def select_episode(episodes, select, episode_idx):
    """Pick the episode to plot and report the outcome distribution.

    select="auto": modal trajectory within the dominant outcome class.
    select in {success,penalty,timeout}: modal trajectory within that class.
    episode_idx not None: that exact episode.
    """
    dist = Counter(outcome(ep) for ep in episodes)
    if episode_idx is not None:
        chosen = next(ep for ep in episodes if ep["episode"] == episode_idx)
        return chosen, dist

    if select == "auto":
        target = dist.most_common(1)[0][0]
    else:
        target = select
    pool = [ep for ep in episodes if outcome(ep) == target]
    if not pool:
        raise SystemExit(f"No episodes with outcome '{target}' (have {dict(dist)})")

    # Modal trajectory within the class; tie-break on the lowest episode index.
    traj_counts = Counter(tuple(ep["trajectory"]) for ep in pool)
    modal_traj = traj_counts.most_common(1)[0][0]
    chosen = min(
        (ep for ep in pool if tuple(ep["trajectory"]) == modal_traj),
        key=lambda ep: ep["episode"],
    )
    return chosen, dist


def draw_path(ax, env, trajectory):
    """Overlay the trajectory as numbered, arrowed, revisit-aware curved edges."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch

    gx = env.grid_size_x
    pts = [state_to_xy(s, gx) for s in trajectory]

    # An edge that is also traversed in reverse somewhere gets curved so the two
    # directions don't draw on top of each other.
    edges = [(trajectory[i], trajectory[i + 1]) for i in range(len(trajectory) - 1)]
    edge_set = set(edges)

    norm = Normalize(vmin=0, vmax=max(len(trajectory) - 2, 1))
    for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
        a, b = trajectory[i], trajectory[i + 1]
        if a == b:
            continue  # stayed put (wall / sink); no visible move
        step_color = PATH_CMAP(norm(i))
        rad = 0.22 if (b, a) in edge_set else 0.0
        arrow = FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=14, lw=2.0,
            color=step_color, alpha=0.95, zorder=8,
        )
        ax.add_patch(arrow)

        # Step number placed "along the line": at the chord midpoint, pushed to
        # the left of travel (matching arc3's bulge direction).
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dist = (dx * dx + dy * dy) ** 0.5 or 1.0
        off = 0.16 + 0.5 * rad
        lx, ly = mx + (-dy / dist) * off, my + (dx / dist) * off
        ax.text(lx, ly, str(i + 1), ha="center", va="center", fontsize=7.5,
                color=INK, fontweight="bold", zorder=9,
                bbox=dict(boxstyle="circle,pad=0.12", fc=CELL_COLOR,
                          ec=step_color, lw=0.8))


def main():
    p = argparse.ArgumentParser(description="Plot a representative maze trajectory")
    p.add_argument("--results", type=str, required=True, help="path to results.json")
    p.add_argument("--label", type=str, required=True, help="planner label for the title")
    p.add_argument("--output", type=str, required=True, help="output path prefix (no extension)")
    p.add_argument("--select", type=str, default="auto",
                   choices=["auto", "success", "penalty", "timeout"])
    p.add_argument("--episode", type=int, default=None, help="pin a specific episode index")
    p.add_argument("--format", type=str, default="both", choices=["pdf", "png", "both"])
    args = p.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    episodes = data["episodes"]

    chosen, dist = select_episode(episodes, args.select, args.episode)
    out = outcome(chosen)
    print(f"[{args.label}] outcome distribution over {len(episodes)} episodes: {dict(dist)}")
    print(f"[{args.label}] selected episode {chosen['episode']} "
          f"(outcome={out}, reward={chosen['total_reward']:+.0f}, "
          f"steps={chosen['n_steps']}): {chosen['trajectory']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    theme.apply_rcparams()

    env = StochasticMaze.create(start_state=DEFAULT_START_STATE)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = draw_maze_background(ax, env)
    draw_path(ax, env, chosen["trajectory"])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Title is just the planner; episode/outcome/step details go in the caption.
    ax.set_title(args.label)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Observation noise  (1 − p correct)")

    handles, handler_map = legend_handles()
    handles.append(plt.Line2D([0], [0], color=PATH_COLOR, lw=2.0,
                              marker=">", markersize=6,
                              label="Trajectory (numbered, light→dark)"))
    ax.legend(handles=handles, handler_map=handler_map, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False,
              handletextpad=0.5, columnspacing=1.0)

    fig.tight_layout()

    formats = ("pdf", "png") if args.format == "both" else (args.format,)
    for path in theme.savefig_both(fig, args.output, formats=formats):
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
