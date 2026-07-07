#!/usr/bin/env python
"""Side-by-side RockSample trajectory comparison across planning methods.

Reads the per-method result JSONs produced by ``run_rocksample.py``
(``data/results/rocksample/<method>.json``), which embed the full per-episode
trajectories (positions, per-step actions, sensing/collection masks, and the
agent's per-step posterior over rock quality). For a given config it picks the
first episode whose true latent config matches and draws that trajectory.

The intended main-text figure is two panels on the same rock configuration:
AIF-MP senses, then samples the good rocks, then exits, versus Direct-prior MP
walking straight to the exit. Figure styling follows the shared dissertation
theme (``dissertation_theme.py`` at the repo root).

Usage:
    # the illustrative config used in the chapter (default)
    uv run python plot_rocksample_trajectory.py --config-idx 6
    # every config, one figure each (used by the DVC pipeline)
    uv run python plot_rocksample_trajectory.py --all-configs
"""
import sys
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent))

# --- Repo-root dissertation theme (imported at runtime, not a DVC dep) -------
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from environments.rocksample import (  # noqa: E402
    pos_to_rc,
    is_exit,
    N_MOVEMENT_ACTIONS,
)


METHOD_LABELS = {
    "loopy": "BP",
    "dyn-channel": "RM-MP",
    "nuijten": "Direct-prior MP",
    "vbp-channel": "VBP",
    "active-inference": "AIF-MP",
}

# The chapter figure contrasts the information-seeking planner with the
# direct-prior baseline: AIF-MP (senses + samples the good rocks) on the left,
# Direct-prior MP (beelines to the exit) on the right.
DEFAULT_METHODS = ["active-inference", "nuijten"]

# --- Colors, all from the shared palette so the dissertation re-skins from one
# place, and consistent with the Wumpus figure's grid grammar: good rocks wear
# the reward green (as Wumpus gold), bad rocks the hazard coral-red (as the
# Wumpus pit), the exit column a faint reward tint, and the step path the
# navigation-amber early->late ramp (Wumpus uses the same flat arrow amber).
# SENSE and the agent start reuse their cross-figure cyan/teal rings; a SAMPLE
# (good rock collected) is ringed in charcoal -- the one mark Wumpus lacks, kept
# neutral so it pops off the green/red rocks and the amber path. ---------------
GOOD_COLOR = theme.PALETTE["goal"]      # good rock (clean green, as WW gold)
BAD_COLOR = theme.PALETTE["penalty"]    # bad rock (coral-red, as the WW pit)
EXIT_COLOR = theme.PALETTE["goal"]      # exit column tint (faint reward green)
CELL_COLOR = theme.PALETTE["cell"]
GRID_COLOR = theme.PALETTE["grid"]
INK = theme.PALETTE["ink"]
START_COLOR = theme.PALETTE["start"]    # thesis-primary teal ring
SENSE_COLOR = theme.PALETTE["sense"]    # cyan SENSE ring
# SAMPLE is the decisive AIF-MP action, so it wears the hero teal. It shares the
# start ring's hue, but the shape carries the distinction (SAMPLE is the big
# cell-border rectangle; the agent start / SENSE are small corner circles).
SAMPLE_COLOR = theme.PALETTE["aif_mp"]  # hero teal "collected" ring
# Letter ink on the rock fills: charcoal on the mid-green good rock, paper-white
# on the coral-red bad rock (mirrors the Wumpus terrain-letter convention).
GOOD_LETTER_INK = theme.PALETTE["ink"]
BAD_LETTER_INK = theme.PALETTE["cell"]


# =============================================================================
# Data loading — per-method result JSONs from run_rocksample.py
# =============================================================================


def load_method_results(results_dir, method):
    """Load one method's result JSON (raises FileNotFoundError if absent)."""
    path = Path(results_dir) / f"{method}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    with open(path) as f:
        return json.load(f)


def build_config_traj(data, config_idx):
    """Assemble a per-config trajectory dict from a method's results JSON.

    Picks the first episode whose true latent config equals ``config_idx`` (the
    config sequence is seed-determined and identical across methods, so this
    yields a consistent cross-method comparison). Returns ``None`` if the method
    never encountered that config.
    """
    env = data["env"]
    g = int(env["grid_size"])
    n_rocks = int(env["n_rocks"])
    traj_rec = None
    for tr in data["trajectories"]:
        if int(tr["theta_true"]) == config_idx:
            traj_rec = tr
            break
    if traj_rec is None:
        return None
    # Per-rock ground-truth quality for this config (1 = good, 0 = bad).
    qualities = [int(round(q)) for q in env["qualities"][config_idx]]
    return {
        "grid_size": g,
        "n_rocks": n_rocks,
        "rock_positions": [int(p) for p in env["rock_positions"]],
        "qualities": qualities,
        "start_pos": int(env["start_pos"]),
        "positions": traj_rec["positions"],
        "actions": traj_rec["actions"],
        "masks": traj_rec["masks"],
        "q_static": traj_rec.get("q_static"),
        "rewards": traj_rec["rewards"],
        "terminated": traj_rec["terminated"],
        "truncated": traj_rec["truncated"],
        "config_idx": config_idx,
    }


def action_kind(action, n_rocks):
    """Classify an action as ('move'|'sense'|'sample', rock_or_None)."""
    if action < N_MOVEMENT_ACTIONS:
        return "move", None
    if action < N_MOVEMENT_ACTIONS + n_rocks:
        return "sense", action - N_MOVEMENT_ACTIONS
    return "sample", None


# =============================================================================
# Rendering
# =============================================================================


def draw_panel(ax, traj, method_label):
    g = int(traj["grid_size"])
    n_rocks = int(traj["n_rocks"])
    positions = traj["positions"]
    actions = traj["actions"]
    rock_positions = traj["rock_positions"]
    qualities = traj["qualities"]

    # Blank cell background so the grid lines and marks read on the page.
    ax.imshow(np.zeros((g, g, 3)) + _to_rgb(CELL_COLOR), interpolation="nearest",
              zorder=0)

    # Exit column (rightmost) — a faint reward tint marking the goal edge.
    exit_col = g - 1
    ax.add_patch(Rectangle((exit_col - 0.5, -0.5), 1, g, facecolor=EXIT_COLOR,
                           alpha=0.14, edgecolor="none", zorder=0.5))
    ax.text(exit_col, -0.62, "exit", ha="center", va="bottom",
            fontsize=theme.CELL_BADGE_SIZE, color=theme.PALETTE["goal_ink"],
            fontweight="bold")

    for i in range(g + 1):
        ax.axhline(i - 0.5, color=GRID_COLOR, lw=1.2, zorder=1)
        ax.axvline(i - 0.5, color=GRID_COLOR, lw=1.2, zorder=1)

    # Rocks: cell filled by ground-truth quality, labelled with the rock index.
    for r, pos in enumerate(rock_positions):
        rr, cc = pos_to_rc(pos, g)
        good = qualities[r] > 0.5
        ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1,
                               facecolor=GOOD_COLOR if good else BAD_COLOR,
                               alpha=0.85, edgecolor="none", zorder=1.5))
        ax.text(cc, rr, f"R{r}", ha="center", va="center",
                fontsize=theme.CELL_LABEL_SIZE, fontweight="bold",
                color=GOOD_LETTER_INK if good else BAD_LETTER_INK, zorder=2)

    # Trajectory arrows, coloured light -> dark by step order (navigation-amber
    # ramp, same family as the Wumpus figure's flat arrow amber).
    cmap = theme.nav_cmap()
    norm = Normalize(0, max(len(actions) - 1, 1))
    for t in range(len(actions)):
        p_from = int(positions[t])
        p_to = int(positions[t + 1])
        if p_from == p_to:
            continue  # stationary action (SENSE / SAMPLE / no-op) — no arrow
        color = cmap(norm(t))
        r_from, c_from = pos_to_rc(p_from, g)
        r_to, c_to = pos_to_rc(p_to, g)
        ax.annotate(
            "",
            xy=(c_to, r_to), xytext=(c_from, r_from),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                            mutation_scale=15, shrinkA=15, shrinkB=15),
            zorder=6,
        )
        # Step-order badge, perpendicular to the move (as in the Wumpus figure).
        mx, my = (c_from + c_to) / 2, (r_from + r_to) / 2
        dx, dy = c_to - c_from, r_to - r_from
        dnorm = (dx * dx + dy * dy) ** 0.5 or 1.0
        lx = mx + (-dy / dnorm) * 0.22
        ly = my + (dx / dnorm) * 0.22
        ax.text(lx, ly, str(t + 1), ha="center", va="center",
                fontsize=theme.CELL_BADGE_SIZE, fontweight="bold", color=INK,
                zorder=9,
                bbox=dict(boxstyle="circle,pad=0.10", fc=CELL_COLOR,
                          ec=color, lw=1.0))

    # SENSE badges (one per sensing cell) and SAMPLE "collected" rings.
    sensed_cells = set()
    for t, a in enumerate(actions):
        kind, _rock = action_kind(int(a), n_rocks)
        p = int(positions[t])
        rr, cc = pos_to_rc(p, g)
        if kind == "sense" and p not in sensed_cells:
            sensed_cells.add(p)
            ax.text(cc - 0.30, rr - 0.30, "S", ha="center", va="center",
                    fontsize=theme.CELL_BADGE_SIZE, fontweight="bold", color=INK,
                    zorder=10,
                    bbox=dict(boxstyle="circle,pad=0.12", fc=CELL_COLOR,
                              ec=SENSE_COLOR, lw=1.2))
        elif kind == "sample" and p in rock_positions:
            ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, fill=False,
                                   edgecolor=SAMPLE_COLOR, lw=3.0, zorder=7))

    # Agent start badge.
    p0 = int(traj["start_pos"])
    r0, c0 = pos_to_rc(p0, g)
    ax.text(c0 + 0.30, r0 - 0.30, "A", ha="center", va="center",
            fontsize=theme.CELL_BADGE_SIZE, fontweight="bold", color=INK,
            zorder=10,
            bbox=dict(boxstyle="circle,pad=0.12", fc=CELL_COLOR,
                      ec=START_COLOR, lw=1.2))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(g - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(method_label, fontsize=10)

    # Outcome caption: reward, steps, and good-rock retrieval (ties to the
    # results-table retrieval column).
    total_reward = float(np.sum(traj["rewards"]))
    n_steps = len(actions)
    final_mask = int(traj["masks"][-1])
    n_good = sum(qualities)
    good_collected = sum(1 for r in range(n_rocks)
                         if (final_mask >> r) & 1 and qualities[r] > 0.5)
    ax.set_xlabel(f"R={total_reward:+.0f}, {n_steps} steps, "
                  f"{good_collected}/{n_good} good rocks", fontsize=8)


def _to_rgb(hex_color):
    from matplotlib.colors import to_rgb
    return np.array(to_rgb(hex_color))


def add_legend(fig):
    """Shared legend below the row of panels."""
    handles = [
        mpatches.Patch(facecolor=GOOD_COLOR, edgecolor=GRID_COLOR, alpha=0.85,
                       label="good rock"),
        mpatches.Patch(facecolor=BAD_COLOR, edgecolor=GRID_COLOR, alpha=0.85,
                       label="bad rock"),
        Line2D([0], [0], marker="o", markerfacecolor=CELL_COLOR,
               markeredgecolor=START_COLOR, markersize=10, markeredgewidth=1.8,
               linestyle="None", label="A  agent start"),
        Line2D([0], [0], marker="o", markerfacecolor=CELL_COLOR,
               markeredgecolor=SENSE_COLOR, markersize=10, markeredgewidth=1.8,
               linestyle="None", label="S  SENSE"),
        Line2D([0], [0], marker="s", markerfacecolor="none",
               markeredgecolor=SAMPLE_COLOR, markersize=10, markeredgewidth=2.6,
               linestyle="None", label="SAMPLE (collected)"),
        Line2D([0], [0], color=theme.PALETTE["arrow"], lw=2.2, marker=">",
               markersize=6, label="trajectory (light→dark)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.02))


def render_figure(method_data, methods, config_idx, out_prefix, suptitle=False):
    """Render one comparison figure across methods for a single config.

    ``method_data`` maps method name -> loaded results JSON. ``out_prefix`` is a
    path without extension; the figure is written as both PDF and PNG. Returns
    True if at least one method contributed a panel.
    """
    trajectories = []
    for method in methods:
        data = method_data.get(method)
        if data is None:
            continue
        traj = build_config_traj(data, config_idx)
        if traj is None:
            print(f"WARNING: {method} has no episode with config {config_idx}, "
                  f"skipping", file=sys.stderr)
            continue
        trajectories.append((method, traj))

    if not trajectories:
        return False

    n = len(trajectories)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.9))
    if n == 1:
        axes = [axes]

    for ax, (method, traj) in zip(axes, trajectories):
        draw_panel(ax, traj, METHOD_LABELS.get(method, method))

    if suptitle:
        fig.suptitle(f"RockSample — config {config_idx}", fontsize=11)
        fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0.12, 1, 1.0])

    add_legend(fig)

    written = theme.savefig_both(fig, out_prefix)
    plt.close(fig)
    print(f"Saved {', '.join(str(p) for p in written)}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="RockSample trajectory comparison plot")
    parser.add_argument("--results-dir", default="data/results/rocksample",
                        help="Directory of per-method result JSONs from "
                             "run_rocksample.py")
    parser.add_argument("--config-idx", type=int, default=6,
                        help="Which config (theta_true) to render. Ignored if "
                             "--all-configs is set.")
    parser.add_argument("--all-configs", action="store_true",
                        help="Render one figure per config (used by the DVC "
                             "pipeline).")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", default=None,
                        help="Output path prefix (no extension) for "
                             "single-config mode.")
    parser.add_argument("--output-dir", default="data/plots/rocksample",
                        help="Output directory for --all-configs mode.")
    parser.add_argument("--suptitle", action="store_true",
                        help="Add a figure-level config title (off by default; "
                             "the dissertation caption carries that info).")
    args = parser.parse_args()

    theme.apply_rcparams()

    method_data = {}
    for method in args.methods:
        try:
            method_data[method] = load_method_results(args.results_dir, method)
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)
    if not method_data:
        print(f"No result JSONs found under {args.results_dir} for methods "
              f"{args.methods}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)

    if args.all_configs:
        n_configs = int(next(iter(method_data.values()))["env"]["n_configs"])
        any_ok = False
        for c in range(n_configs):
            prefix = out_dir / f"rocksample_traj_config_{c:02d}"
            ok = render_figure(method_data, args.methods, c, prefix,
                               suptitle=args.suptitle)
            any_ok = any_ok or ok
        if not any_ok:
            print("No figures rendered (no matching configs).", file=sys.stderr)
            sys.exit(1)
        return

    prefix = (Path(args.output) if args.output is not None
              else out_dir / f"rocksample_traj_config_{args.config_idx:02d}")
    ok = render_figure(method_data, args.methods, args.config_idx, prefix,
                       suptitle=args.suptitle)
    if not ok:
        print(f"No trajectories found for config_idx={args.config_idx}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
