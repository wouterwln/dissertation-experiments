#!/usr/bin/env python
"""Side-by-side trajectory comparison across planning methods for Wumpus World configs.

Reads the per-method result JSONs produced by ``run_wumpus_world.py``
(``data/results/wumpus_world/<method>.json``), which embed the full
per-episode trajectories. For a given config it picks the first episode whose
true latent config matches and draws that trajectory. Figure styling follows
the shared dissertation theme (``dissertation_theme.py`` at the repo root).

Usage:
    # every config, one figure each (used by the DVC pipeline)
    uv run python plot_wumpus_trajectory.py --all-configs
    # a single config
    uv run python plot_wumpus_trajectory.py --config-idx 18
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

from environments.wumpus_world import pos_to_rc  # noqa: E402


METHOD_LABELS = {
    "loopy": "BP",
    "dyn-channel": "RM-MP",
    "nuijten": "Direct-prior MP",
    "vbp-channel": "VBP",
    "active-inference": "AIF-MP",
}

DEFAULT_METHODS = ["loopy", "dyn-channel", "nuijten", "vbp-channel", "active-inference"]

EMPTY, PIT, WUMPUS, GOLD = 0, 1, 2, 3

# Terrain colors drawn from the shared palette so the whole dissertation can be
# re-skinned from one place: gold is the reward target (goal green), pits are a
# hazard (penalty red), the wumpus is the second, deadlier hazard (near-black
# sink). Empty cells stay a light neutral so the grid lines and arrows read.
TERRAIN_COLORS = [
    "#f2f2f2",              # EMPTY
    theme.PALETTE["penalty"],  # PIT
    theme.PALETTE["sink"],     # WUMPUS
    theme.PALETTE["goal"],     # GOLD
]
TERRAIN_LETTERS = {PIT: "P", WUMPUS: "W", GOLD: "G"}
# Letters sit on saturated/dark cells, so draw them white for contrast.
TERRAIN_LETTER_COLOR = "white"
TERRAIN_CMAP = ListedColormap(TERRAIN_COLORS)

START_COLOR = theme.PALETTE["start"]
SENSE_COLOR = theme.PALETTE["sink"]

# Arrow color gradient encodes step order (early -> late), anchored on the
# theme's trajectory hue so it matches the other dissertation figures.
PATH_CMAP = LinearSegmentedColormap.from_list(
    "path_gradient", ["#f2b779", theme.PALETTE["path"]], N=256
)


# =============================================================================
# Data loading — per-method result JSONs from run_wumpus_world.py
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

    Picks the first episode whose true latent config equals ``config_idx``
    (the config sequence is seed-determined and identical across methods, so
    this yields a consistent cross-method comparison). Returns ``None`` if the
    method never encountered that config.
    """
    env = data["env"]
    g = int(env["grid_size"])
    traj_rec = None
    for tr in data["trajectories"]:
        if int(tr["theta_true"]) == config_idx:
            traj_rec = tr
            break
    if traj_rec is None:
        return None
    return {
        "grid_size": g,
        "pits": env["pits"][config_idx],
        "wumpus": env["wumpus"][config_idx],
        "gold": env["gold"][config_idx],
        "positions": traj_rec["positions"],
        "actions": traj_rec["actions"],
        "sensed": traj_rec["sensed"],
        "rewards": traj_rec["rewards"],
        "terminated": traj_rec["terminated"],
        "truncated": traj_rec["truncated"],
        "config_idx": config_idx,
    }


# =============================================================================
# Rendering
# =============================================================================


def build_terrain(traj):
    g = int(traj["grid_size"])
    grid = np.full((g, g), EMPTY, dtype=int)
    for code, key in [(PIT, "pits"), (WUMPUS, "wumpus"), (GOLD, "gold")]:
        for pos, val in enumerate(traj[key]):
            if val > 0.5:
                r, c = pos_to_rc(pos, g)
                grid[r, c] = code
    return grid


def classify_outcome(traj, terrain):
    g = int(traj["grid_size"])
    total_reward = float(np.sum(traj["rewards"]))
    if bool(traj["truncated"]):
        return total_reward, "timeout"
    if bool(traj["terminated"]) and total_reward > 0:
        return total_reward, "success"
    if bool(traj["terminated"]) and total_reward < 0:
        last_pos = int(traj["positions"][-1])
        r, c = pos_to_rc(last_pos, g)
        cell = terrain[r, c]
        if cell == PIT:
            return total_reward, "pit"
        if cell == WUMPUS:
            return total_reward, "wumpus"
    return total_reward, "fail"


def find_sense_steps(traj):
    """Indices of actions that triggered a SENSE (the bit is transient)."""
    sensed = traj["sensed"]
    return [t - 1 for t in range(1, len(sensed)) if sensed[t] == 1]


def draw_panel(ax, traj, method_label):
    g = int(traj["grid_size"])
    terrain = build_terrain(traj)
    positions = traj["positions"]
    actions = traj["actions"]

    ax.imshow(terrain, cmap=TERRAIN_CMAP, vmin=0, vmax=3, interpolation="nearest")

    for i in range(g + 1):
        ax.axhline(i - 0.5, color="white", lw=1.2)
        ax.axvline(i - 0.5, color="white", lw=1.2)

    for r in range(g):
        for c in range(g):
            cell = terrain[r, c]
            if cell in TERRAIN_LETTERS:
                ax.text(c, r, TERRAIN_LETTERS[cell],
                        ha="center", va="center",
                        fontsize=13, fontweight="bold", color=TERRAIN_LETTER_COLOR)

    n_moves = max(len(actions), 1)
    norm = Normalize(vmin=0, vmax=max(n_moves - 1, 1))
    for t in range(len(actions)):
        p_from = int(positions[t])
        p_to = int(positions[t + 1])
        if p_from == p_to:
            continue
        r_from, c_from = pos_to_rc(p_from, g)
        r_to, c_to = pos_to_rc(p_to, g)
        color = PATH_CMAP(norm(t))
        ax.annotate(
            "",
            xy=(c_to, r_to), xytext=(c_from, r_from),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.0,
                mutation_scale=14,
                shrinkA=14, shrinkB=14,
            ),
        )

    p0 = int(positions[0])
    r0, c0 = pos_to_rc(p0, g)
    ax.text(c0 + 0.3, r0 - 0.3, "A",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=START_COLOR,
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=START_COLOR, lw=1))

    sense_steps = find_sense_steps(traj)
    for sense_step in sense_steps:
        p_sense = int(positions[sense_step])
        r_s, c_s = pos_to_rc(p_sense, g)
        ax.text(c_s - 0.3, r_s - 0.3, "S",
                ha="center", va="center",
                fontsize=8, fontweight="bold", color=SENSE_COLOR,
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec=SENSE_COLOR, lw=1))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(g - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(method_label, fontsize=10)

    total_reward, outcome = classify_outcome(traj, terrain)
    n_steps = len(actions)
    ax.set_xlabel(f"R={total_reward:+.0f}, {n_steps} steps, {outcome}", fontsize=8)


def add_legend(fig):
    """Place a shared legend and a step-gradient colorbar below the row of panels."""
    handles = [
        mpatches.Patch(facecolor=TERRAIN_COLORS[PIT], edgecolor="black", label="P  pit"),
        mpatches.Patch(facecolor=TERRAIN_COLORS[WUMPUS], edgecolor="black", label="W  wumpus"),
        mpatches.Patch(facecolor=TERRAIN_COLORS[GOLD], edgecolor="black", label="G  gold"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=START_COLOR,
               markersize=9, markeredgewidth=1.2, linestyle="None",
               label="A  agent start"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=SENSE_COLOR,
               markersize=9, markeredgewidth=1.2, linestyle="None",
               label="S  SENSE action"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.02),
    )

    # Gradient colorbar showing step order (early -> late)
    cbar_ax = fig.add_axes([0.35, -0.01, 0.30, 0.018])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=PATH_CMAP)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["early", "late"])
    cbar.ax.tick_params(labelsize=7, length=0, pad=2)
    cbar.outline.set_visible(False)
    cbar_ax.set_title("arrow color = step order", fontsize=7, pad=2)


def render_figure(method_data, methods, config_idx, out_prefix, suptitle=True):
    """Render one comparison figure across methods for a single config.

    ``method_data`` maps method name -> loaded results JSON. ``out_prefix`` is
    a path without extension; the figure is written as both PDF and PNG.
    Returns True if at least one method contributed a panel.
    """
    trajectories = []
    for method in methods:
        data = method_data.get(method)
        if data is None:
            continue
        traj = build_config_traj(data, config_idx)
        if traj is None:
            print(f"WARNING: {method} has no episode with config {config_idx}, skipping",
                  file=sys.stderr)
            continue
        trajectories.append((method, traj))

    if not trajectories:
        return False

    n = len(trajectories)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.9))
    if n == 1:
        axes = [axes]

    for ax, (method, traj) in zip(axes, trajectories):
        draw_panel(ax, traj, METHOD_LABELS.get(method, method))

    if suptitle:
        fig.suptitle(f"Wumpus World — config {config_idx}", fontsize=11)
        fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0.10, 1, 1.0])

    add_legend(fig)

    written = theme.savefig_both(fig, out_prefix)
    plt.close(fig)
    print(f"Saved {', '.join(str(p) for p in written)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Wumpus World trajectory comparison plot")
    parser.add_argument("--results-dir", default="data/results/wumpus_world",
                        help="Directory of per-method result JSONs from run_wumpus_world.py")
    parser.add_argument("--config-idx", type=int, default=None,
                        help="Which config to render. Ignored if --all-configs is set.")
    parser.add_argument("--all-configs", action="store_true",
                        help="Render one figure per config (used by the DVC pipeline).")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", default=None,
                        help="Output path prefix (no extension) for single-config mode.")
    parser.add_argument("--output-dir", default="data/plots/wumpus_world",
                        help="Output directory for --all-configs mode.")
    parser.add_argument("--no-suptitle", action="store_true",
                        help="Omit the figure-level title (useful for paper inclusion).")
    args = parser.parse_args()

    theme.apply_rcparams()
    suptitle = not args.no_suptitle

    # Load every requested method's results once; warn (don't fail) on missing.
    method_data = {}
    for method in args.methods:
        try:
            method_data[method] = load_method_results(args.results_dir, method)
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)
    if not method_data:
        print(f"No result JSONs found under {args.results_dir} for methods {args.methods}",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)

    if args.all_configs:
        n_configs = int(next(iter(method_data.values()))["env"]["n_configs"])
        any_ok = False
        for c in range(n_configs):
            prefix = out_dir / f"method_comparison_config_{c:02d}"
            ok = render_figure(method_data, args.methods, c, prefix, suptitle=suptitle)
            any_ok = any_ok or ok
        if not any_ok:
            print("No figures rendered (no matching configs).", file=sys.stderr)
            sys.exit(1)
        return

    config_idx = args.config_idx if args.config_idx is not None else 0
    prefix = (Path(args.output) if args.output is not None
              else out_dir / f"method_comparison_config_{config_idx:02d}")
    ok = render_figure(method_data, args.methods, config_idx, prefix, suptitle=suptitle)
    if not ok:
        print(f"No trajectories found for config_idx={config_idx}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
