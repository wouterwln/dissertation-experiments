#!/usr/bin/env python3
"""Benchmark environment schematics for Chapter 6 (``fig:environments-ch6``).

Emits one file per benchmark (so each becomes a captioned subfigure):
    env_door_code        — combination-lock system-ID: probe (P) vs commit (C)
    env_minigrid          — raw MiniGrid render of the spatial Door-Code sibling
    env_factored_tiger    — K independent doors: listen (L) vs open (O)
    env_rock_sample       — 5x5 grid, K=4 rocks, exit edge; two theta-components
                            (episodic quality + persistent sensor reliability)

Static schematics — no run data. Figures carry no explanatory prose, titles, or
legends: the dissertation captions carry that. Only intrinsic labels remain
(button letters, reward values, rock/exit labels). Colors come from the shared
repo-root ``dissertation_theme.py``, imported at runtime.

Usage:
    uv run python scripts/plot_environments.py --output-dir output/environments
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Rectangle, FancyArrowPatch, Circle, FancyBboxPatch, Arc,
)

# Repo-root dissertation theme (lives above the chapter's DVC root).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402

from learningpomdp.environments.rock_sample import sample_layout  # noqa: E402
from learningpomdp.environments.minigrid_door_code import (  # noqa: E402
    MiniGridDoorCodeEnv,
)

# Semantic colors (current dissertation grid semantics: green good / coral bad).
GOOD = theme.PALETTE["goal"]
BAD = theme.PALETTE["penalty"]
INFO = theme.PALETTE["sense"]         # probe / listen buttons (cyan, info action)
ACT = theme.PALETTE["arrow"]          # commit / open buttons (amber, decisive act)
HERO = theme.PALETTE["aif_mp"]        # reliable sensor / agent (teal)
INK = theme.PALETTE["ink"]
CELL = theme.PALETTE["cell"]
GRID = theme.PALETTE["grid"]
MUTED = theme.PALETTE["muted"]
METAL = "#B9BBBE"                      # neutral lock/hardware grey (off-palette)


# --- Shared stylized primitives ----------------------------------------------


def _button(ax, cx, cy, r, letter, color):
    """A stylized round console button: soft shadow, bevel, white letter."""
    ax.add_patch(Circle((cx + 0.006, cy - 0.008), r, facecolor=INK, alpha=0.16,
                        edgecolor="none", zorder=3))                 # shadow
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="white",
                        lw=1.6, zorder=4))                           # face
    ax.add_patch(Circle((cx, cy + 0.28 * r), 0.62 * r, facecolor="white",
                        alpha=0.16, edgecolor="none", zorder=5))     # bevel
    ax.text(cx, cy, letter, ha="center", va="center", color="white",
            fontsize=13, fontweight="bold", zorder=6)


def _noisy_bit(ax, cx, cy, r=0.045):
    """A single binary reading wrapped in a dashed 'noisy' ring (info obs)."""
    ax.add_patch(Circle((cx, cy), r, facecolor=CELL, edgecolor=INFO, lw=1.4,
                        zorder=4))
    ax.text(cx, cy, "0/1", ha="center", va="center", fontsize=7,
            color=INK, fontweight="bold", zorder=5)
    ax.add_patch(Circle((cx, cy), r * 1.5, facecolor="none", edgecolor=INFO,
                        lw=1.3, linestyle=(0, (2, 2)), zorder=4))    # noise ring


def _reward(ax, cx, cy, value, color, r=0.05):
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none",
                        alpha=0.92, zorder=4))
    ax.text(cx, cy, value, ha="center", va="center", fontsize=8.5,
            color=theme.PALETTE["cell"] if color == BAD else INK,
            fontweight="bold", zorder=5)


def _code_strip(ax, x0, y, n, wrong_idx, cw=0.032, gap=0.007):
    """A candidate code of n cells: all green, or one red at wrong_idx.

    The n-cell strip encodes that Door-Code's reward is conjunctive over the
    whole code (all cells green -> success; any one red -> failure)."""
    for i in range(n):
        x = x0 + i * (cw + gap)
        col = BAD if i == wrong_idx else GOOD
        ax.add_patch(Rectangle((x, y - cw / 2), cw, cw, facecolor=col,
                               edgecolor="white", lw=0.8, zorder=4))
    return x0 + n * cw + (n - 1) * gap  # right edge


def _arrow(ax, xy_from, xy_to, color, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw,
                                 shrinkA=3, shrinkB=3, zorder=3))


# =============================================================================
# (a) Door-Code — combination-lock system identification
# =============================================================================


def draw_door_code(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Combination padlock with a hidden 4-dial code (theta in S_6) --------
    bx, by, bw, bh = 0.30, 0.62, 0.40, 0.24          # lock body
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0.006,rounding_size=0.03",
                                facecolor=METAL, edgecolor=INK, lw=1.6, zorder=2))
    # Shackle (arc + two legs) rising from the body top.
    cx = bx + bw / 2
    ax.add_patch(Arc((cx, by + bh), 0.20, 0.22, angle=0, theta1=0, theta2=180,
                     lw=6, color=METAL, zorder=1))
    for dx in (-0.10, 0.10):
        ax.plot([cx + dx, cx + dx], [by + bh, by + bh + 0.045],
                color=METAL, lw=6, solid_capstyle="round", zorder=1)
    # Four masked dials = the hidden code.
    n = 4
    dw = 0.066
    gap = (bw - n * dw) / (n + 1)
    for i in range(n):
        dx = bx + gap + i * (dw + gap)
        ax.add_patch(FancyBboxPatch((dx, by + 0.06), dw, bh - 0.12,
                                    boxstyle="round,pad=0.002,rounding_size=0.012",
                                    facecolor=CELL, edgecolor=INK, lw=1.0, zorder=3))
        ax.text(dx + dw / 2, by + bh / 2, "?", ha="center", va="center",
                fontsize=11, color=INK, fontweight="bold", zorder=4)

    # --- PROBE (P): info action -> one noisy bit toward the lock -------------
    _button(ax, 0.15, 0.30, 0.072, "P", INFO)
    _noisy_bit(ax, 0.15, 0.50)
    _arrow(ax, (0.15, 0.372), (0.15, 0.452), INFO)
    _arrow(ax, (0.175, 0.545), (0.31, 0.63), INFO, lw=1.4)

    # --- COMMIT (C): reward is conjunctive over the ENTIRE 4-symbol code -----
    _button(ax, 0.58, 0.30, 0.072, "C", ACT)
    _code_strip(ax, 0.70, 0.45, 4, wrong_idx=None)     # all correct -> +1
    _reward(ax, 0.93, 0.462, "$+1$", GOOD, r=0.045)
    _code_strip(ax, 0.70, 0.27, 4, wrong_idx=2)        # one wrong -> -1
    _reward(ax, 0.93, 0.282, "$-1$", BAD, r=0.045)
    _arrow(ax, (0.648, 0.325), (0.695, 0.44), ACT, lw=1.4)
    _arrow(ax, (0.648, 0.285), (0.695, 0.275), ACT, lw=1.4)


# =============================================================================
# (b) MiniGrid Door-Code — raw Farama render
# =============================================================================


def draw_minigrid(ax):
    ax.axis("off")
    env = MiniGridDoorCodeEnv(
        K=3, lock_length=1, probe_noise=0.1, slip_prob=0.1,
        grid_size=7, agent_view_size=3, layout_seed=1, obs_seed=1,
        render_mode="rgb_array",
    )
    env.reset(seed=1, theta=0)
    ax.imshow(env.render())


# =============================================================================
# (c) Factored Tiger — K independent doors, listen (L) / open (O)
# =============================================================================


def draw_factored_tiger(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    K = 6
    dw, dh = 0.108, 0.24
    gap = (1.0 - K * dw) / (K + 1)
    dy = 0.56
    focus = 2  # the door the representative LISTEN/OPEN act on
    centers = []
    for k in range(K):
        dx = gap + k * (dw + gap)
        centers.append(dx + dw / 2)
        lw = 2.4 if k == focus else 1.3
        ax.add_patch(FancyBboxPatch((dx, dy), dw, dh,
                                    boxstyle="round,pad=0.002,rounding_size=0.012",
                                    facecolor=CELL, edgecolor=INK, lw=lw, zorder=2))
        ax.add_patch(Circle((dx + 0.82 * dw, dy + 0.5 * dh), 0.010,
                            facecolor=INK, edgecolor="none", zorder=3))
        ax.text(dx + dw / 2, dy + dh / 2, "?", ha="center", va="center",
                fontsize=12, color=INK, fontweight="bold", zorder=3)
        # theta labels ABOVE the doors so the listen arrow does not cross them.
        ax.text(dx + dw / 2, dy + dh + 0.05, f"$\\theta_{k}$", ha="center",
                va="center", fontsize=8, color=INK)

    fx = centers[focus]
    # Component-local observation: LISTEN's noisy bit is tied to the focus door.
    _noisy_bit(ax, fx, 0.40)
    _arrow(ax, (fx, 0.448), (fx, dy - 0.005), INFO, lw=1.4)

    # Two stylized buttons: LISTEN (info) and OPEN (decisive, per-door).
    _button(ax, 0.24, 0.17, 0.072, "L", INFO)
    _arrow(ax, (0.305, 0.195), (fx - 0.03, 0.365), INFO, lw=1.4)
    _button(ax, 0.74, 0.17, 0.072, "O", ACT)
    # OPEN acts on a single door -> that door's own +1 / -1 (per-door reward).
    _reward(ax, 0.58, 0.13, "$+1$", GOOD, r=0.045)
    _reward(ax, 0.90, 0.13, "$-1$", BAD, r=0.045)
    _arrow(ax, (0.678, 0.185), (0.62, 0.145), ACT, lw=1.4)
    _arrow(ax, (0.802, 0.185), (0.86, 0.145), ACT, lw=1.4)


# =============================================================================
# (d) RockSample-derived — 5x5 grid, two theta-components
# =============================================================================


def draw_rocksample(ax):
    G, K = 5, 4
    layout = sample_layout(G, K, seed=0)
    rock_cells = list(layout.rock_cells)
    start_cell = layout.agent_start_cell

    ax.set_xlim(-0.05, G + 0.05)
    ax.set_ylim(-0.05, G + 0.35)
    ax.set_aspect("equal")
    ax.axis("off")

    def rc(cell):
        return cell % G, G - 1 - cell // G  # (col, display-row), row 0 at top

    for cell in range(G * G):
        col, row = rc(cell)
        face = GOOD if col == G - 1 else CELL
        alpha = 0.14 if col == G - 1 else 1.0
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face, alpha=alpha,
                               edgecolor=GRID, lw=0.8, zorder=1))
    ax.text(G - 0.5, G + 0.06, "exit", ha="center", va="bottom", fontsize=9,
            color=theme.PALETTE["goal_ink"], fontweight="bold")

    # inner fill = episodic quality; outer ring = persistent sensor reliability.
    combos = [(True, True), (False, True), (True, False), (False, False)]
    for ri, cell in enumerate(rock_cells):
        good, reliable = combos[ri % len(combos)]
        col, row = rc(cell)
        pad = 0.16
        ax.add_patch(Rectangle((col + pad, row + pad), 1 - 2 * pad, 1 - 2 * pad,
                               facecolor=GOOD if good else BAD, edgecolor="none",
                               zorder=3))
        if reliable:
            ax.add_patch(Rectangle((col + 0.04, row + 0.04), 0.92, 0.92,
                                   fill=False, edgecolor=HERO, lw=2.8, zorder=4))
        else:
            ax.add_patch(Rectangle((col + 0.04, row + 0.04), 0.92, 0.92,
                                   fill=False, edgecolor=MUTED, lw=1.9,
                                   linestyle=(0, (3, 2)), zorder=4))
        ax.text(col + 0.5, row + 0.5, f"R{ri}", ha="center", va="center",
                fontsize=15, fontweight="bold",
                color=INK if good else CELL, zorder=5)

    scol, srow = rc(start_cell)
    ax.plot(scol + 0.5, srow + 0.5, "o", ms=20, color=HERO, mec="white",
            mew=1.8, zorder=6)
    ax.text(scol + 0.5, srow + 0.5, "A", ha="center", va="center", fontsize=12,
            color="white", fontweight="bold", zorder=7)


# =============================================================================
# Main — one file per panel
# =============================================================================

PANELS = [
    ("env_door_code", draw_door_code, (4.2, 4.2)),
    ("env_minigrid", draw_minigrid, (4.2, 4.2)),
    ("env_factored_tiger", draw_factored_tiger, (5.6, 4.2)),
    ("env_rock_sample", draw_rocksample, (4.4, 4.6)),
]


def main():
    parser = argparse.ArgumentParser(description="Chapter-6 environment schematics")
    parser.add_argument("--output-dir", type=Path, default=Path("output/environments"),
                        help="directory for the per-panel PDF/PNG files")
    args = parser.parse_args()

    theme.apply_rcparams()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, draw_fn, figsize in PANELS:
        fig, ax = plt.subplots(figsize=figsize)
        draw_fn(ax)
        fig.tight_layout(pad=0.4)
        for ext in ("pdf", "png"):
            fig.savefig(args.output_dir / f"{name}.{ext}", dpi=150,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {name}.pdf / .png")
    print(f"schematics saved to {args.output_dir}")


if __name__ == "__main__":
    main()
