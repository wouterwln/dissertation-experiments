"""Shared drawing helpers for visualizing the Stochastic Maze (chapter 4).

Used by both `plot_environment.py` (the static environment figure) and
`plot_trajectory.py` (the overlaid-path figures). Keeping the maze rendering in
one place means the trajectory plots sit on exactly the same background as the
environment figure.

Colors come from the repo-root `dissertation_theme` module (single source of
truth across all chapters). The maze is 0-indexed with linear state index
`s = x + y * grid_size_x`; plots use `origin="lower"` so +y is up and +x is
right, matching the action convention NORTH=0, EAST=1, SOUTH=2, WEST=3.

Cell-type markers are deliberately sober for a thesis figure: observation noise
is a light shaded field (with a colorbar), stochastic-transition and
absorbing-sink cells are distinguished by a clean colored rounded border (the
sink additionally carries inward chevrons), and the goal is a small checkered
flag.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from src.environments.stochastic_maze import state_to_xy

# Make the repo-root theme importable (it lives above each chapter's DVC root).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402

# Action index -> (dx, dy). NORTH=0 (+y), EAST=1 (+x), SOUTH=2 (-y), WEST=3 (-x).
ACTION_DELTA = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

NOISE_CMAP = theme.noise_cmap()
GOAL_COLOR = theme.PALETTE["goal"]
GOAL_INK = theme.PALETTE["goal_ink"]
PENALTY_COLOR = theme.PALETTE["penalty"]
START_COLOR = theme.PALETTE["start"]
STOCH_COLOR = theme.PALETTE["stochastic"]
SINK_COLOR = theme.PALETTE["sink"]
GRID_COLOR = theme.PALETTE["grid"]
CELL_COLOR = theme.PALETTE["cell"]
INK = theme.PALETTE["ink"]

# A paper-colored outline so the goal "+1" label stays legible over shaded cells.
_HALO = [pe.withStroke(linewidth=2.6, foreground=CELL_COLOR)]


def maze_noise_grid(env):
    """Per-cell observation noise = 1 - p(correct obs), shaped (gy, gx).

    Read straight off the environment's observation matrix A (A[s, s] is the
    probability the location observation is correct at state s), so the figure
    visualizes exactly the ambiguity the planner's EFE term sees.
    """
    gx, gy = env.grid_size_x, env.grid_size_y
    A = np.asarray(env.observation_matrix)
    grid = np.zeros((gy, gx))
    for s in range(gx * gy):
        x, y = state_to_xy(s, gx)
        grid[y, x] = 1.0 - float(A[s, s])
    return grid


def _rounded_cell(x, y, *, inset, lw, edgecolor, linestyle="solid", zorder=3):
    """A rounded-rectangle frame inset inside cell (x, y)."""
    side = 1.0 - 2 * inset
    return FancyBboxPatch(
        (x - 0.5 + inset, y - 0.5 + inset), side, side,
        boxstyle="round,pad=0,rounding_size=0.12",
        fill=False, edgecolor=edgecolor, lw=lw, linestyle=linestyle,
        joinstyle="round", zorder=zorder,
    )


def _draw_sink(ax, x, y, scale=1.0):
    """Thick dark rounded frame + inward chevrons (absorbing-sink icon)."""
    ax.add_patch(_rounded_cell(x, y, inset=0.06, lw=2.4 * scale,
                               edgecolor=SINK_COLOR, zorder=3))
    chevrons = [((x, y + 0.40), "v"), ((x, y - 0.40), "^"),
                ((x - 0.40, y), ">"), ((x + 0.40, y), "<")]
    for (cx, cy), marker in chevrons:
        ax.scatter([cx], [cy], marker=marker, s=26 * scale, c=SINK_COLOR,
                   zorder=5, linewidths=0)


def _draw_checkered_flag(ax, x, y, scale=1.0):
    """A playful waving checkered flag on a dark pole.

    The checker pattern (nc x nr squares) is rendered on a FINE warped mesh so
    the wavy edges curve smoothly instead of faceting ("low-poly"). The cloth
    rises/falls together across rows, with amplitude growing toward the free
    edge plus a little tip flutter; a thin silhouette outline gives it a clean
    clipart finish.
    """
    pole_x = x - 0.26 * scale
    pole_bot, pole_top = y - 0.36 * scale, y + 0.38 * scale
    ax.plot([pole_x, pole_x], [pole_bot, pole_top], color=SINK_COLOR,
            lw=2.4 * scale, solid_capstyle="round", zorder=5)
    ax.scatter([pole_x], [pole_top], s=20 * scale, color=SINK_COLOR, zorder=6)

    nc, nr = 6, 4                           # visible checker squares
    fw, fh = 0.46 * scale, 0.32 * scale     # flag width / height
    fx0, fy0 = pole_x, pole_top - fh - 0.02 * scale
    nu, nv = 81, 41                         # fine mesh -> smooth curves
    u = np.linspace(0.0, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    U, V = np.meshgrid(u, v)
    # Gentle cloth: one soft wave, amplitude growing toward the free (right) edge
    # and shared across rows, with a faint tip flutter. Kept well below the
    # checker-square height so the squares stay crisp (not stripes).
    amp = 0.055 * scale * (0.25 + U)
    wave = (amp * np.sin(2 * np.pi * 1.1 * U + 0.4)
            + 0.012 * scale * U ** 2 * np.sin(2 * np.pi * 3.0 * U))
    X = fx0 + U * fw
    Y = fy0 + V * fh + wave

    uc, vc = 0.5 * (u[:-1] + u[1:]), 0.5 * (v[:-1] + v[1:])
    UC, VC = np.meshgrid(uc, vc)
    checker = (np.floor(UC * nc) + np.floor(VC * nr)).astype(int) % 2
    ax.pcolormesh(X, Y, checker, cmap=ListedColormap([CELL_COLOR, SINK_COLOR]),
                  shading="flat", edgecolors="none", antialiased=True, zorder=5)
    # Clean clipart-style silhouette outline (4 edges of the warped sheet).
    for xs, ys in [(X[0, :], Y[0, :]), (X[-1, :], Y[-1, :]),
                   (X[:, 0], Y[:, 0]), (X[:, -1], Y[:, -1])]:
        ax.plot(xs, ys, color=SINK_COLOR, lw=1.0 * scale, zorder=6,
                solid_capstyle="round", solid_joinstyle="round")


def draw_maze_background(ax, env, *, vmax=None, show_markers=True,
                         cell_fontsize=8, marker_scale=1.0):
    """Draw the maze grid (light noise field + cell-type markers) onto `ax`.

    Returns the image handle so callers can attach a shared colorbar.
    """
    gx, gy = env.grid_size_x, env.grid_size_y
    noise = maze_noise_grid(env)
    if vmax is None:
        vmax = max(0.4, float(noise.max()))

    im = ax.imshow(noise, origin="lower", cmap=NOISE_CMAP, vmin=0.0, vmax=vmax,
                   extent=[-0.5, gx - 0.5, -0.5, gy - 0.5],
                   interpolation="nearest", zorder=0)

    # Cell grid lines.
    for x in range(gx + 1):
        ax.axvline(x - 0.5, color=GRID_COLOR, lw=0.8, zorder=1)
    for y in range(gy + 1):
        ax.axhline(y - 0.5, color=GRID_COLOR, lw=0.8, zorder=1)

    if show_markers:
        # Stochastic-transition cells: a clean solid accent rounded border.
        for (x, y) in env.stochastic_states:
            ax.add_patch(_rounded_cell(x, y, inset=0.07, lw=2.2 * marker_scale,
                                       edgecolor=STOCH_COLOR, zorder=3))

        for (x, y) in env.sink_states:
            _draw_sink(ax, x, y, scale=marker_scale)

        # Reward cells.
        for s, r in env.reward_states:
            x, y = state_to_xy(s, gx)
            if r > 0:
                _draw_checkered_flag(ax, x, y, scale=marker_scale)
                ax.text(x, y - 0.40, "+1", ha="center", va="center",
                        fontsize=cell_fontsize, color=GOAL_INK,
                        fontweight="bold", zorder=7).set_path_effects(_HALO)
            else:
                ax.text(x, y, "−1", ha="center", va="center",
                        fontsize=cell_fontsize + 1, color=CELL_COLOR,
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.18", fc=PENALTY_COLOR,
                                  ec=CELL_COLOR, lw=0.8))

        # Start state: colored ring, charcoal letter (text wears ink, not the
        # accent -- the ring carries the identity).
        sx, sy = state_to_xy(int(env.agent_state), gx)
        ax.scatter([sx], [sy], marker="o", s=260 * marker_scale,
                   facecolors=CELL_COLOR, edgecolors=START_COLOR,
                   linewidths=2.4, zorder=6)
        ax.text(sx, sy, "S", ha="center", va="center",
                fontsize=cell_fontsize, color=INK,
                fontweight="bold", zorder=7)

    ax.set_xticks(range(gx))
    ax.set_yticks(range(gy))
    ax.set_xlim(-0.5, gx - 0.5)
    ax.set_ylim(-0.5, gy - 0.5)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    return im


# --------------------------------------------------------------------------- #
# Legend: custom handlers so the swatches match the in-cell clipart.
# --------------------------------------------------------------------------- #
class _CheckerHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        nx, ny = 4, 3
        w, h = width / nx, height / ny
        arts = []
        for i in range(ny):
            for j in range(nx):
                c = SINK_COLOR if (i + j) % 2 == 0 else CELL_COLOR
                r = Rectangle((xdescent + j * w, ydescent + i * h), w, h,
                              facecolor=c, edgecolor=GRID_COLOR, lw=0.3)
                r.set_transform(trans)
                arts.append(r)
        return arts


def legend_handles():
    """Return (handles, handler_map) describing the maze cell types.

    The goal entry is a proxy artist rendered by a custom handler so its legend
    swatch matches the in-cell checkered flag. Observation noise is conveyed by
    the colorbar, so it is not duplicated in the legend.
    """
    start = Line2D([0], [0], marker="o", color="none", markerfacecolor=CELL_COLOR,
                   markeredgecolor=START_COLOR, markeredgewidth=2.2,
                   markersize=11, label="Start")
    goal = Line2D([], [], linestyle="none", label="Goal (+1)")
    penalty = Patch(facecolor=PENALTY_COLOR, edgecolor=CELL_COLOR, label="Penalty (−1)")
    stoch = Patch(facecolor="none", edgecolor=STOCH_COLOR, linewidth=2.2,
                  label="Stochastic transition")
    sink = Patch(facecolor="none", edgecolor=SINK_COLOR, linewidth=2.2,
                 label="Absorbing sink")
    handles = [start, goal, penalty, stoch, sink]
    handler_map = {goal: _CheckerHandler()}
    return handles, handler_map
