#!/usr/bin/env python3
"""Render rock-sample trajectories as animated GIF and MP4.

Per-planner: one GIF+MP4 cycling through all episodes for --seed-idx.
Comparison: one GIF+MP4 showing all planners side-by-side at the same step.
"""
import argparse
import json
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from learningpomdp.environments.rock_sample import decode_state, sample_layout


# --- Drawing helpers ---------------------------------------------------------

def _rock_quality(theta_true: int, K: int) -> np.ndarray:
    """Return quality[0..K-1] ∈ {0=bad, 1=good} from the flat theta index."""
    return np.array(np.unravel_index(theta_true, (2,) * (2 * K))[:K])


_MOVE_LABELS = ["↑", "↓", "→", "←"]
_ACTION_COLORS = {"move": "#888888", "sense": "#e6842a", "sample": "#2ca02c"}


def _action_label(action: int, K: int) -> str:
    if action < 4:
        return _MOVE_LABELS[action]
    if action < 4 + K:
        return f"sense rock {action - 4}"
    return "sample"


def _action_type(action: int, K: int) -> str:
    if action < 4:
        return "move"
    if action < 4 + K:
        return "sense"
    return "sample"


def _draw_grid(ax, pos, mask, action, quality, K, grid_size, rock_cells,
               step, ep, planner, cum_r, show_planner=False):
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.axis("off")

    cell_to_rock = {c: i for i, c in enumerate(rock_cells)}

    for cell in range(grid_size * grid_size):
        col = cell % grid_size
        # Flip y-axis so row 0 (top of grid) is at the top of the figure.
        row = grid_size - 1 - (cell // grid_size)

        if cell in cell_to_rock:
            ri = cell_to_rock[cell]
            sampled = bool((mask >> ri) & 1)
            if sampled:
                color = "#aaaaaa"
                label = "✓" if quality[ri] else "✗"
            else:
                color = "#4daf4a" if quality[ri] else "#e41a1c"
                label = f"r{ri}"
        else:
            color, label = "#f0f0f0", ""

        ax.add_patch(mpatches.Rectangle(
            (col, row), 1, 1, lw=0.5, edgecolor="#888888", facecolor=color,
        ))
        if label:
            ax.text(col + 0.5, row + 0.5, label, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")

    # Dashed sense beam from agent to targeted rock.
    if 4 <= action < 4 + K:
        k = action - 4
        rc = rock_cells[k]
        ax_c = pos % grid_size + 0.5
        ax_r = grid_size - 1 - pos // grid_size + 0.5
        rx_c = rc % grid_size + 0.5
        rx_r = grid_size - 1 - rc // grid_size + 0.5
        if rc == pos:
            # Agent is on the rock — draw a small "pulse" circle instead.
            ax.add_patch(mpatches.Circle((ax_c, ax_r), 0.42, lw=1.5,
                                          edgecolor="orange", facecolor="none",
                                          linestyle="dashed", zorder=4))
        else:
            ax.plot([ax_c, rx_c], [ax_r, rx_r],
                    "--", color="orange", lw=1.5, alpha=0.85, zorder=4)

    # Agent as a filled blue circle.
    ax.plot(
        pos % grid_size + 0.5,
        grid_size - 1 - pos // grid_size + 0.5,
        "o", ms=12, color="#377eb8", mec="white", mew=1.5, zorder=5,
    )

    # Per-frame title: episode + step only.
    title = f"Episode {ep + 1}  ·  Step {step}"
    if show_planner:
        title = f"{planner}\n{title}"
    ax.set_title(title, fontsize=8 if show_planner else 9, pad=3)

    # Action label (bottom-left, coloured by type).
    act_color = _ACTION_COLORS[_action_type(action, K)]
    ax.text(0.03, -0.03, _action_label(action, K),
            transform=ax.transAxes, fontsize=7.5, color=act_color,
            ha="left", va="top", clip_on=False)

    # Reward badge (bottom-right).
    r_color = "#2ca02c" if cum_r > 0 else ("#d62728" if cum_r < 0 else "#666666")
    ax.text(0.97, -0.03, f"R = {cum_r:+.0f}",
            transform=ax.transAxes, fontsize=7.5, color=r_color,
            ha="right", va="top", fontweight="bold", clip_on=False)


def _frame(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return rgba[:, :, :3].copy()


def _pad_even(frames):
    """Pad all frames to even (h, w) — required by libx264."""
    h, w = frames[0].shape[:2]
    ph, pw = h + h % 2, w + w % 2
    if ph == h and pw == w:
        return frames
    return [
        np.pad(f, ((0, ph - h), (0, pw - w), (0, 0)), constant_values=255)
        for f in frames
    ]


# --- Reward accumulation -----------------------------------------------------

def _step_reward(action, state, K, n_events, rock_cells, quality, sample_reward):
    """Reward earned when taking `action` from `state`."""
    if action != 4 + K:
        return 0.0
    pos, mask, _ = decode_state(state, K, n_events)
    cell_to_rock = {c: i for i, c in enumerate(rock_cells)}
    if pos not in cell_to_rock:
        return 0.0
    ri = cell_to_rock[pos]
    if (mask >> ri) & 1:
        return 0.0
    return sample_reward if quality[ri] else -sample_reward


# --- Frame builders ----------------------------------------------------------

def _build_single(episodes, planner, K, grid_size, rock_cells, sample_reward):
    n_events = K + 2
    fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.09)
    # Planner name shown once at figure level — not repeated per frame.
    fig.text(0.5, 0.99, planner, ha="center", va="top",
             fontsize=12, fontweight="bold", color="#222222")
    frames = []

    for ep, ep_data in enumerate(episodes):
        states = ep_data["states_visited"]
        actions = ep_data["actions"]
        quality = _rock_quality(ep_data["episode_theta_true"], K)
        cum_r = 0.0

        for t, (s, a) in enumerate(zip(states, actions)):
            cum_r += _step_reward(a, s, K, n_events, rock_cells, quality, sample_reward)
            pos, mask, _ = decode_state(s, K, n_events)
            _draw_grid(ax, pos, mask, a, quality, K, grid_size, rock_cells,
                       t, ep, planner, cum_r)
            frames.append(_frame(fig))

        # Hold final state for 2 extra frames, then 2 blank separators.
        frames.extend([frames[-1]] * 2)
        blank = np.full_like(frames[-1], 255)
        frames.extend([blank, blank])

    plt.close(fig)
    return frames


def _build_comparison(traj_by_planner, planners, K, grid_size, rock_cells, sample_reward):
    n = len(planners)
    n_events = K + 2
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.5), dpi=80)
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.09, wspace=0.04)
    frames = []

    n_episodes = min(len(traj_by_planner[p]) for p in planners)

    for ep in range(n_episodes):
        ep_data = {p: traj_by_planner[p][ep] for p in planners}
        qualities = {p: _rock_quality(ep_data[p]["episode_theta_true"], K) for p in planners}
        n_steps = min(len(d["actions"]) for d in ep_data.values())
        cum_r = {p: 0.0 for p in planners}

        for t in range(n_steps):
            for ax, p in zip(axes, planners):
                s = ep_data[p]["states_visited"][t]
                a = ep_data[p]["actions"][t]
                cum_r[p] += _step_reward(a, s, K, n_events, rock_cells,
                                          qualities[p], sample_reward)
                pos, mask, _ = decode_state(s, K, n_events)
                _draw_grid(ax, pos, mask, a, qualities[p], K, grid_size,
                           rock_cells, t, ep, p, cum_r[p], show_planner=True)
            frames.append(_frame(fig))

        # Hold final state.
        for ax, p in zip(axes, planners):
            s = ep_data[p]["states_visited"][-1]
            a = ep_data[p]["actions"][-1] if ep_data[p]["actions"] else 0
            pos, mask, _ = decode_state(s, K, n_events)
            _draw_grid(ax, pos, mask, a, qualities[p], K, grid_size,
                       rock_cells, n_steps, ep, p, cum_r[p], show_planner=True)
        frames.extend([_frame(fig)] * 2)
        blank = np.full_like(frames[-1], 255)
        frames.extend([blank, blank])

    plt.close(fig)
    return frames


# --- Output ------------------------------------------------------------------

def _save(frames, base: Path, fps: float):
    padded = _pad_even(frames)
    gif_path = str(base) + ".gif"
    mp4_path = str(base) + ".mp4"
    with imageio.get_writer(gif_path, mode="I", fps=fps) as w:
        for f in padded:
            w.append_data(f)
    with imageio.get_writer(mp4_path, fps=fps, macro_block_size=1) as w:
        for f in padded:
            w.append_data(f)
    print(f"  {base.name}: {len(padded)} frames → .gif + .mp4")


# --- Entry point -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="output/rock_sample")
    parser.add_argument("--output-dir", default="output/rock_sample/viz")
    parser.add_argument("--planners", nargs="+",
                        default=["aif", "bamcp", "psrl", "oracle", "random"])
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=4.0)
    # Optional: pick specific episode indices (within --seed-idx) to chain.
    # Each entry is "seed:ep" (e.g. "0:2 1:5") or bare "ep" (uses --seed-idx).
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="Episodes to include, e.g. '2 3' or '0:2 1:5'.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_by_planner: dict[str, dict] = {}
    env_cfg = None
    for planner in args.planners:
        with (in_dir / f"{planner}.json").open() as f:
            data = json.load(f)
        traj_by_planner[planner] = data["trajectories"]
        if env_cfg is None:
            env_cfg = data["config"]["env"]

    K = env_cfg["K"]
    grid_size = env_cfg["grid_size"]
    layout_seed = env_cfg.get("layout_seed", 0)
    sample_reward = float(env_cfg.get("sample_reward", 1.0))
    rock_cells = sample_layout(grid_size, K, layout_seed).rock_cells

    # Resolve the episode list as [(seed_idx, ep_idx), ...].
    if args.episodes is not None:
        ep_pairs = []
        for tok in args.episodes:
            if ":" in tok:
                s, e = tok.split(":", 1)
                ep_pairs.append((int(s), int(e)))
            else:
                ep_pairs.append((args.seed_idx, int(tok)))
    else:
        n_eps = len(next(iter(traj_by_planner.values()))[args.seed_idx])
        ep_pairs = [(args.seed_idx, e) for e in range(n_eps)]

    def _episodes_for(planner):
        trajs = traj_by_planner[planner]
        return [trajs[s][e] for s, e in ep_pairs]

    for planner in args.planners:
        frames = _build_single(_episodes_for(planner), planner,
                               K, grid_size, rock_cells, sample_reward)
        _save(frames, out_dir / planner, args.fps)

    frames = _build_comparison(
        {p: _episodes_for(p) for p in args.planners},
        args.planners, K, grid_size, rock_cells, sample_reward,
    )
    _save(frames, out_dir / "comparison", args.fps)


if __name__ == "__main__":
    main()
