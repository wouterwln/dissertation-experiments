"""Plot the RockSample MCTS depth-sensitivity ablation.

Reads output/rock_sample_depth/depth<D>/<planner>.json for the depths in
params.rock_sample_depth.depth_sweep and plots, per planner:

  Left:   block-mean reward_normalised vs depth
  Middle: block-mean SENSE count per episode vs depth
  Right:  terminal H(q(theta_sensor)) vs depth

`bamcp_depth` bounds the TOTAL simulated lookahead of one MCTS
simulation (tree descent plus leaf rollout), so it is the tree-search
family's counterpart to AIF-MP's `runner.horizon`. The dashed vertical
line marks that horizon, and the sweep runs up to it and stops:
`max_depth = min(depth, max(horizon, 1))` saturates there and `depth`
enters nowhere else, so above-H cells would be bit-identical no-ops. That
also makes matched lookahead the *saturation point* of this axis rather
than a tuned parameter — see plot_door_code_depth.py. RockSample is the
benchmark where the old unmatched depth was most severe — AIF-MP saw 15
steps of lookahead while every tree-search baseline saw 6 — so this curve
is what separates "AIF-MP plans better" from "AIF-MP was given more
lookahead".

BAMCP carries the curve and runs every depth in the sweep. IDS is optional
here, and in practice absent: no IDS depth cells are defined, because even
two endpoint cells cost ~3043 SBU — 46% of the genoa budget remaining — and
would only confirm a flat line. Its depth-invariance is reported from
profiling instead (8.07 h -> 8.18 h across this sweep).

So a planner may be absent entirely (omission, named on the figure) or
sampled at fewer than all depths if the cells do exist (drawn as **markers
with error bars and no connecting line**, with the legend saying so). The
interior is not measured either way, and a solid curve through it would
assert a shape the data does not contain.

Usage:
    uv run python scripts/plot_rock_sample_depth.py \\
        --input-dir output/rock_sample_depth \\
        --output output/rock_sample_depth/figure.pdf \\
        --horizon 15
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

    NaN-safe: reward_normalised is NaN on episodes with no good rock.
    """
    per_ep = np.asarray(summary["per_episode"][metric], dtype=float)
    per_seed = np.nanmean(per_ep, axis=1)
    n = max(np.count_nonzero(~np.isnan(per_seed)), 1)
    sem = np.nanstd(per_seed, ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return float(np.nanmean(per_seed)), float(1.96 * sem)


def _terminal_curve_value(summary: dict, metric: str) -> float:
    """Across-seed mean of `metric` at the last episode of the block."""
    return float(np.asarray(summary["curves"][metric]["mean"], dtype=float)[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RockSample MCTS depth-sensitivity ablation.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/rock_sample_depth"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="AIF-MP planning horizon to mark on the depth axis "
             "(params rock_sample.runner.horizon)",
    )
    args = parser.parse_args()

    depths = _discover_depths(args.input_dir)
    if not depths:
        raise FileNotFoundError(
            f"no depth<D> subdirectories under {args.input_dir}; "
            f"run `dvc repro rock_sample_depth` first"
        )

    # Per-planner depth lists: the slate is deliberately ragged (see the
    # module docstring), so a planner may legitimately be absent at some
    # depths. Two cells is the floor — one point is not a sensitivity
    # measurement, and the DVC stage enumerates the exact expected files,
    # so a genuinely missing cell fails before this script runs.
    runs: dict[tuple[int, str], dict] = {}
    planner_depths: dict[str, list[int]] = {}
    for planner in PLANNERS:
        sampled: list[int] = []
        for depth in depths:
            path = args.input_dir / f"depth{depth}" / f"{planner}.json"
            if not path.exists():
                continue
            with path.open() as f:
                runs[(depth, planner)] = json.load(f)
            sampled.append(depth)
        if not sampled:
            # No cells at all means the planner was deliberately left out of
            # the ablation, not that a run failed: IDS has no depth cells on
            # cost grounds (see params.yaml). Draw what exists and name the
            # omission on the figure.
            continue
        if len(sampled) < 2:
            raise FileNotFoundError(
                f"{planner}: found 1 depth cell under {args.input_dir}; "
                f"expected either none (omitted from the slate) or at least "
                f"2 (one point is not a sensitivity measurement). Run "
                f"`dvc repro rock_sample_depth` first"
            )
        planner_depths[planner] = sampled
    if "bamcp" not in planner_depths:
        raise FileNotFoundError(
            f"no bamcp depth cells under {args.input_dir}; BAMCP carries the "
            f"depth curve, so the figure cannot be drawn without it. Run "
            f"`dvc repro rock_sample_depth` first"
        )

    theme.apply_rcparams()
    _first = next(iter(planner_depths))
    cfg = runs[(planner_depths[_first][0], _first)]["config"]
    # Depth is an ordered sweep, so per-depth markers wear the theme's
    # ordinal teal ramp; planners are told apart by linestyle.
    depth_colors = dict(zip(depths, theme.ordinal_colors(len(depths))))
    linestyles = {"bamcp": "-", "ids": "--"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_reward, ax_sense, ax_entropy = axes
    fig.suptitle(
        f"RockSample MCTS depth sensitivity — K={cfg['env']['K']}, "
        f"grid={cfg['env']['grid_size']}, "
        f"n_seeds={cfg['n_seeds']}, n_episodes={cfg['n_episodes']} "
        f"(depth bounds tree + rollout lookahead)",
        fontsize=11,
    )

    panels = (
        (ax_reward, "reward_normalised", False,
         "block-mean reward / max_possible", "Normalised reward"),
        (ax_sense, "n_senses", False,
         "SENSE actions / episode", "Sensing effort"),
        (ax_entropy, "posterior_entropy_persistent", True,
         "H(q(θ_sensor)) at final episode", "Terminal persistent entropy"),
    )

    for ax, metric, terminal, ylabel, title in panels:
        for planner in planner_depths:
            p_depths = planner_depths[planner]
            if terminal:
                means = np.array(
                    [_terminal_curve_value(runs[(d, planner)], metric)
                     for d in p_depths]
                )
                errs = np.zeros_like(means)
            else:
                stats = [
                    _block_mean(runs[(d, planner)], metric) for d in p_depths
                ]
                means = np.array([m for m, _ in stats])
                errs = np.array([e for _, e in stats])
            if len(p_depths) < len(depths):
                # Endpoint-only sampling: error bars plus the markers drawn
                # below, and NO connecting line — the interior depths were
                # never run. (The legend lives on a non-terminal panel, so
                # the label only needs to be attached there.)
                if not terminal:
                    ax.errorbar(
                        p_depths, means, yerr=errs, fmt="none",
                        ecolor=theme.planner_color(planner),
                        elinewidth=1.5, capsize=4, zorder=2,
                        label=f"{PLANNER_LABELS[planner]} "
                              f"(endpoints only, no line)",
                    )
            else:
                ax.plot(
                    depths, means, color=theme.planner_color(planner),
                    linestyle=linestyles[planner], lw=2, zorder=2,
                    label=PLANNER_LABELS[planner],
                )
                if not terminal:
                    ax.fill_between(
                        depths, means - errs, means + errs,
                        color=theme.planner_color(planner), alpha=0.12,
                    )
            # Square markers for a partially sampled planner, so its
            # points cannot be misread as points on a sampled curve.
            marker = "o" if len(p_depths) == len(depths) else "s"
            for d, m in zip(p_depths, means):
                ax.plot(
                    [d], [m], marker=marker, ms=6, zorder=3,
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
        suffix = "" if terminal else " (mean ± 1.96·SEM)"
        ax.set_title(f"{title}{suffix}")
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
    ax_reward.axhline(
        1.0, color=theme.PALETTE["ink"], lw=0.5, ls=":", alpha=0.5,
    )
    ax_reward.legend(loc="best", fontsize=9)
    notes = [
        f"{PLANNER_LABELS[p]} at depths {planner_depths[p]} only "
        f"(markers, no interpolating line)"
        for p in planner_depths if len(planner_depths[p]) < len(depths)
    ]
    omitted = [PLANNER_LABELS[p] for p in PLANNERS if p not in planner_depths]
    if omitted:
        notes.append(
            f"{', '.join(omitted)} not run — per-seed cost ~110x BAMCP's and "
            f"depth-invariant (8.07 h -> 8.18 h across this sweep), so the "
            f"invariance is reported from that timing, not measured here"
        )
    if notes:
        # State the reduced sampling on the figure itself, not only in the
        # legend entry — a reader skimming panels must not read two markers
        # as a sampled curve, nor a missing planner as an untested one.
        fig.text(
            0.5, 0.005,
            f"Reduced sampling: {'; '.join(notes)}. "
            f"See params.yaml rock_sample_depth.",
            ha="center", fontsize=8, color=theme.PALETTE["ink"],
        )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
