"""Plot the RockSample K-scaling figure.

Aggregates output/rock_sample_kscaling/K<K>/<planner>.json over the
K values defined in params.rock_sample.k_sweep and plots:

  Left:   reward_normalised (block-mean, mean ± 90% CI across seeds)
          per planner vs K. Shows the absolute performance curve as
          #factors grows.
  Middle: AIF − BAMCP and AIF − Factored-BAMCP planning-time gap
          (block-mean reward_normalised difference) vs K. The
          load-bearing visual evidence for "exact factor-aware BP beats
          sample-based factor-aware tree search as #factors grows".
  Right:  posterior_entropy_persistent at the final episode in the
          block, per planner vs K. Should remain low for AIF/Oracle and
          grow with K for weaker planners.

The aggregate per (planner, K) collapses across episodes (mean over
the block) and seeds (mean across the m outer-loop draws).

Usage:
    uv run python scripts/plot_rock_sample_kscaling.py \\
        --input-dir output/rock_sample_kscaling \\
        --output output/rock_sample_kscaling/figure.pdf
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


PLANNER_LABELS = {
    "aif": "AIF",
    "vbp": "VBP",
    "psrl": "PSRL",
    "bamcp": "BAMCP",
    "boss": "BOSS",
    "factored_bamcp": "Factored BA-POMCP",
    "ids": "IDS",
    "oracle": "Oracle",
    "random": "Random",
}
PLANNER_COLORS = {name: theme.planner_color(name) for name in PLANNER_LABELS}
PLANNER_MARKERS = {name: theme.planner_marker(name) for name in PLANNER_LABELS}
# Bounds dashed; Factored BA-POMCP shares BAMCP's plum (it is a BAMCP
# variant) and is told apart by a dash-dot linestyle.
PLANNER_LINESTYLES = dict(theme.PLANNER_LINESTYLES)
PLANNER_ORDER = (
    "aif", "vbp", "psrl", "bamcp", "boss", "factored_bamcp", "ids",
    "oracle", "random",
)


def _discover_K_values(input_dir: Path) -> list[int]:
    """List K subdirectories under input_dir, sorted ascending."""
    ks: list[int] = []
    for child in input_dir.iterdir():
        if child.is_dir() and child.name.startswith("K"):
            try:
                ks.append(int(child.name[1:]))
            except ValueError:
                continue
    return sorted(ks)


def _block_mean_reward_normalised(summary: dict) -> tuple[float, float, float]:
    """Mean ± 90% CI of reward_normalised, averaged across the block."""
    per_ep = np.array(summary["per_episode"]["reward_normalised"])
    # Shape (n_seeds, n_episodes). Average across episodes -> per-seed scalar,
    # then aggregate across seeds (NaN-safe).
    per_seed = np.nanmean(per_ep, axis=1)  # (n_seeds,)
    return (
        float(np.nanmean(per_seed)),
        float(np.nanpercentile(per_seed, 5)),
        float(np.nanpercentile(per_seed, 95)),
    )


def _terminal_persistent_entropy(summary: dict) -> float:
    """Mean across-seed posterior_entropy_persistent at the last episode."""
    curve = np.array(summary["curves"]["posterior_entropy_persistent"]["mean"])
    return float(curve[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RockSample K-scaling.")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/rock_sample_kscaling"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    theme.apply_rcparams()

    k_values = _discover_K_values(args.input_dir)
    if not k_values:
        raise FileNotFoundError(
            f"no K<K> subdirectories under {args.input_dir}; "
            f"run `dvc repro rock_sample_kscaling` first"
        )

    # Aggregate per (planner, K).
    reward_means: dict[str, list[float]] = {p: [] for p in PLANNER_ORDER}
    reward_p05: dict[str, list[float]] = {p: [] for p in PLANNER_ORDER}
    reward_p95: dict[str, list[float]] = {p: [] for p in PLANNER_ORDER}
    entropy_terminal: dict[str, list[float]] = {p: [] for p in PLANNER_ORDER}

    for K in k_values:
        for planner in PLANNER_ORDER:
            path = args.input_dir / f"K{K}" / f"{planner}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path}; run `dvc repro rock_sample_kscaling` first"
                )
            with path.open() as f:
                summary = json.load(f)
            mean, p05, p95 = _block_mean_reward_normalised(summary)
            reward_means[planner].append(mean)
            reward_p05[planner].append(p05)
            reward_p95[planner].append(p95)
            entropy_terminal[planner].append(_terminal_persistent_entropy(summary))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_reward, ax_gap, ax_entropy = axes

    title = (
        f"RockSample K-scaling — block-mean normalised reward and "
        f"planning-time gap (K ∈ {k_values})"
    )
    fig.suptitle(title, fontsize=11)

    # Left: reward_normalised vs K.
    for planner in PLANNER_ORDER:
        c = PLANNER_COLORS[planner]
        ls = PLANNER_LINESTYLES.get(planner, "-")
        means = np.array(reward_means[planner])
        p05 = np.array(reward_p05[planner])
        p95 = np.array(reward_p95[planner])
        ax_reward.plot(
            k_values, means, color=c, label=PLANNER_LABELS[planner],
            lw=2, linestyle=ls, marker=PLANNER_MARKERS.get(planner) or "o",
        )
        if ls == "-":
            ax_reward.fill_between(k_values, p05, p95, color=c, alpha=0.12)
    ax_reward.set_xlabel("K (number of rocks / θ factors)")
    ax_reward.set_ylabel("block-mean reward / max_possible")
    ax_reward.set_title("Normalised reward vs K (mean ± 90% CI)")
    ax_reward.set_ylim(-0.1, 1.1)
    ax_reward.axhline(1.0, color=theme.PALETTE["ink"], lw=0.5, ls=":", alpha=0.5)
    ax_reward.legend(loc="lower left", fontsize=8, ncol=2)
    ax_reward.grid(True, alpha=0.3)

    # Middle: AIF − {BAMCP, Factored BA-POMCP, BOSS, PSRL} gap vs K.
    aif_means = np.array(reward_means["aif"])
    gap_baselines = ["bamcp", "factored_bamcp", "boss", "psrl", "ids"]
    for planner in gap_baselines:
        if planner not in reward_means:
            continue
        c = PLANNER_COLORS[planner]
        gap = aif_means - np.array(reward_means[planner])
        ax_gap.plot(
            k_values, gap, color=c, lw=2,
            marker=PLANNER_MARKERS.get(planner) or "o",
            linestyle=PLANNER_LINESTYLES.get(planner, "-"),
            label=f"AIF − {PLANNER_LABELS[planner]}",
        )
    ax_gap.axhline(0.0, color=theme.PALETTE["ink"], lw=0.5, ls=":", alpha=0.5)
    ax_gap.set_xlabel("K (number of rocks / θ factors)")
    ax_gap.set_ylabel("AIF − baseline (normalised reward)")
    ax_gap.set_title("Planning-time gap vs K")
    ax_gap.legend(loc="best", fontsize=9)
    ax_gap.grid(True, alpha=0.3)

    # Right: terminal persistent entropy vs K.
    for planner in PLANNER_ORDER:
        c = PLANNER_COLORS[planner]
        ls = PLANNER_LINESTYLES.get(planner, "-")
        ax_entropy.plot(
            k_values, entropy_terminal[planner], color=c,
            label=PLANNER_LABELS[planner], lw=2, linestyle=ls,
            marker=PLANNER_MARKERS.get(planner) or "o",
        )
    ax_entropy.set_xlabel("K (number of rocks / θ factors)")
    ax_entropy.set_ylabel("H(q(θ_sensor)) at final episode")
    ax_entropy.set_title("Terminal persistent posterior entropy")
    ax_entropy.legend(loc="best", fontsize=8, ncol=2)
    ax_entropy.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
