"""Publication figures for the paper (distilled from the diagnostic dashboards).

Produces one PDF per panel so the paper can compose them freely as
subfigures:

  <env>_return.pdf       per-episode return, mean +/- 95% CI of the mean
                         across seeds (from the raw per_episode arrays)
  <env>_mass.pdf         posterior mass on the true (persistent) theta
  results_legend.pdf     standalone horizontal legend shared by the panels
  results_kscaling.pdf   final-episode normalised reward vs K on the
                         RockSample K-scaling sweep

Panels carry no titles; environment names go in LaTeX subcaptions.
Only the paper's baseline slate is plotted (no VBP / Belief-Q).
Colors come from the repo-root dissertation theme (thesis cover palette):
one hue per planner entity; BAMCP variants (POMCP, Factored BA-POMCP)
share BAMCP's plum and are told apart by linestyle.

Usage:
    uv run python scripts/plot_paper_figures.py \\
        --input-root ../output --output-dir ../figures
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
from matplotlib.lines import Line2D

# Repo-root dissertation theme (single source of truth for figure colors).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402


PLANNER_LABELS = {
    "aif": "AIF-MP",
    "psrl": "PSRL",
    "bamcp": "BAMCP",
    "pomcp": "POMCP",
    "boss": "BOSS",
    "factored_bamcp": "Factored BA-POMCP",
    "ids": "IDS",
    "oracle": r"Known-$\theta$ VBP",
    "random": "Random",
}
PLANNER_COLORS = {name: theme.planner_color(name) for name in PLANNER_LABELS}
# Bounds dashed; the BAMCP variants (POMCP, Factored BA-POMCP) share its plum
# and are told apart by dotted / dash-dot linestyles.
PLANNER_LINESTYLES = dict(theme.PLANNER_LINESTYLES)
PLANNER_ORDER = (
    "aif", "psrl", "bamcp", "pomcp", "boss", "factored_bamcp", "ids",
    "oracle", "random",
)

# (dirname, identification metric in per_episode, return ylim)
# Factored Tiger returns are clipped: BOSS / PSRL means reach -274 / -150 in
# episode 1, which would crush the discriminating region (state in caption).
ENVS = (
    ("door_code", "posterior_mass_on_true_theta", None),
    ("minigrid_door_code", "posterior_mass_on_true_theta", None),
    ("factored_tiger", "posterior_mass_on_persistent_theta", (-60, 35)),
    ("rock_sample", "posterior_mass_on_persistent_theta", None),
)

PANEL_SIZE = (2.4, 1.9)

RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _load_env(env_dir: Path) -> dict[str, dict]:
    """Load every paper-slate planner with a result file in env_dir."""
    runs = {}
    for name in PLANNER_ORDER:
        path = env_dir / f"{name}.json"
        if path.exists():
            with path.open() as f:
                runs[name] = json.load(f)
    if not runs:
        raise FileNotFoundError(f"no planner results in {env_dir}")
    return runs


def _mean_ci(per_seed: list) -> tuple[np.ndarray, np.ndarray]:
    """Mean and half-width of the 95% CI of the mean, NaN-safe.

    per_seed: (n_seeds, n_episodes) raw per-episode values.
    """
    arr = np.asarray(per_seed, dtype=float)
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, 1.96 * sem


def _plot_curves(ax, runs: dict[str, dict], metric: str) -> None:
    for name, s in runs.items():
        mean, ci = _mean_ci(s["per_episode"][metric])
        eps = np.arange(1, len(mean) + 1)
        ls = PLANNER_LINESTYLES.get(name, "-")
        c = PLANNER_COLORS[name]
        ax.plot(eps, mean, color=c, lw=1.4, linestyle=ls)
        if ls == "-":
            ax.fill_between(eps, mean - ci, mean + ci, color=c, alpha=0.18,
                            lw=0)
    ax.set_xlim(1, len(mean))
    ax.grid(True, alpha=0.25, lw=0.4)


def _save_panel(
    runs: dict, metric: str, ylabel: str, ylim, output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    _plot_curves(ax, runs, metric)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    fig.tight_layout(pad=0.3)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {output}", flush=True)


def plot_envs(input_root: Path, output_dir: Path) -> None:
    for dirname, mass_metric, reward_ylim in ENVS:
        runs = _load_env(input_root / dirname)
        _save_panel(
            runs, "cumulative_reward", "episode return", reward_ylim,
            output_dir / f"{dirname}_return.pdf",
        )
        # Oracle is excluded from the mass panel: it plans on a one-hot
        # at theta-star, so the logged filter mass only measures what a
        # Bayesian filter would learn from *watching* purely exploitative
        # behaviour — a reward bound, not an identification baseline.
        # Random stays: it is a meaningful passive-identification floor.
        _save_panel(
            {n: s for n, s in runs.items() if n != "oracle"},
            mass_metric, r"$q(\theta = \theta^\star)$", (-0.02, 1.02),
            output_dir / f"{dirname}_mass.pdf",
        )


def plot_legend(output: Path) -> None:
    handles = [
        Line2D([0], [0], color=PLANNER_COLORS[n], lw=1.4,
               linestyle=PLANNER_LINESTYLES.get(n, "-"))
        for n in PLANNER_ORDER
    ]
    labels = [PLANNER_LABELS[n] for n in PLANNER_ORDER]
    fig = plt.figure(figsize=(9.0, 0.25))
    fig.legend(handles, labels, loc="center", ncol=len(labels),
               frameon=False, columnspacing=1.2, handlelength=1.6)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {output}", flush=True)


def plot_kscaling(input_root: Path, output: Path) -> None:
    k_dirs = sorted(input_root.glob("K*"))
    ks = [int(d.name[1:]) for d in k_dirs]
    runs_by_k = [_load_env(d) for d in k_dirs]
    planners = [n for n in PLANNER_ORDER if all(n in r for r in runs_by_k)]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, name in enumerate(planners):
        means, cis = [], []
        for runs in runs_by_k:
            arr = np.asarray(
                runs[name]["per_episode"]["reward_normalised"], dtype=float,
            )[:, -1]  # final episode of the block
            m, ci = _mean_ci(arr[:, None])
            means.append(m[0])
            cis.append(ci[0])
        jitter = (i - (len(planners) - 1) / 2) * 0.04
        ls = PLANNER_LINESTYLES.get(name, "-")
        ax.errorbar(np.array(ks) + jitter, means, yerr=cis,
                    color=PLANNER_COLORS[name], lw=1.4, linestyle=ls,
                    marker="o", ms=3, capsize=2,
                    label=PLANNER_LABELS[name])
    ax.set_xticks(ks)
    ax.set_xlabel(r"number of parameter components $K$")
    ax.set_ylabel("final-episode normalised reward")
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication figures.")
    parser.add_argument("--input-root", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theme.apply_rcparams()
    with plt.rc_context(RC):
        plot_envs(args.input_root, args.output_dir)
        plot_legend(args.output_dir / "results_legend.pdf")
        plot_kscaling(
            args.input_root / "rock_sample_kscaling",
            args.output_dir / "results_kscaling.pdf",
        )


if __name__ == "__main__":
    main()
