"""Plot the Factored Tiger planner comparison.

Reads output/factored_tiger/{aif,vbp,psrl,bamcp,boss,factored_bamcp,ids,
oracle,random}.json and produces a 3×3 figure:

  Row 1 (performance + epistemic):
    - cumulative reward vs episode (mean ± 90% CI)
    - reward / max_possible (∈ [0,1]; NaN-safe across-seed mean)
    - persistent posterior entropy q(θ) — across-block curve

  Row 2 (action diagnostics + truth mass):
    - mass on the TRUE persistent θ
    - safe  opens / episode
    - tiger opens / episode

  Row 3 (paper-grade scalars):
    - cumulative regret vs oracle (sum of per-episode reward gaps)
    - planning wall-clock per step (median, log scale)
    - Brier score on q(persistent θ)

Belief-Q is omitted (door-code-specific belief feature). The episodic
posterior is uniformly trivial here (Factored Tiger has only persistent
θ), so we do not plot it.

Usage:
    uv run python scripts/plot_factored_tiger.py \\
        --input-dir output/factored_tiger \\
        --output output/factored_tiger/figure.pdf
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


def _load(input_dir: Path) -> dict[str, dict]:
    out = {}
    for name in PLANNER_ORDER:
        path = input_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run `dvc repro` first")
        with path.open() as f:
            out[name] = json.load(f)
    return out


def _curve_with_band(
    ax, summary: dict, metric: str, color: str, label: str,
    linestyle: str = "-", group: str = "curves", marker: str = "",
):
    curve = summary[group][metric]
    mean = np.array(curve["mean"])
    p05 = np.array(curve["p05"])
    p95 = np.array(curve["p95"])
    eps = np.arange(len(mean))
    # Sparse markers double as a non-color encoding (CVD / greyscale print).
    ax.plot(eps, mean, color=color, label=label, lw=2, linestyle=linestyle,
            marker=marker, markevery=max(len(mean) // 10, 1), ms=4)
    if linestyle == "-":
        ax.fill_between(eps, p05, p95, color=color, alpha=0.15)


def _cumulative_regret(summary: dict, oracle_summary: dict) -> np.ndarray:
    """Per-seed cumulative regret = cumsum(oracle.mean - planner) over
    episodes. Uses `cumulative_reward` (per-episode total, not terminal),
    which is the canonical reward signal in dense-reward envs."""
    planner = np.asarray(
        summary["per_episode"]["cumulative_reward"], dtype=float,
    )
    oracle = np.asarray(
        oracle_summary["per_episode"]["cumulative_reward"], dtype=float,
    )
    oracle_mean = oracle.mean(axis=0)
    return np.cumsum(oracle_mean[None, :] - planner, axis=1)


def _planning_time_per_step(summary: dict) -> float:
    pt = np.asarray(summary["per_episode"]["planning_time_s"], dtype=float)
    n_steps = np.asarray(summary["per_episode"]["n_steps"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_step = np.where(n_steps > 0, pt / n_steps, np.nan)
    finite = per_step[np.isfinite(per_step)]
    return float(np.median(finite)) if finite.size else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Factored Tiger comparison.")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/factored_tiger"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    theme.apply_rcparams()

    runs = _load(args.input_dir)
    cfg = runs[PLANNER_ORDER[0]]["config"]
    n_episodes = cfg["n_episodes"]
    n_seeds = cfg["n_seeds"]
    env_cfg = cfg["env"]
    K = env_cfg["K"]
    listen_accuracy = env_cfg["listen_accuracy"]
    max_steps = cfg["max_steps"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    title = (
        f"Factored Tiger: K={K}, listen_accuracy={listen_accuracy}, "
        f"max_steps={max_steps}, n_seeds={n_seeds}, n_episodes={n_episodes}"
    )
    fig.suptitle(title, fontsize=11)

    (ax_reward, ax_norm, ax_persistent) = axes[0]
    (ax_mass, ax_safe, ax_tiger) = axes[1]
    (ax_regret, ax_ptime, ax_brier) = axes[2]

    oracle_run = runs["oracle"]
    eps_axis = np.arange(n_episodes)

    for name in PLANNER_ORDER:
        s = runs[name]
        c = PLANNER_COLORS[name]
        lab = PLANNER_LABELS[name]
        ls = PLANNER_LINESTYLES.get(name, "-")
        mk = PLANNER_MARKERS.get(name, "")
        _curve_with_band(ax_reward, s, "cumulative_reward", c, lab, linestyle=ls, marker=mk)
        _curve_with_band(ax_norm, s, "reward_normalised", c, lab, linestyle=ls, marker=mk)
        _curve_with_band(
            ax_persistent, s, "posterior_entropy_persistent", c, lab,
            linestyle=ls, marker=mk,
        )
        # Oracle is excluded from the identification panels (mass, Brier):
        # it plans on a one-hot at theta-star, so the logged filter mass
        # only measures what a Bayesian filter would learn from watching
        # purely exploitative behaviour — a reward bound, not an
        # identification baseline. Random stays (passive floor).
        if name != "oracle":
            _curve_with_band(
                ax_mass, s, "posterior_mass_on_persistent_theta", c, lab,
                linestyle=ls, marker=mk,
            )
        _curve_with_band(ax_safe, s, "n_safe_opens", c, lab, linestyle=ls, marker=mk)
        _curve_with_band(ax_tiger, s, "n_tiger_opens", c, lab, linestyle=ls, marker=mk)

        regret = _cumulative_regret(s, oracle_run)
        regret_mean = regret.mean(axis=0)
        regret_p05 = np.percentile(regret, 5, axis=0)
        regret_p95 = np.percentile(regret, 95, axis=0)
        ax_regret.plot(eps_axis, regret_mean, color=c, label=lab, lw=2, linestyle=ls,
                       marker=mk, markevery=max(len(eps_axis) // 10, 1), ms=4)
        if ls == "-":
            ax_regret.fill_between(eps_axis, regret_p05, regret_p95, color=c, alpha=0.15)

        if "derived" in s and name != "oracle":
            _curve_with_band(ax_brier, s, "brier", c, lab, linestyle=ls,
                             marker=mk, group="derived")

    ax_reward.set_xlabel("episode within block")
    ax_reward.set_ylabel("cumulative reward / episode")
    ax_reward.set_title("Cumulative reward (mean ± 90% CI)")
    ax_reward.legend(loc="lower right", fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    ax_norm.set_xlabel("episode within block")
    ax_norm.set_ylabel("reward / max_possible")
    ax_norm.set_title("Normalised reward (1.0 = optimal play)")
    ax_norm.set_ylim(-0.1, 1.1)
    ax_norm.axhline(1.0, color=theme.PALETTE["ink"], lw=0.5, ls=":", alpha=0.5)
    ax_norm.grid(True, alpha=0.3)

    ax_persistent.set_xlabel("episode within block")
    ax_persistent.set_ylabel("H(q(θ))")
    ax_persistent.set_title("Persistent θ entropy (across-block)")
    ax_persistent.grid(True, alpha=0.3)

    ax_mass.set_xlabel("episode within block")
    ax_mass.set_ylabel("q(θ = truth)")
    ax_mass.set_title("Mass on TRUE persistent θ")
    ax_mass.set_ylim(-0.05, 1.05)
    ax_mass.axhline(1.0, color=theme.PALETTE["ink"], lw=0.5, ls=":", alpha=0.5)
    ax_mass.grid(True, alpha=0.3)

    ax_safe.set_xlabel("episode within block")
    ax_safe.set_ylabel("safe opens / ep")
    ax_safe.set_title("Safe doors opened (good)")
    ax_safe.grid(True, alpha=0.3)

    ax_tiger.set_xlabel("episode within block")
    ax_tiger.set_ylabel("tiger opens / ep")
    ax_tiger.set_title("Tiger doors opened (bad)")
    ax_tiger.grid(True, alpha=0.3)

    ax_regret.set_xlabel("episode within block")
    ax_regret.set_ylabel("cumulative regret vs oracle")
    ax_regret.set_title("Cumulative regret (mean ± 90% CI)")
    ax_regret.axhline(0, color=theme.PALETTE["ink"], lw=0.5, alpha=0.4)
    ax_regret.grid(True, alpha=0.3)

    xs = np.arange(len(PLANNER_ORDER))
    bar_colors = [PLANNER_COLORS[n] for n in PLANNER_ORDER]
    labs = [PLANNER_LABELS[n] for n in PLANNER_ORDER]
    ptime_med = [_planning_time_per_step(runs[n]) for n in PLANNER_ORDER]
    ax_ptime.bar(xs, ptime_med, color=bar_colors, alpha=0.85)
    ax_ptime.set_xticks(xs)
    ax_ptime.set_xticklabels(labs, rotation=30, ha="right")
    ax_ptime.set_ylabel("seconds / planning step (median)")
    ax_ptime.set_title("Planning wall-clock cost")
    ax_ptime.set_yscale("log")
    ax_ptime.grid(True, alpha=0.3, axis="y", which="both")

    ax_brier.set_xlabel("episode within block")
    ax_brier.set_ylabel("Brier score on q(persistent θ)")
    ax_brier.set_title("Posterior calibration (Brier)")
    ax_brier.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
