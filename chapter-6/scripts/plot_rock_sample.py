"""Plot the RockSample planner comparison (M4).

Reads output/rock_sample/{aif,vbp,psrl,bamcp,boss,factored_bamcp,oracle,
random}.json and produces a 3×4 figure:

  Row 1 (performance + epistemic):
    - cumulative reward vs episode (mean ± 90% CI)
    - reward / max_possible (∈ [0,1]; NaN-safe across-seed mean)
    - persistent posterior entropy q(θ_sensor) — across-block curve
    - episodic posterior entropy q(θ_quality) — per-episode end

  Row 2 (action diagnostics + truth mass):
    - mass on the TRUE persistent θ (sensor calibration recovery)
    - good rocks sampled / episode
    - bad  rocks sampled / episode
    - rocks consumed / episode (good + bad combined)

  Row 3 (paper-grade scalars):
    - cumulative regret vs oracle (sum of per-episode reward gaps)
    - planning wall-clock per step (median, log scale)
    - Brier score on q(persistent θ)
    - blank (reserved for a fourth scalar; intentionally left empty)

BeliefQ is omitted: its belief feature (per-stage permutation argmax,
``belief_state_q.py:107-132``) is door-code-specific and is deferred
to a follow-up env-agnostic feature redesign.

BAMCP and Factored BA-POMCP curves typically overlap on this env: the
persistent prior factorises across rocks by construction (independent
rocks), so per-factor sampling = flat-marginal sampling
distributionally. This is informative for the paper claim — even
sample-based methods that explicitly exploit factor structure cannot
match exact factor-aware BP.

Usage:
    uv run python scripts/plot_rock_sample.py \\
        --input-dir output/rock_sample \\
        --output output/rock_sample/figure.pdf
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

from _cards import save_legend_card, save_panel_cards  # noqa: E402


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
    # 90% CI of the *mean* over seeds (not of per-seed values): mean ± 1.645·SE,
    # from the raw per-episode curves. Falls back to the stored percentile band
    # for any metric without raw per-seed data.
    pe = summary.get("per_episode", {})
    raw = None
    if metric in pe:
        raw = np.asarray(pe[metric], dtype=float)
    elif metric == "brier":
        mass_key = next((k for k in ("posterior_mass_on_true_theta",
                                     "posterior_mass_on_persistent_theta")
                         if k in pe), None)
        if mass_key is not None:
            raw = (1.0 - np.asarray(pe[mass_key], dtype=float)) ** 2
    if raw is not None:
        mean = raw.mean(axis=0)
        band = 1.645 * raw.std(axis=0, ddof=1) / np.sqrt(raw.shape[0])
        p05, p95 = mean - band, mean + band
    else:
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
    parser = argparse.ArgumentParser(description="Plot RockSample comparison.")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/rock_sample"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cards-dir", type=Path, default=None,
        help="if set, also write each panel as its own PDF card here",
    )
    args = parser.parse_args()
    theme.apply_rcparams()

    runs = _load(args.input_dir)
    cfg = runs[PLANNER_ORDER[0]]["config"]
    n_episodes = cfg["n_episodes"]
    n_seeds = cfg["n_seeds"]
    env_cfg = cfg["env"]
    K = env_cfg["K"]
    grid_size = env_cfg["grid_size"]
    eta = env_cfg["sensor_noise_eta"]
    max_steps = cfg["max_steps"]

    fig, axes = plt.subplots(3, 4, figsize=(20, 13.5))
    title = (
        f"RockSample: K={K}, grid={grid_size}x{grid_size}, "
        f"sensor_eta={eta}, max_steps={max_steps}, "
        f"n_seeds={n_seeds}, n_episodes={n_episodes}"
    )
    fig.suptitle(title, fontsize=11)

    (ax_reward, ax_norm, ax_persistent, ax_episodic) = axes[0]
    (ax_mass, ax_good, ax_bad, ax_samples) = axes[1]
    (ax_regret, ax_ptime, ax_brier, ax_blank) = axes[2]
    ax_blank.axis("off")

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
        _curve_with_band(
            ax_episodic, s, "posterior_entropy_episodic", c, lab,
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
        _curve_with_band(ax_good, s, "n_good_sampled", c, lab, linestyle=ls, marker=mk)
        _curve_with_band(ax_bad, s, "n_bad_sampled", c, lab, linestyle=ls, marker=mk)
        _curve_with_band(
            ax_samples, s, "n_samples_successful", c, lab, linestyle=ls, marker=mk,
        )

        regret = _cumulative_regret(s, oracle_run)
        regret_mean = regret.mean(axis=0)
        regret_band = 1.645 * regret.std(axis=0, ddof=1) / np.sqrt(regret.shape[0])
        regret_p05 = regret_mean - regret_band
        regret_p95 = regret_mean + regret_band
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
    ax_persistent.set_ylabel("H(q(θ_sensor))")
    ax_persistent.set_title("Persistent θ entropy (across-block)")
    ax_persistent.grid(True, alpha=0.3)

    ax_episodic.set_xlabel("episode within block")
    ax_episodic.set_ylabel("H(q(θ_quality))")
    ax_episodic.set_title("Episodic θ entropy (per-episode end)")
    ax_episodic.grid(True, alpha=0.3)

    ax_mass.set_xlabel("episode within block")
    ax_mass.set_ylabel("q(θ_sensor = truth)")
    ax_mass.set_title("Mass on TRUE persistent θ")
    ax_mass.set_ylim(-0.05, 1.05)
    ax_mass.axhline(1.0, color=theme.PALETTE["ink"], lw=0.5, ls=":", alpha=0.5)
    ax_mass.grid(True, alpha=0.3)

    ax_good.set_xlabel("episode within block")
    ax_good.set_ylabel("good rocks sampled / ep")
    ax_good.set_title("Good rocks correctly sampled")
    ax_good.grid(True, alpha=0.3)

    ax_bad.set_xlabel("episode within block")
    ax_bad.set_ylabel("bad rocks sampled / ep")
    ax_bad.set_title("Bad rocks mistakenly sampled")
    ax_bad.grid(True, alpha=0.3)

    ax_samples.set_xlabel("episode within block")
    ax_samples.set_ylabel("rocks consumed / ep")
    ax_samples.set_title("Rocks consumed per episode (good + bad)")
    ax_samples.grid(True, alpha=0.3)

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
    print(f"figure saved to {args.output}", flush=True)

    if args.cards_dir is not None:
        named_axes = [
            ("reward", ax_reward), ("norm", ax_norm),
            ("persistent", ax_persistent), ("episodic", ax_episodic),
            ("mass", ax_mass), ("good", ax_good), ("bad", ax_bad),
            ("samples", ax_samples), ("regret", ax_regret),
            ("ptime", ax_ptime), ("brier", ax_brier),
        ]
        handles, labels = ax_reward.get_legend_handles_labels()
        if ax_reward.get_legend() is not None:
            ax_reward.get_legend().remove()
        save_panel_cards(fig, named_axes, args.cards_dir, "rock_sample")
        save_legend_card(handles, labels,
                         args.cards_dir / "rock_sample_legend.pdf")
        print(f"cards saved to {args.cards_dir}", flush=True)

    plt.close(fig)


if __name__ == "__main__":
    main()
