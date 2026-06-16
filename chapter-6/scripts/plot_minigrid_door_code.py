"""Plot the spatial minigrid door-code planner comparison.

Reads output/minigrid_door_code/{aif,vbp,psrl,belief_q,boss,bamcp,
pomcp,oracle,random}.json and produces a single multi-panel figure
(Oracle and Random drawn as dashed bounds). Eight panels in a 2x4 grid:

  Row 1: success | posterior mass on true theta | probe visits |
         outcome breakdown (deaths vs truncations)
  Row 2: cumulative regret vs oracle | planning wall-clock per step |
         Brier score on q(theta) | episodes-to-identify (median + IQR)

Usage:
    uv run python scripts/plot_minigrid_door_code.py \\
        --input-dir output/minigrid_door_code \\
        --output output/minigrid_door_code/figure.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLANNER_LABELS = {
    "aif": "AIF",
    "vbp": "VBP",
    "psrl": "PSRL",
    "belief_q": "Belief-Q",
    "boss": "BOSS",
    "bamcp": "BAMCP",
    "pomcp": "POMCP (literal)",
    "ids": "IDS",
    "oracle": "Oracle",
    "random": "Random",
}
PLANNER_COLORS = {
    "aif": "#1f77b4",
    "vbp": "#ff7f0e",
    "psrl": "#2ca02c",
    "belief_q": "#d62728",
    "boss": "#9467bd",
    "bamcp": "#8c564b",
    "pomcp": "#e377c2",
    "ids": "#17becf",
    "oracle": "#000000",
    "random": "#7f7f7f",
}
# Bounds (Oracle / Random) drawn as dashed lines; methods solid.
PLANNER_LINESTYLES = {name: "--" for name in ("oracle", "random")}
PLANNER_ORDER = (
    "aif", "vbp", "psrl", "belief_q", "boss", "bamcp", "pomcp", "ids",
    "oracle", "random",
)


def _load(input_dir: Path) -> dict[str, dict]:
    """Load whichever planner JSONs are present in `input_dir`. Missing
    files are skipped so the same plot script handles full slates and
    small ablations."""
    out = {}
    for name in PLANNER_ORDER:
        path = input_dir / f"{name}.json"
        if not path.exists():
            continue
        with path.open() as f:
            out[name] = json.load(f)
    if not out:
        raise FileNotFoundError(
            f"no planner JSONs found in {input_dir}; run `dvc repro` first"
        )
    return out


def _curve_with_band(
    ax, summary: dict, metric: str, color: str, label: str,
    linestyle: str = "-", group: str = "curves",
):
    curve = summary[group][metric]
    mean = np.array(curve["mean"])
    p05 = np.array(curve["p05"])
    p95 = np.array(curve["p95"])
    eps = np.arange(len(mean))
    ax.plot(eps, mean, color=color, label=label, lw=2, linestyle=linestyle)
    # Bounds: skip the band (Oracle/Random are reference lines, not
    # methods we want to visually crowd against the comparator slate).
    if linestyle == "-":
        ax.fill_between(eps, p05, p95, color=color, alpha=0.15)


def _outcome_counts(summary: dict) -> dict[str, tuple[float, float]]:
    rewards = np.asarray(summary["per_episode"]["terminal_reward"])
    successes = (rewards > 0).sum(axis=1)
    deaths = (rewards < 0).sum(axis=1)
    truncs = (rewards == 0).sum(axis=1)
    return {
        "successes": (float(successes.mean()), float(successes.std())),
        "deaths": (float(deaths.mean()), float(deaths.std())),
        "truncations": (float(truncs.mean()), float(truncs.std())),
    }


def _cumulative_regret(summary: dict, oracle_summary: dict) -> np.ndarray:
    planner = np.asarray(summary["per_episode"]["terminal_reward"], dtype=float)
    oracle = np.asarray(
        oracle_summary["per_episode"]["terminal_reward"], dtype=float,
    )
    oracle_mean = oracle.mean(axis=0)
    instantaneous = oracle_mean[None, :] - planner
    return np.cumsum(instantaneous, axis=1)


def _planning_time_per_step(summary: dict) -> tuple[float, float]:
    pt = np.asarray(summary["per_episode"]["planning_time_s"], dtype=float)
    n_steps = np.asarray(summary["per_episode"]["n_steps"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_step = np.where(n_steps > 0, pt / n_steps, np.nan)
    finite = per_step[np.isfinite(per_step)]
    if finite.size == 0:
        return (np.nan, np.nan)
    return float(np.median(finite)), float(finite.std())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot minigrid door-code comparison.")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/minigrid_door_code"),
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="output figure path (PDF, PNG, or SVG)",
    )
    args = parser.parse_args()

    runs = _load(args.input_dir)
    available = [n for n in PLANNER_ORDER if n in runs]
    cfg = runs[available[0]]["config"]
    n_episodes = cfg["n_episodes"]
    n_seeds = cfg["n_seeds"]
    K = cfg["env"]["K"]
    L = cfg["env"].get("lock_length", 1)
    probe_noise = cfg["env"]["probe_noise"]
    grid_size = cfg["env"]["grid_size"]
    slip_prob = cfg["env"]["slip_prob"]
    learn_on_failure = cfg["learn_on_failure"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    title = (
        f"MiniGrid door-code: K={K}, L={L}, grid={grid_size}x{grid_size}, "
        f"slip={slip_prob}, probe_noise={probe_noise}, "
        f"max_steps={cfg['max_steps']}, learn_on_failure={learn_on_failure}, "
        f"n_seeds={n_seeds}, n_episodes={n_episodes}"
    )
    fig.suptitle(title, fontsize=10)

    ax_succ, ax_mass, ax_probe, ax_outcome = axes[0]
    ax_regret, ax_ptime, ax_brier, ax_identify = axes[1]

    oracle_run = runs.get("oracle")
    eps_axis = np.arange(n_episodes)

    # Oracle is excluded from the identification panels (mass, Brier,
    # episodes-to-identify): it plans on a one-hot at theta-star, so the
    # logged filter mass only measures what a Bayesian filter would learn
    # from watching purely exploitative behaviour — a reward bound, not
    # an identification baseline. Random stays (passive-identification
    # floor).
    ident_names = [n for n in available if n != "oracle"]

    for name in available:
        s = runs[name]
        c = PLANNER_COLORS[name]
        lab = PLANNER_LABELS[name]
        ls = PLANNER_LINESTYLES.get(name, "-")
        _curve_with_band(ax_succ, s, "success", c, lab, linestyle=ls)
        if name in ident_names:
            _curve_with_band(ax_mass, s, "posterior_mass_on_true_theta", c, lab, linestyle=ls)
        _curve_with_band(ax_probe, s, "probe_visits", c, lab, linestyle=ls)

        if oracle_run is not None:
            regret = _cumulative_regret(s, oracle_run)
            regret_mean = regret.mean(axis=0)
            regret_p05 = np.percentile(regret, 5, axis=0)
            regret_p95 = np.percentile(regret, 95, axis=0)
            ax_regret.plot(eps_axis, regret_mean, color=c, label=lab, lw=2, linestyle=ls)
            if ls == "-":
                ax_regret.fill_between(eps_axis, regret_p05, regret_p95, color=c, alpha=0.15)

        if "derived" in s and name in ident_names:
            _curve_with_band(ax_brier, s, "brier", c, lab, linestyle=ls, group="derived")

    ax_succ.set_xlabel("episode within block")
    ax_succ.set_ylabel("success rate")
    ax_succ.set_title("Success rate (mean ± 90% CI)")
    ax_succ.set_ylim(-0.02, 1.02)
    ax_succ.legend(loc="lower right", fontsize=9)
    ax_succ.grid(True, alpha=0.3)

    ax_mass.set_xlabel("episode within block")
    ax_mass.set_ylabel("q(theta_true)")
    ax_mass.set_title("Posterior mass on true theta")
    ax_mass.set_ylim(-0.02, 1.02)
    ax_mass.grid(True, alpha=0.3)

    ax_probe.set_xlabel("episode within block")
    ax_probe.set_ylabel("probe visits / episode")
    ax_probe.set_title("Probe visits per episode")
    ax_probe.grid(True, alpha=0.3)

    deaths_means, deaths_stds = [], []
    truncs_means, truncs_stds = [], []
    labs = []
    for name in available:
        counts = _outcome_counts(runs[name])
        d_m, d_s = counts["deaths"]
        t_m, t_s = counts["truncations"]
        deaths_means.append(d_m)
        deaths_stds.append(d_s)
        truncs_means.append(t_m)
        truncs_stds.append(t_s)
        labs.append(PLANNER_LABELS[name])
    xs = np.arange(len(available))
    width = 0.4
    ax_outcome.bar(xs - width / 2, deaths_means, width, yerr=deaths_stds,
                   color="#d62728", alpha=0.9, capsize=3,
                   label="catastrophes (FAILURE)")
    ax_outcome.bar(xs + width / 2, truncs_means, width, yerr=truncs_stds,
                   color="#7f7f7f", alpha=0.6, capsize=3,
                   label="truncations (no commit)")
    ax_outcome.set_xticks(xs)
    ax_outcome.set_xticklabels(labs, rotation=30, ha="right")
    ax_outcome.set_ylabel(f"episodes / block of {n_episodes}")
    ax_outcome.set_title("Outcome breakdown per block")
    ax_outcome.legend(loc="upper left", fontsize=9)
    ax_outcome.grid(True, alpha=0.3, axis="y")

    ax_regret.set_xlabel("episode within block")
    ax_regret.set_ylabel("cumulative regret vs oracle")
    ax_regret.set_title("Cumulative regret (mean ± 90% CI)")
    ax_regret.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax_regret.grid(True, alpha=0.3)

    ptime_med = []
    for name in available:
        m, _sd = _planning_time_per_step(runs[name])
        ptime_med.append(m)
    bar_colors = [PLANNER_COLORS[n] for n in available]
    ax_ptime.bar(xs, ptime_med, color=bar_colors, alpha=0.85)
    ax_ptime.set_xticks(xs)
    ax_ptime.set_xticklabels(labs, rotation=30, ha="right")
    ax_ptime.set_ylabel("seconds / planning step (median)")
    ax_ptime.set_title("Planning wall-clock cost")
    ax_ptime.set_yscale("log")
    ax_ptime.grid(True, alpha=0.3, axis="y", which="both")

    ax_brier.set_xlabel("episode within block")
    ax_brier.set_ylabel("Brier score (1 − q(θ*))²")
    ax_brier.set_title("Posterior calibration (Brier)")
    ax_brier.grid(True, alpha=0.3)

    identify_med = []
    identify_lo = []
    identify_hi = []
    frac_ids = []
    for name in ident_names:
        d = runs[name].get("derived", {}).get("episodes_to_identify", {})
        med = d.get("median")
        p25 = d.get("p25")
        p75 = d.get("p75")
        frac = d.get("fraction_identified", 0.0)
        identify_med.append(med if med is not None else np.nan)
        identify_lo.append(med - p25 if med is not None and p25 is not None else 0.0)
        identify_hi.append(p75 - med if med is not None and p75 is not None else 0.0)
        frac_ids.append(frac)
    identify_med_arr = np.asarray(identify_med, dtype=float)
    yerr = np.vstack([identify_lo, identify_hi])
    ident_xs = np.arange(len(ident_names))
    ax_identify.bar(
        ident_xs, np.where(np.isfinite(identify_med_arr), identify_med_arr, 0.0),
        color=[PLANNER_COLORS[n] for n in ident_names], alpha=0.85,
        yerr=yerr, capsize=3,
    )
    _y_top = ax_identify.get_ylim()[1]
    _y_offset = 0.02 * _y_top if _y_top > 0 else 0.2
    for x, frac, med, hi in zip(ident_xs, frac_ids, identify_med_arr, identify_hi):
        label = f"{int(round(100 * frac))}%" if np.isfinite(med) else "0%"
        bar_top = (med if np.isfinite(med) else 0.0) + (hi or 0.0)
        ax_identify.text(
            x, bar_top + _y_offset,
            label, ha="center", fontsize=8, color="black",
        )
    ax_identify.set_xticks(ident_xs)
    ax_identify.set_xticklabels(
        [PLANNER_LABELS[n] for n in ident_names], rotation=30, ha="right",
    )
    ax_identify.set_ylabel("episodes until q(θ*) ≥ 0.9")
    ax_identify.set_title("Sample efficiency (median + IQR)")
    ax_identify.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
