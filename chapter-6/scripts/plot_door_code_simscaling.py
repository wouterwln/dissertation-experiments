"""Plot the door-code BAMCP sims-budget ablation.

Reads output/door_code_simscaling/sims<N>/bamcp.json for the simulation
budgets in params.door_code_simscaling.sims_sweep and overlays one curve
per budget:

  Left:   terminal reward per episode (mean ± 95% CI of the mean)
  Middle: success rate per episode
  Right:  probe visits per episode

All three bands are 1.96·SEM across seeds, the same convention as the
main results figures.

Evidence that BAMCP's 0.00 flatline on door-code is budget-invariant
around the canonical 10000 simulations (Guez 2012 paper-grade norm),
i.e. a property of uniform-rollout value estimation on the sparse
fail-deadly lock corridor, not a simulation-budget artifact.

Usage:
    uv run python scripts/plot_door_code_simscaling.py \\
        --input-dir output/door_code_simscaling \\
        --output output/door_code_simscaling/figure.pdf
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


def _discover_sims_values(input_dir: Path) -> list[int]:
    """List sims<N> subdirectories under input_dir, sorted ascending."""
    sims: list[int] = []
    for child in input_dir.iterdir():
        if child.is_dir() and child.name.startswith("sims"):
            try:
                sims.append(int(child.name[4:]))
            except ValueError:
                continue
    return sorted(sims)


def _mean_ci_curve(
    summary: dict, metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± 95% CI of the mean, per episode across seeds.

    1.96·SEM, matching every main results figure
    (`plot_door_code.py:110-116`). The `curves` block in the JSON stores
    p05/p95 instead, which is the spread of per-seed values rather than a
    confidence interval of the mean, so it is not used here.
    """
    raw = np.asarray(summary["per_episode"][metric], dtype=float)
    mean = raw.mean(axis=0)
    band = 1.96 * raw.std(axis=0, ddof=1) / np.sqrt(raw.shape[0])
    return mean, mean - band, mean + band


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot door-code BAMCP sims-budget ablation.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/door_code_simscaling"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sims_values = _discover_sims_values(args.input_dir)
    if not sims_values:
        raise FileNotFoundError(
            f"no sims<N> subdirectories under {args.input_dir}; "
            f"run `dvc repro door_code_simscaling` first"
        )

    runs = {}
    for sims in sims_values:
        path = args.input_dir / f"sims{sims}" / "bamcp.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run `dvc repro door_code_simscaling` first"
            )
        with path.open() as f:
            runs[sims] = json.load(f)

    theme.apply_rcparams()
    cfg = runs[sims_values[0]]["config"]
    n_episodes = cfg["n_episodes"]
    eps = np.arange(n_episodes)
    # Simulation budget is ordered, so it wears the theme's ordinal teal ramp.
    colors = dict(zip(sims_values, theme.ordinal_colors(len(sims_values))))

    fig, (ax_reward, ax_succ, ax_probe) = plt.subplots(1, 3, figsize=(15, 5))
    title = (
        f"Door-code BAMCP sims-budget ablation — K={cfg['env']['K']}, "
        f"L={cfg['env'].get('lock_length', 1)}, "
        f"probe_noise={cfg['env']['probe_noise']}, "
        f"n_seeds={cfg['n_seeds']}, n_episodes={n_episodes}"
    )
    fig.suptitle(title, fontsize=11)

    for sims in sims_values:
        s = runs[sims]
        c = colors[sims]
        lab = f"{sims} sims"
        for ax, metric in ((ax_reward, "terminal_reward"),
                           (ax_succ, "success"),
                           (ax_probe, "probe_visits")):
            mean, lo, hi = _mean_ci_curve(s, metric)
            ax.plot(eps, mean, color=c, label=lab, lw=2)
            ax.fill_between(eps, lo, hi, color=c, alpha=0.15)

    ax_reward.set_xlabel("episode within block")
    ax_reward.set_ylabel("terminal reward")
    ax_reward.set_title("Terminal reward (mean ± 95% CI)")
    ax_reward.axhline(0, color=theme.PALETTE["ink"], lw=0.5, alpha=0.4)
    ax_reward.legend(loc="best", fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    ax_succ.set_xlabel("episode within block")
    ax_succ.set_ylabel("success rate")
    ax_succ.set_title("Success rate")
    ax_succ.set_ylim(-0.02, 1.02)
    ax_succ.grid(True, alpha=0.3)

    ax_probe.set_xlabel("episode within block")
    ax_probe.set_ylabel("probe visits / episode")
    ax_probe.set_title("Probe visits per episode")
    ax_probe.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
