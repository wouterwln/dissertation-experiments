"""Plot the door-code rollout-policy ablation.

Reads output/door_code_rollout/<policy>/<planner>.json for the policies
in params.door_code_rollout.policies and overlays one curve per policy:

  Left:   terminal reward per episode (mean ± 1.96·SEM across seeds)
  Middle: success rate per episode
  Right:  probe visits per episode

This is the decisive experiment for the paper's claim that the
tree-search slate earns exactly 0.00 on the door-code lock corridor. The
shipped rollout draws actions uniformly; the published BAMCP (Guez et
al. 2012) supplies an epsilon-greedy rollout against a Q-function
learned off-policy from the agent's own experience, introduced for
exactly the case where reward requires a carefully selected action
sequence. If the flatline survives `eps_greedy_q`, the result is
genuinely striking. If it does not, the claim rests on a misconfigured
baseline and must be withdrawn.

Usage:
    uv run python scripts/plot_door_code_rollout.py \\
        --input-dir output/door_code_rollout \\
        --output output/door_code_rollout/figure.pdf
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


# Ordered worst-to-best-configured, so the ordinal ramp reads correctly.
POLICY_ORDER = ("uniform", "eps_greedy_q")
POLICY_LABELS = {
    "uniform": "uniform rollout (shipped)",
    "eps_greedy_q": "ε-greedy learned-Q rollout (Guez 2012)",
}
PLANNERS = ("bamcp", "boss")
PLANNER_LABELS = {"bamcp": "BAMCP", "boss": "BOSS"}
PLANNER_LINESTYLES = {"bamcp": "-", "boss": "--"}


def _discover_policies(input_dir: Path) -> list[str]:
    """List policy subdirectories under input_dir, in POLICY_ORDER."""
    present = {
        child.name for child in input_dir.iterdir()
        if child.is_dir() and child.name in POLICY_ORDER
    }
    return [p for p in POLICY_ORDER if p in present]


def _mean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode mean and 1.96·SEM across seeds (axis 0)."""
    mean = arr.mean(axis=0)
    n = arr.shape[0]
    sem = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, 1.96 * sem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot door-code rollout-policy ablation.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/door_code_rollout"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policies = _discover_policies(args.input_dir)
    if not policies:
        raise FileNotFoundError(
            f"no rollout-policy subdirectories under {args.input_dir}; "
            f"run `dvc repro door_code_rollout` first"
        )

    runs: dict[tuple[str, str], dict] = {}
    for policy in policies:
        for planner in PLANNERS:
            path = args.input_dir / policy / f"{planner}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path}; run `dvc repro door_code_rollout` first"
                )
            with path.open() as f:
                runs[(policy, planner)] = json.load(f)

    theme.apply_rcparams()
    cfg = runs[(policies[0], PLANNERS[0])]["config"]
    n_episodes = cfg["n_episodes"]
    eps = np.arange(n_episodes)
    # Rollout competence is an ordered axis, so it wears the ordinal ramp;
    # planners are told apart by linestyle.
    colors = dict(zip(policies, theme.ordinal_colors(len(policies))))

    fig, (ax_reward, ax_succ, ax_probe) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Door-code rollout-policy ablation — K={cfg['env']['K']}, "
        f"L={cfg['env'].get('lock_length', 1)}, "
        f"probe_noise={cfg['env']['probe_noise']}, "
        f"n_seeds={cfg['n_seeds']}, n_episodes={n_episodes}",
        fontsize=11,
    )

    for policy in policies:
        for planner in PLANNERS:
            s = runs[(policy, planner)]
            c = colors[policy]
            ls = PLANNER_LINESTYLES[planner]
            lab = f"{PLANNER_LABELS[planner]} — {POLICY_LABELS[policy]}"
            rewards = np.asarray(
                s["per_episode"]["terminal_reward"], dtype=float,
            )
            mean, err = _mean_sem(rewards)
            ax_reward.plot(eps, mean, color=c, ls=ls, lw=2, label=lab)
            ax_reward.fill_between(
                eps, mean - err, mean + err, color=c, alpha=0.15,
            )
            for ax, metric in ((ax_succ, "success"), (ax_probe, "probe_visits")):
                per_ep = np.asarray(s["per_episode"][metric], dtype=float)
                m, e = _mean_sem(per_ep)
                ax.plot(eps, m, color=c, ls=ls, lw=2, label=lab)
                ax.fill_between(eps, m - e, m + e, color=c, alpha=0.15)

    ax_reward.set_xlabel("episode within block")
    ax_reward.set_ylabel("terminal reward")
    ax_reward.set_title("Terminal reward (mean ± 1.96·SEM)")
    ax_reward.axhline(0, color=theme.PALETTE["ink"], lw=0.5, alpha=0.4)
    ax_reward.legend(loc="best", fontsize=8)
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
