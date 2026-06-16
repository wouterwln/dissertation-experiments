"""Run the Factored Tiger BA-POMDP experiment.

Canonical literature benchmark (Katt et al. 2019, FBA-POMDP, AAMAS).
K independent doors with persistent factored θ; agent LISTENs to
disambiguate then OPENs safe doors. Same planner slate as M4 RockSample
minus Belief-Q (door-code-specific feature).

Example:

    uv run python scripts/run_factored_tiger.py \\
        --planner aif --K 4 \\
        --n-seeds 8 --n-episodes 10 --max-steps 20 \\
        --output output/factored_tiger/aif.json
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

from learningpomdp.experiment.factored_tiger_runner import run_outer_loop
from learningpomdp.experiment.io import is_valid_shard_json
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.boss_planner import BOSSPlanner
from learningpomdp.planning.ids_planner import IDSPlanner
from learningpomdp.planning.oracle_planner import OraclePlanner
from learningpomdp.planning.psrl_planner import PSRLPlanner
from learningpomdp.planning.random_planner import RandomPlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


# Top-level factory helpers — must be picklable for the 'spawn' MP context.


def _make_aif(iterations, damping):
    return AIFPlanner(n_iterations=iterations, damping=damping)


def _make_vbp(iterations, damping):
    return VBPPlanner(n_iterations=iterations, damping=damping)


def _make_oracle(iterations, damping):
    return OraclePlanner(VBPPlanner(n_iterations=iterations, damping=damping))


def _make_random():
    return RandomPlanner()


def _make_psrl(iterations, damping):
    return PSRLPlanner(VBPPlanner(n_iterations=iterations, damping=damping))


def _make_bamcp(n_sims, depth, ucb_c, gamma, seed, factored):
    return BAMCPPlanner(
        n_simulations=n_sims, depth=depth, ucb_c=ucb_c, gamma=gamma,
        rollout_seed=seed, factored_root_sample=factored,
    )


def _make_boss(n_samples, n_sims, depth, ucb_c, gamma, seed):
    return BOSSPlanner(
        n_samples=n_samples, n_simulations=n_sims, depth=depth,
        ucb_c=ucb_c, gamma=gamma, rollout_seed=seed,
    )


def _make_ids(ids_n_sims, depth, ucb_c, gamma, seed, epsilon_info):
    return IDSPlanner(
        bamcp=BAMCPPlanner(
            n_simulations=ids_n_sims, depth=depth, ucb_c=ucb_c, gamma=gamma,
            rollout_seed=seed, resample_theta_per_sim=True,
        ),
        epsilon_info=epsilon_info,
    )


def _build_planner_factory(args):
    if args.planner == "aif":
        return partial(_make_aif, args.planner_iterations, args.planner_damping)
    if args.planner == "vbp":
        return partial(_make_vbp, args.planner_iterations, args.planner_damping)
    if args.planner == "oracle":
        return partial(_make_oracle, args.planner_iterations, args.planner_damping)
    if args.planner == "random":
        return _make_random
    if args.planner == "psrl":
        return partial(_make_psrl, args.planner_iterations, args.planner_damping)
    if args.planner == "bamcp":
        return partial(
            _make_bamcp,
            args.bamcp_n_simulations, args.bamcp_depth, args.bamcp_ucb_c,
            args.bamcp_gamma, args.seed, False,
        )
    if args.planner == "factored_bamcp":
        return partial(
            _make_bamcp,
            args.bamcp_n_simulations, args.bamcp_depth, args.bamcp_ucb_c,
            args.bamcp_gamma, args.seed, True,
        )
    if args.planner == "boss":
        return partial(
            _make_boss,
            args.boss_n_samples, args.bamcp_n_simulations, args.bamcp_depth,
            args.bamcp_ucb_c, args.bamcp_gamma, args.seed,
        )
    if args.planner == "ids":
        return partial(
            _make_ids,
            args.ids_bamcp_n_simulations, args.bamcp_depth,
            args.bamcp_ucb_c, args.bamcp_gamma, args.seed, args.ids_epsilon_info,
        )
    raise ValueError(f"unknown planner: {args.planner}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Factored Tiger BA-POMDP experiment.",
    )
    parser.add_argument(
        "--planner",
        choices=[
            "aif", "vbp", "psrl", "bamcp", "boss", "factored_bamcp",
            "ids", "oracle", "random",
        ],
        required=True,
    )

    # ---- Environment ----
    parser.add_argument("--K", type=int, default=4,
                        help="number of doors")
    parser.add_argument("--listen-accuracy", type=float, default=0.85,
                        help="P(LISTEN_k emits correct bit | θ_k) ∈ [0.5, 1]")
    parser.add_argument("--open-safe-reward", type=float, default=10.0,
                        help="reward for OPEN_k on unopened door with θ_k=1")
    parser.add_argument("--open-tiger-penalty", type=float, default=100.0,
                        help="positive magnitude; reward is negation for "
                             "OPEN_k on unopened door with θ_k=0")
    parser.add_argument("--listen-cost", type=float, default=0.0,
                        help="positive magnitude; reward is negation per "
                             "LISTEN. Default 0 (canonical Katt 2019 uses 1).")
    parser.add_argument("--good-logit", type=float, default=2.0,
                        help="goal-prior logit per safe door opened "
                             "(AIF/VBP preference; ignored by sample-based)")
    parser.add_argument("--bad-logit", type=float, default=4.0,
                        help="goal-prior penalty per tiger door opened. "
                             "Asymmetric (>good_logit) prevents AIF from "
                             "picking OPEN purely for info gain.")

    # ---- Runner ----
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--clip-horizon", type=str, choices=["true", "false"], default="true",
    )

    # ---- Planner hyperparams ----
    parser.add_argument("--planner-iterations", type=int, default=70)
    parser.add_argument("--planner-damping", type=float, default=0.9)
    parser.add_argument("--bamcp-n-simulations", type=int, default=1024)
    parser.add_argument("--bamcp-depth", type=int, default=6)
    parser.add_argument("--bamcp-ucb-c", type=float, default=1.0)
    parser.add_argument("--bamcp-gamma", type=float, default=0.99)
    parser.add_argument("--boss-n-samples", type=int, default=4)
    parser.add_argument("--ids-bamcp-n-simulations", type=int, default=1024)
    parser.add_argument("--ids-epsilon-info", type=float, default=1.0e-6)

    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel seed workers; >1 only for CPU-only sample-based planners.",
    )
    # ---- Sharded execution (SLURM array jobs). Default n-shards=1 means
    # no sharding; with n-shards>1 each invocation runs one chunk of seeds
    # and scripts/aggregate_shards.py combines the outputs.
    parser.add_argument("--n-shards", type=int, default=1,
                        help="total number of shards to split seeds across")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="index of this shard in [0, n_shards)")
    parser.add_argument("--total-seeds", type=int, default=None,
                        help="total seed count across all shards; "
                             "defaults to --n-seeds")

    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total_seeds = args.total_seeds if args.total_seeds is not None else args.n_seeds
    seeds_per_shard = math.ceil(total_seeds / args.n_shards)
    shard_start = args.shard_index * seeds_per_shard
    shard_size = min(seeds_per_shard, total_seeds - shard_start)
    if shard_size <= 0:
        raise SystemExit(
            f"shard {args.shard_index}/{args.n_shards} is empty "
            f"(total_seeds={total_seeds}, seeds_per_shard={seeds_per_shard})"
        )
    shard_seed = args.seed + args.shard_index * 1_000_003

    if is_valid_shard_json(
        args.output,
        runner="factored_tiger",
        expected_n_seeds=shard_size,
        expected_n_episodes=args.n_episodes,
    ):
        print(
            f"skip: {args.output} exists and validates "
            f"(n_seeds={shard_size}, n_episodes={args.n_episodes})",
            flush=True,
        )
        return

    env_kwargs = dict(
        K=args.K,
        listen_accuracy=args.listen_accuracy,
        open_safe_reward=args.open_safe_reward,
        open_tiger_penalty=args.open_tiger_penalty,
        listen_cost=args.listen_cost,
        good_logit=args.good_logit,
        bad_logit=args.bad_logit,
        max_steps=args.max_steps,
    )

    planner_factory = _build_planner_factory(args)
    clip_horizon = args.clip_horizon == "true"

    print(
        f"planner={args.planner}  K={args.K}  "
        f"listen_accuracy={args.listen_accuracy}  max_steps={args.max_steps}  "
        f"horizon={args.horizon}  n_episodes={args.n_episodes}  "
        f"n_seeds={shard_size}/{total_seeds} "
        f"(shard {args.shard_index}/{args.n_shards}, base seed={shard_seed})  "
        f"deterministic={not args.stochastic}  "
        f"workers={args.workers}",
        flush=True,
    )

    result = run_outer_loop(
        env_kwargs=env_kwargs,
        planner_factory=planner_factory,
        n_seeds=shard_size,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        seed=shard_seed,
        clip_horizon=clip_horizon,
        progress_desc=args.planner,
        workers=args.workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = result.to_summary_dict()
    summary["config"]["planner"] = args.planner
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)

    cum_curve = summary["curves"]["cumulative_reward"]["mean"]
    norm_curve = summary["curves"]["reward_normalised"]["mean"]
    listen_curve = summary["curves"]["n_listens"]["mean"]
    safe_curve = summary["curves"]["n_safe_opens"]["mean"]
    tiger_curve = summary["curves"]["n_tiger_opens"]["mean"]
    persistent_mass_curve = summary["curves"]["posterior_mass_on_persistent_theta"]["mean"]
    persistent_curve = summary["curves"]["posterior_entropy_persistent"]["mean"]

    print(f"\nSummary written to {args.output}", flush=True)
    print(f"  cumulative reward (ep 0 -> ep {args.n_episodes - 1}): "
          f"{cum_curve[0]:.2f} -> {cum_curve[-1]:.2f}", flush=True)
    print(f"  reward / max_possible (∈[0,1]): "
          f"{norm_curve[0]:.2f} -> {norm_curve[-1]:.2f}", flush=True)
    print(f"  listens per ep:               {listen_curve[0]:.1f} -> "
          f"{listen_curve[-1]:.1f}", flush=True)
    print(f"  safe  opens per ep:           {safe_curve[0]:.2f} -> "
          f"{safe_curve[-1]:.2f}", flush=True)
    print(f"  tiger opens per ep:           {tiger_curve[0]:.2f} -> "
          f"{tiger_curve[-1]:.2f}", flush=True)
    print(f"  mass on TRUE persistent θ:    {persistent_mass_curve[0]:.3f} -> "
          f"{persistent_mass_curve[-1]:.3f}", flush=True)
    print(f"  persistent posterior entropy: {persistent_curve[0]:.3f} -> "
          f"{persistent_curve[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
