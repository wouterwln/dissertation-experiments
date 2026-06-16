"""Run the RockSample BA-POMDP experiment (M4).

m × n protocol from `learningpomdp.experiment.rock_sample_runner`:
m outer-loop seeds (each pinning a freshly sampled block-level θ) × n
episodes per block (with q(θ_sensor) carried, q(θ_quality) re-uniformed).
Per-episode metric arrays plus mean / 5th / 95th percentile curves are
written as JSON.

Wired planners: AIF, VBP, PSRL, BAMCP, BOSS, Factored BA-POMCP, Oracle,
Random. BeliefQ is deferred (its belief feature is door-code-specific
and needs a separate env-agnostic redesign).

Example:

    uv run python scripts/run_rock_sample.py \\
        --planner aif --K 2 --grid-size 3 \\
        --n-seeds 8 --n-episodes 10 --max-steps 12 \\
        --output output/rock_sample/aif.json
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path

from learningpomdp.experiment.io import is_valid_shard_json
from learningpomdp.experiment.rock_sample_runner import run_outer_loop
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.boss_planner import BOSSPlanner
from learningpomdp.planning.ids_planner import IDSPlanner
from learningpomdp.planning.oracle_planner import OraclePlanner
from learningpomdp.planning.psrl_planner import PSRLPlanner
from learningpomdp.planning.random_planner import RandomPlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


# Module-level factory helpers — must be top-level (not lambdas) so the
# multiprocessing 'spawn' context can pickle them.

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
    """Return a picklable factory callable that constructs a fresh
    planner. ``functools.partial`` over top-level constructors so the
    multiprocessing 'spawn' context can pickle them."""
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
        description="Run the RockSample BA-POMDP experiment (M4).",
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
    parser.add_argument("--K", type=int, default=2,
                        help="number of rocks")
    parser.add_argument("--grid-size", type=int, default=3,
                        help="side length of the square grid")
    parser.add_argument("--d0", type=float, default=None,
                        help="distance half-life for sensor accuracy "
                             "(default grid_size / 2)")
    parser.add_argument("--sensor-noise-eta", type=float, default=0.3,
                        help="multiplicative attenuation for noisy "
                             "sensors (theta_sensor=1)")
    parser.add_argument("--sample-reward", type=float, default=1.0,
                        help="±reward magnitude for a successful SAMPLE")
    parser.add_argument("--good-logit", type=float, default=2.0,
                        help="goal-prior logit per good rock sampled "
                             "(AIF/VBP preference; ignored by sample-based planners)")
    parser.add_argument("--bad-logit", type=float, default=4.0,
                        help="goal-prior penalty per bad rock sampled. "
                             "Asymmetric (>good_logit) so E[log C | sample "
                             "under uniform belief] < 0, preventing AIF "
                             "from picking SAMPLE purely for info gain.")
    parser.add_argument("--layout-seed", type=int, default=0,
                        help="seed for rock-cell placement")

    # ---- Runner ----
    parser.add_argument("--max-steps", type=int, default=12,
                        help="max steps per episode")
    parser.add_argument("--horizon", type=int, default=3,
                        help="model-based planning horizon")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="episodes per outer-loop seed")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="number of outer-loop seeds (m)")
    parser.add_argument("--seed", type=int, default=0,
                        help="base RNG seed")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions from the planner distribution "
                             "instead of argmax")
    parser.add_argument(
        "--clip-horizon", type=str, choices=["true", "false"], default="false",
        help="cap per-step planning horizon at remaining episode steps",
    )

    # ---- Planner hyperparams (AIF / VBP / Oracle / PSRL inner) ----
    parser.add_argument("--planner-iterations", type=int, default=150)
    parser.add_argument("--planner-damping", type=float, default=0.9)
    # ---- BAMCP / BOSS / Factored BA-POMCP hyperparams ----
    parser.add_argument("--bamcp-n-simulations", type=int, default=256)
    parser.add_argument("--bamcp-depth", type=int, default=6)
    parser.add_argument("--bamcp-ucb-c", type=float, default=1.0)
    parser.add_argument("--bamcp-gamma", type=float, default=0.99)
    parser.add_argument("--boss-n-samples", type=int, default=4)
    # IDS — Information-Directed Sampling
    parser.add_argument(
        "--ids-bamcp-n-simulations", type=int, default=1024,
        help="BAMCP simulations used by IDS as its Q oracle (RockSample "
             "default mirrors --bamcp-n-simulations)",
    )
    parser.add_argument(
        "--ids-epsilon-info", type=float, default=1.0e-6,
        help="ε regulariser in the IDS information ratio Δ²/(I+ε)",
    )

    # ---- Multiprocessing (CPU-only sample-based planners) ----
    parser.add_argument(
        "--workers", type=int, default=1,
        help="parallel seed workers; >1 only effective for CPU-only "
             "planners (BAMCP/BOSS/Factored-BAMCP/Random). Each spawn "
             "worker re-imports JAX (~500 MB), so cap at memory budget.",
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

    # ---- Output ----
    parser.add_argument("--output", type=Path, required=True,
                        help="path to write JSON summary")
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
        runner="rock_sample",
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
        grid_size=args.grid_size,
        sensor_noise_eta=args.sensor_noise_eta,
        sample_reward=args.sample_reward,
        good_logit=args.good_logit,
        bad_logit=args.bad_logit,
        layout_seed=args.layout_seed,
        max_steps=args.max_steps,
    )
    if args.d0 is not None:
        env_kwargs["d0"] = args.d0

    planner_factory = _build_planner_factory(args)
    clip_horizon = args.clip_horizon == "true"

    print(
        f"planner={args.planner}  K={args.K}  grid_size={args.grid_size}  "
        f"sensor_eta={args.sensor_noise_eta}  max_steps={args.max_steps}  "
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
    sense_curve = summary["curves"]["n_senses"]["mean"]
    sample_curve = summary["curves"]["n_samples_successful"]["mean"]
    good_curve = summary["curves"]["n_good_sampled"]["mean"]
    bad_curve = summary["curves"]["n_bad_sampled"]["mean"]
    persistent_mass_curve = summary["curves"]["posterior_mass_on_persistent_theta"]["mean"]
    persistent_curve = summary["curves"]["posterior_entropy_persistent"]["mean"]
    episodic_curve = summary["curves"]["posterior_entropy_episodic"]["mean"]

    print(f"\nSummary written to {args.output}", flush=True)
    print(f"  cumulative reward (ep 0 -> ep {args.n_episodes - 1}): "
          f"{cum_curve[0]:.2f} -> {cum_curve[-1]:.2f}", flush=True)
    print(f"  reward / max_possible (∈[0,1]): "
          f"{norm_curve[0]:.2f} -> {norm_curve[-1]:.2f}", flush=True)
    print(f"  senses per ep:                {sense_curve[0]:.1f} -> "
          f"{sense_curve[-1]:.1f}", flush=True)
    print(f"  rocks consumed per ep:        {sample_curve[0]:.2f} -> "
          f"{sample_curve[-1]:.2f}", flush=True)
    print(f"  good rocks sampled per ep:    {good_curve[0]:.2f} -> "
          f"{good_curve[-1]:.2f}", flush=True)
    print(f"  bad  rocks sampled per ep:    {bad_curve[0]:.2f} -> "
          f"{bad_curve[-1]:.2f}", flush=True)
    print(f"  mass on TRUE persistent θ:    {persistent_mass_curve[0]:.3f} -> "
          f"{persistent_mass_curve[-1]:.3f}", flush=True)
    print(f"  persistent posterior entropy: {persistent_curve[0]:.3f} -> "
          f"{persistent_curve[-1]:.3f}", flush=True)
    print(f"  episodic   posterior entropy: {episodic_curve[0]:.3f} -> "
          f"{episodic_curve[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
