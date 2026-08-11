"""Run the RockSample BA-POMDP experiment (M4).

m × n protocol from `learningpomdp.experiment.rock_sample_runner`:
m outer-loop seeds (each pinning a freshly sampled block-level θ) × n
episodes per block (with q(θ_sensor) carried, q(θ_quality) re-uniformed).
Per-episode metric arrays plus mean / 5th / 95th percentile curves are
written as JSON.

Wired planners: AIF, VBP, PSRL, BAMCP, BOSS, Factored BA-POMCP, Oracle,
Persistent-Oracle (AIF inner solve, plus a `_vbp` diagnostic), Random.
BeliefQ is deferred (its belief feature is door-code-specific and needs a
separate env-agnostic redesign).

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

from learningpomdp.experiment.io import invocation_record, is_valid_shard_json
from learningpomdp.experiment.rock_sample_runner import run_outer_loop
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.boss_planner import BOSSPlanner
from learningpomdp.planning.ids_planner import IDSPlanner
from learningpomdp.planning.oracle_planner import OraclePlanner
from learningpomdp.planning.persistent_oracle_planner import PersistentOraclePlanner
from learningpomdp.planning.psrl_planner import PSRLPlanner
from learningpomdp.planning.random_planner import RandomPlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


# Module-level factory helpers — must be top-level (not lambdas) so the
# multiprocessing 'spawn' context can pickle them.
#
# Every helper takes ``seed_index`` last: the runner calls the factory once
# per outer-loop seed and passes that seed's index, and the MCTS families
# fold it into their ``rollout_seed``. Without it all m seeds construct
# ``default_rng(<same int>)`` and — since the draws per plan() call are
# deterministic given n_simulations — draw a bit-identical Monte-Carlo
# stream, which makes the tree-search noise *common* across seeds and the
# cross-seed ±1.96·SEM understate planner variance. ``seed_index`` is at
# most the shard's seed count (tens), so it stays far below
# SHARD_SEED_STRIDE and per-seed offsets can never reach into the next
# shard's range. Index 0 is unshifted, so seed 0 of every shard keeps its
# previous stream.
#
# Planners with no planner-side RNG (AIF / VBP / Oracle / PSRL / Random
# draw from the per-seed generator the runner hands to prepare_episode /
# plan) accept and ignore the argument, so the factory protocol stays
# uniform across the slate.

def _make_aif(iterations, damping, seed_index):
    return AIFPlanner(n_iterations=iterations, damping=damping)


def _make_vbp(iterations, damping, seed_index):
    return VBPPlanner(n_iterations=iterations, damping=damping)


def _make_oracle(iterations, damping, seed_index):
    return OraclePlanner(VBPPlanner(n_iterations=iterations, damping=damping))


def _make_persistent_oracle(iterations, damping, seed_index):
    return PersistentOraclePlanner(
        AIFPlanner(n_iterations=iterations, damping=damping)
    )


def _make_persistent_oracle_vbp(iterations, damping, seed_index):
    return PersistentOraclePlanner(
        VBPPlanner(n_iterations=iterations, damping=damping)
    )


def _make_random(seed_index):
    return RandomPlanner()


def _make_psrl(iterations, damping, seed_index):
    return PSRLPlanner(VBPPlanner(n_iterations=iterations, damping=damping))


def _make_bamcp(
    n_sims, depth, ucb_c, gamma, seed, factored,
    rollout_policy, rollout_epsilon, rollout_q_lr, seed_index,
):
    return BAMCPPlanner(
        n_simulations=n_sims, depth=depth, ucb_c=ucb_c, gamma=gamma,
        rollout_seed=seed + seed_index, factored_root_sample=factored,
        rollout_policy=rollout_policy, rollout_epsilon=rollout_epsilon,
        rollout_q_lr=rollout_q_lr,
    )


def _make_boss(
    n_samples, n_sims, depth, ucb_c, gamma, seed,
    rollout_policy, rollout_epsilon, rollout_q_lr, seed_index,
):
    return BOSSPlanner(
        n_samples=n_samples, n_simulations=n_sims, depth=depth,
        ucb_c=ucb_c, gamma=gamma, rollout_seed=seed + seed_index,
        rollout_policy=rollout_policy, rollout_epsilon=rollout_epsilon,
        rollout_q_lr=rollout_q_lr,
    )


def _make_ids(
    ids_n_sims, depth, ucb_c, gamma, seed, epsilon_info,
    rollout_policy, rollout_epsilon, rollout_q_lr, seed_index,
):
    return IDSPlanner(
        bamcp=BAMCPPlanner(
            n_simulations=ids_n_sims, depth=depth, ucb_c=ucb_c, gamma=gamma,
            rollout_seed=seed + seed_index, resample_theta_per_sim=True,
            rollout_policy=rollout_policy, rollout_epsilon=rollout_epsilon,
            rollout_q_lr=rollout_q_lr,
        ),
        epsilon_info=epsilon_info,
    )


# Per-shard RNG stride. Shard i's seeds start at `--seed + i * STRIDE`,
# and every planner-side RNG base is offset by the same amount so that no
# two shards share a Monte-Carlo stream.
SHARD_SEED_STRIDE = 1_000_003


def _planner_seeds(args) -> tuple[int, int]:
    """Per-shard (planner rollout seed, IDS seed).

    Planner RNG bases must vary per shard, or every seed a sharded run
    averages over draws the same MCTS noise and the cross-seed CI
    understates variance. Both bases take the same shard offset so they
    stay a constant apart at every shard. See run_door_code.py.
    """
    offset = args.shard_index * SHARD_SEED_STRIDE
    return args.seed + offset, args.ids_rollout_seed + offset


def _build_planner_factory(args):
    """Return a picklable ``factory(seed_index) -> Planner``.

    ``functools.partial`` over top-level constructors so the
    multiprocessing 'spawn' context can pickle them; ``seed_index`` is
    supplied by the runner per outer-loop seed (a counter inside the
    factory would not work — the parallel path re-pickles the factory for
    every task, so each worker copy would restart at zero)."""
    rollout_seed, ids_seed = _planner_seeds(args)
    rollout = (args.rollout_policy, args.rollout_epsilon, args.rollout_q_lr)
    if args.planner == "aif":
        return partial(_make_aif, args.planner_iterations, args.planner_damping)
    if args.planner == "vbp":
        return partial(_make_vbp, args.planner_iterations, args.planner_damping)
    if args.planner == "oracle":
        return partial(_make_oracle, args.planner_iterations, args.planner_damping)
    if args.planner == "persistent_oracle":
        return partial(
            _make_persistent_oracle,
            args.planner_iterations, args.planner_damping,
        )
    if args.planner == "persistent_oracle_vbp":
        return partial(
            _make_persistent_oracle_vbp,
            args.planner_iterations, args.planner_damping,
        )
    if args.planner == "random":
        return _make_random
    if args.planner == "psrl":
        return partial(_make_psrl, args.planner_iterations, args.planner_damping)
    if args.planner == "bamcp":
        return partial(
            _make_bamcp,
            args.bamcp_n_simulations, args.bamcp_depth, args.bamcp_ucb_c,
            args.bamcp_gamma, rollout_seed, False, *rollout,
        )
    if args.planner == "factored_bamcp":
        return partial(
            _make_bamcp,
            args.bamcp_n_simulations, args.bamcp_depth, args.bamcp_ucb_c,
            args.bamcp_gamma, rollout_seed, True, *rollout,
        )
    if args.planner == "boss":
        return partial(
            _make_boss,
            args.boss_n_samples, args.bamcp_n_simulations, args.bamcp_depth,
            args.bamcp_ucb_c, args.bamcp_gamma, rollout_seed, *rollout,
        )
    if args.planner == "ids":
        # IDS's own depth and rollout seed (not the BAMCP arm's) so its
        # Monte-Carlo noise is independent.
        return partial(
            _make_ids,
            args.ids_bamcp_n_simulations,
            args.bamcp_depth if args.ids_bamcp_depth is None
            else args.ids_bamcp_depth,
            args.bamcp_ucb_c, args.bamcp_gamma, ids_seed,
            args.ids_epsilon_info, *rollout,
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
            "ids", "oracle", "persistent_oracle", "persistent_oracle_vbp",
            "random",
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
    # Leaf-rollout policy, shared by BAMCP / Factored-BAMCP / BOSS / IDS.
    parser.add_argument(
        "--rollout-policy", choices=["uniform", "eps_greedy_q"],
        default="uniform",
        help="MCTS leaf-rollout policy. 'uniform' (default) is the "
             "uniform-random rollout; 'eps_greedy_q' is Guez et al. 2012's "
             "epsilon-greedy rollout in a Q-hat learned off-policy from the "
             "rollouts themselves.",
    )
    parser.add_argument(
        "--rollout-epsilon", type=float, default=0.1,
        help="exploration rate of the eps_greedy_q rollout (ignored under "
             "--rollout-policy uniform)",
    )
    parser.add_argument(
        "--rollout-q-lr", type=float, default=0.1,
        help="learning rate of the eps_greedy_q rollout's off-policy Q-hat "
             "update (ignored under --rollout-policy uniform)",
    )
    # IDS — Information-Directed Sampling
    parser.add_argument(
        "--ids-bamcp-n-simulations", type=int, default=1024,
        help="BAMCP simulations used by IDS as its Q oracle (RockSample "
             "default mirrors --bamcp-n-simulations)",
    )
    parser.add_argument(
        "--ids-bamcp-depth", type=int, default=None,
        help="tree+rollout depth of IDS's own BAMCP instance; defaults to "
             "--bamcp-depth",
    )
    parser.add_argument(
        "--ids-rollout-seed", type=int, default=7919,
        help="RNG base for IDS's own BAMCP instance; deliberately != --seed "
             "so IDS's Monte-Carlo noise is independent of the BAMCP arm's. "
             "Must not differ from --seed by a multiple of the shard stride "
             "(1_000_003); 7919 is coprime to it.",
    )
    parser.add_argument(
        "--ids-epsilon-info", type=float, default=1.0e-6,
        help="ε regulariser in the IDS information ratio Δ(π)²/(I(π)+ε)",
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
    shard_seed = args.seed + args.shard_index * SHARD_SEED_STRIDE

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
    # `ids_bamcp_depth` is recorded post-fallback: the CLI default is None
    # and no cluster job script passes the flag, so the raw arg says
    # nothing about the depth IDS actually searched to.
    summary["config"]["invocation"] = invocation_record(
        args,
        ids_bamcp_depth=(
            args.bamcp_depth if args.ids_bamcp_depth is None
            else args.ids_bamcp_depth
        ),
    )
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
