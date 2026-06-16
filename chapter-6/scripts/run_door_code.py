"""Run the door-code system-identification experiment.

m x n protocol: m outer-loop seeds, each pinning a freshly sampled
theta_true and running n episodes with q(theta) carried across resets.
Per-episode metric arrays plus mean / 5th / 95th percentile curves are
written as JSON. tqdm shows progress over outer seeds.

Open env knobs (see DoorCodeEnv docstring): partitions, lock_doors_fn,
goal. Not currently wired through CLI flags — edit this script for
ablations on those axes. Standard config knobs (K, M, lock_length,
probe_noise, max_steps, horizon, n_seeds, n_episodes, seed,
learn_on_failure) ARE on the CLI and tracked by params.yaml/dvc.yaml.

Example:

    uv run python scripts/run_door_code.py \\
        --planner aif --K 4 --lock-length 2 --probe-noise 0.2 \\
        --max-steps 4 --n-episodes 30 --n-seeds 12 \\
        --no-learn-on-failure --output output/door_code/aif_K4_L2.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from learningpomdp.experiment.door_code_runner import run_outer_loop
from learningpomdp.experiment.io import is_valid_shard_json
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.belief_state_q import BeliefStateQLearner
from learningpomdp.planning.boss_planner import BOSSPlanner
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.ids_planner import IDSPlanner
from learningpomdp.planning.oracle_planner import OraclePlanner
from learningpomdp.planning.psrl_planner import PSRLPlanner
from learningpomdp.planning.random_planner import RandomPlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


def _build_planner(args, K: int, M: int, L: int):
    if args.planner == "aif":
        return AIFPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        )
    if args.planner == "vbp":
        return VBPPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        )
    if args.planner == "psrl":
        # PSRL wraps VBPPlanner — sharp-prior planning; q(theta) is
        # replaced by a one-hot sample at episode start. Pairs with
        # OraclePlanner (also VBP-based) so both sharp-θ baselines use
        # the same inner planner.
        return PSRLPlanner(VBPPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        ))
    if args.planner == "belief_q":
        return BeliefStateQLearner(
            K=K, M=M, L=L,
            lr=args.q_lr, epsilon=args.q_epsilon, gamma=args.q_gamma,
            rng_seed=args.seed,
        )
    if args.planner == "boss":
        # Faithful BOSS: merged-POMDP MCTS with augmented action space
        # A x K. Reuses BAMCP's MCTS hyperparams since the search
        # machinery is shared.
        return BOSSPlanner(
            n_samples=args.boss_n_samples,
            n_simulations=args.bamcp_n_simulations,
            depth=args.bamcp_depth,
            ucb_c=args.bamcp_ucb_c,
            gamma=args.bamcp_gamma,
            rollout_seed=args.seed,
        )
    if args.planner == "bamcp":
        return BAMCPPlanner(
            n_simulations=args.bamcp_n_simulations,
            depth=args.bamcp_depth,
            ucb_c=args.bamcp_ucb_c,
            gamma=args.bamcp_gamma,
            rollout_seed=args.seed,
            resample_theta_per_sim=True,
        )
    if args.planner == "pomcp":
        # Literal-POMCP ablation: same MCTS as BAMCP but no theta-aware
        # root resample (theta sampled once per plan() call).
        return BAMCPPlanner(
            n_simulations=args.bamcp_n_simulations,
            depth=args.bamcp_depth,
            ucb_c=args.bamcp_ucb_c,
            gamma=args.bamcp_gamma,
            rollout_seed=args.seed,
            resample_theta_per_sim=False,
        )
    if args.planner == "ids":
        # IDS — Information-Directed Sampling. Δ²/(I+ε) selection rule
        # with BAMCP as the Q oracle and closed-form I(a) over discrete
        # θ-dependent observations.
        return IDSPlanner(
            bamcp=BAMCPPlanner(
                n_simulations=args.ids_bamcp_n_simulations,
                depth=args.bamcp_depth,
                ucb_c=args.bamcp_ucb_c,
                gamma=args.bamcp_gamma,
                rollout_seed=args.seed,
                resample_theta_per_sim=True,
            ),
            epsilon_info=args.ids_epsilon_info,
        )
    if args.planner == "oracle":
        # Oracle: VBP planning against q(theta)=onehot(theta_true).
        # The runner injects theta_true via prepare_episode.
        return OraclePlanner(VBPPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        ))
    if args.planner == "random":
        return RandomPlanner()
    raise ValueError(f"unknown planner: {args.planner}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the door-code identification experiment (m x n protocol).",
    )
    # ---- Planner ----
    parser.add_argument(
        "--planner",
        choices=[
            "aif", "vbp", "psrl", "belief_q", "boss",
            "bamcp", "pomcp", "ids", "oracle", "random",
        ],
        required=True,
        help="which planner / RL agent to evaluate",
    )
    # BOSS hyperparams
    parser.add_argument(
        "--boss-n-samples", type=int, default=4,
        help="BOSS theta samples per episode (default 4); n=1 reduces to "
             "a single-sample MCTS for the episode",
    )
    # BAMCP / literal-POMCP hyperparams (shared)
    parser.add_argument(
        "--bamcp-n-simulations", type=int, default=256,
        help="BAMCP/POMCP MCTS rollouts per plan() call",
    )
    parser.add_argument(
        "--bamcp-depth", type=int, default=6,
        help="BAMCP/POMCP max tree+rollout depth",
    )
    parser.add_argument(
        "--bamcp-ucb-c", type=float, default=1.0,
        help="BAMCP/POMCP UCB1 exploration constant",
    )
    parser.add_argument(
        "--bamcp-gamma", type=float, default=0.99,
        help="BAMCP/POMCP return discount factor",
    )
    # IDS — Information-Directed Sampling hyperparameters
    parser.add_argument(
        "--ids-bamcp-n-simulations", type=int, default=256,
        help="BAMCP simulations used by IDS as its Q oracle",
    )
    parser.add_argument(
        "--ids-epsilon-info", type=float, default=1.0e-6,
        help="ε regulariser in the IDS information ratio Δ²/(I+ε)",
    )
    parser.add_argument(
        "--planner-iterations", type=int, default=150,
        help="model-based planner iteration count (AIF/VBP/PSRL)")
    parser.add_argument(
        "--planner-damping", type=float, default=0.9,
        help="model-based planner damping coefficient (AIF/VBP/PSRL)")
    parser.add_argument(
        "--q-lr", type=float, default=0.2, help="belief_q TD learning rate")
    parser.add_argument(
        "--q-epsilon", type=float, default=0.2,
        help="belief_q epsilon-greedy exploration rate")
    parser.add_argument(
        "--q-gamma", type=float, default=0.99, help="belief_q discount factor")

    # ---- Environment ----
    parser.add_argument("--K", type=int, default=4,
                        help="number of doors / symbols")
    parser.add_argument("--M", type=int, default=None,
                        help="number of probe stations; default = "
                             "len(default_partitions(K))")
    parser.add_argument("--lock-length", type=int, default=1,
                        help="number of doors that must be opened in sequence")
    parser.add_argument("--probe-noise", type=float, default=0.2,
                        help="per-probe bit-flip probability in [0, 0.5]")

    # ---- Runner ----
    parser.add_argument("--max-steps", type=int, default=2,
                        help="max steps per episode")
    parser.add_argument("--horizon", type=int, default=3,
                        help="model-based planning horizon")
    parser.add_argument("--n-episodes", type=int, default=25,
                        help="episodes per outer-loop seed")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="number of outer-loop seeds (m)")
    parser.add_argument("--seed", type=int, default=0,
                        help="base RNG seed for theta sampling and env obs")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions from the planner's distribution "
                             "instead of argmax")
    parser.add_argument(
        "--learn-on-failure", type=str, choices=["true", "false"], default="true",
        help="if false, skip the Bayesian update on the failure transition: "
             "the agent is 'dead' and cannot extract information from a "
             "transition it didn't survive to observe. Probe-derived "
             "q(theta) updates from earlier in the episode are retained.",
    )
    parser.add_argument(
        "--clip-horizon", type=str, choices=["true", "false"], default="false",
        help="if true, cap the per-step planning horizon at the number of "
             "steps remaining in the episode (max_steps - step). "
             "Receding-horizon control: the planner can't pick a first-action "
             "choice that only pays off across an unrealizable later commit.",
    )

    # ---- Sharded execution (SLURM array jobs). Default n-shards=1 means
    # no sharding — the script behaves exactly as before. With n-shards>1
    # each invocation runs one chunk of seeds; combine via
    # scripts/aggregate_shards.py.
    parser.add_argument("--n-shards", type=int, default=1,
                        help="total number of shards to split seeds across")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="index of this shard in [0, n_shards)")
    parser.add_argument("--total-seeds", type=int, default=None,
                        help="total seed count across all shards; "
                             "defaults to --n-seeds (no sharding)")

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
        runner="door_code",
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
        K=args.K, probe_noise=args.probe_noise, lock_length=args.lock_length,
    )
    if args.M is not None:
        env_kwargs["M"] = args.M

    # M needs to be known to size the BeliefQ table.
    from learningpomdp.environments.door_code import default_partitions
    M = args.M if args.M is not None else len(default_partitions(args.K))

    planner = _build_planner(args, K=args.K, M=M, L=args.lock_length)

    learn_on_failure = args.learn_on_failure == "true"
    clip_horizon = args.clip_horizon == "true"
    print(
        f"planner={args.planner}  K={args.K}  M={M}  L={args.lock_length}  "
        f"probe_noise={args.probe_noise}  max_steps={args.max_steps}  "
        f"horizon={args.horizon}  n_episodes={args.n_episodes}  "
        f"n_seeds={shard_size}/{total_seeds} "
        f"(shard {args.shard_index}/{args.n_shards}, base seed={shard_seed})  "
        f"deterministic={not args.stochastic}  "
        f"learn_on_failure={learn_on_failure}  clip_horizon={clip_horizon}",
        flush=True,
    )

    result = run_outer_loop(
        env_kwargs=env_kwargs,
        planner=planner,
        n_seeds=shard_size,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        seed=shard_seed,
        learn_on_failure=learn_on_failure,
        clip_horizon=clip_horizon,
        progress_desc=args.planner,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = result.to_summary_dict()
    summary["config"]["planner"] = args.planner
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)

    success_curve = summary["curves"]["success"]["mean"]
    mass_curve = summary["curves"]["posterior_mass_on_true_theta"]["mean"]
    probe_curve = summary["curves"]["probe_visits"]["mean"]
    cum_succ = sum(success_curve)
    deaths = sum(
        1 for block in result.blocks for ep in block.episodes
        if ep.terminal_reward < 0
    ) / shard_size
    print(f"\nSummary written to {args.output}", flush=True)
    print(f"  cumulative successes/block: {cum_succ:.1f} / {args.n_episodes}",
          flush=True)
    print(f"  deaths/block:               {deaths:.2f}", flush=True)
    print(f"  success rate (ep 0  -> ep {args.n_episodes - 1}): "
          f"{success_curve[0]:.2f} -> {success_curve[-1]:.2f}", flush=True)
    print(f"  posterior mass on true theta: "
          f"{mass_curve[0]:.2f} -> {mass_curve[-1]:.2f}", flush=True)
    print(f"  probe visits per ep:         "
          f"{probe_curve[0]:.2f} -> {probe_curve[-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
