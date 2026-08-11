"""Run the spatial MiniGrid door-code system-identification experiment.

Same m x n protocol as scripts/run_door_code.py — m outer-loop seeds,
each pinning a freshly sampled theta_true and running n episodes with
q(theta) carried across resets — but driven against the Farama-MiniGrid-
backed `MiniGridDoorCodeEnv`. Heading-aware dynamics (turn left / turn
right / forward / toggle), 3x3 partial-FOV position observations, slip
sampled directly from the env's transition tensor (T is the marginal of
the StochasticActionWrapper-style slip-then-det process), probes as
colored Floor tiles observed by standing on them, success requires
walking to a `Goal()` cell behind the partition wall after opening all
lock doors.

Same planner slate as M1: AIF, VBP, PSRL, BAMCP/BA-POMCP, literal-POMCP
ablation, BOSS, BeliefQ, Oracle, Random. The planners are
environment-agnostic (they consume T / B / goal / q_state / q_static
plus optional prepare_episode hooks), so they port 1:1 to the spatial
sibling.

Standard config (K, M, lock_length, probe_noise, grid_size, slip_prob,
agent_view_size, max_steps, horizon, n_seeds, n_episodes, seed,
learn_on_failure, clip_horizon) is on the CLI and tracked by
params.yaml / dvc.yaml.

Example:

    uv run python scripts/run_minigrid_door_code.py \\
        --planner aif --K 3 --lock-length 1 --probe-noise 0.1 \\
        --grid-size 6 --slip-prob 0.1 --agent-view-size 3 \\
        --max-steps 25 --horizon 14 --n-episodes 10 --n-seeds 4 \\
        --learn-on-failure false --clip-horizon true \\
        --output output/minigrid_door_code/aif.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from learningpomdp.environments.minigrid_door_code import (
    MiniGridDoorCodeEnv,
    N_ACTIONS,
)
from learningpomdp.experiment.door_code_runner import run_outer_loop
from learningpomdp.experiment.io import invocation_record, is_valid_shard_json
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.belief_state_q import BeliefStateQLearner
from learningpomdp.planning.boss_planner import BOSSPlanner
from learningpomdp.planning.ids_planner import IDSPlanner
from learningpomdp.planning.oracle_planner import OraclePlanner
from learningpomdp.planning.psrl_planner import PSRLPlanner
from learningpomdp.planning.random_planner import RandomPlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


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


def _build_planner(args, K: int, M: int, L: int, n_states: int):
    rollout_seed, ids_seed = _planner_seeds(args)
    if args.planner == "aif":
        return AIFPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        )
    if args.planner == "vbp":
        return VBPPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        )
    if args.planner == "psrl":
        # PSRL wraps VBPPlanner (matches Oracle's base, both are sharp-θ
        # planners).
        return PSRLPlanner(VBPPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        ))
    if args.planner == "belief_q":
        # Spatial env has its own (n_states, n_actions); pass them so the
        # Q-table sizes correctly. K and L drive the belief-feature
        # dimensionality (MAP pi(stage), entropy bin).
        return BeliefStateQLearner(
            K=K, M=M, L=L,
            n_states=n_states, n_actions=N_ACTIONS,
            lr=args.q_lr, epsilon=args.q_epsilon, gamma=args.q_gamma,
            rng_seed=rollout_seed,
        )
    if args.planner == "boss":
        # Faithful BOSS: merged-POMDP MCTS with augmented action space
        # A x K. Reuses BAMCP's MCTS hyperparams.
        return BOSSPlanner(
            n_samples=args.boss_n_samples,
            n_simulations=args.bamcp_n_simulations,
            depth=args.bamcp_depth,
            ucb_c=args.bamcp_ucb_c,
            gamma=args.bamcp_gamma,
            rollout_seed=rollout_seed,
            rollout_policy=args.rollout_policy,
            rollout_epsilon=args.rollout_epsilon,
            rollout_q_lr=args.rollout_q_lr,
        )
    if args.planner == "bamcp":
        return BAMCPPlanner(
            n_simulations=args.bamcp_n_simulations,
            depth=args.bamcp_depth,
            ucb_c=args.bamcp_ucb_c,
            gamma=args.bamcp_gamma,
            rollout_seed=rollout_seed,
            resample_theta_per_sim=True,
            rollout_policy=args.rollout_policy,
            rollout_epsilon=args.rollout_epsilon,
            rollout_q_lr=args.rollout_q_lr,
        )
    if args.planner == "pomcp":
        # Literal-POMCP ablation: same MCTS as BAMCP but no theta-aware
        # root resample (theta sampled once per plan() call).
        return BAMCPPlanner(
            n_simulations=args.bamcp_n_simulations,
            depth=args.bamcp_depth,
            ucb_c=args.bamcp_ucb_c,
            gamma=args.bamcp_gamma,
            rollout_seed=rollout_seed,
            resample_theta_per_sim=False,
            rollout_policy=args.rollout_policy,
            rollout_epsilon=args.rollout_epsilon,
            rollout_q_lr=args.rollout_q_lr,
        )
    if args.planner == "ids":
        # IDS's BAMCP gets IDS's OWN depth and rollout seed, which is the
        # only thing making its Monte-Carlo noise independent of the
        # BAMCP arm's (see IDSPlanner's module docstring).
        return IDSPlanner(
            bamcp=BAMCPPlanner(
                n_simulations=args.ids_bamcp_n_simulations,
                depth=(
                    args.bamcp_depth if args.ids_bamcp_depth is None
                    else args.ids_bamcp_depth
                ),
                ucb_c=args.bamcp_ucb_c,
                gamma=args.bamcp_gamma,
                rollout_seed=ids_seed,
                resample_theta_per_sim=True,
                rollout_policy=args.rollout_policy,
                rollout_epsilon=args.rollout_epsilon,
                rollout_q_lr=args.rollout_q_lr,
            ),
            epsilon_info=args.ids_epsilon_info,
        )
    if args.planner == "oracle":
        return OraclePlanner(VBPPlanner(
            n_iterations=args.planner_iterations, damping=args.planner_damping,
        ))
    if args.planner == "random":
        return RandomPlanner()
    raise ValueError(f"unknown planner: {args.planner}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the spatial minigrid door-code experiment (m x n protocol).",
    )
    # ---- Planner ----
    parser.add_argument(
        "--planner",
        choices=[
            "aif", "vbp", "psrl", "belief_q", "boss",
            "bamcp", "pomcp", "ids", "oracle", "random",
        ],
        required=True,
    )
    # AIF / VBP / PSRL / BOSS / Oracle (model-based)
    parser.add_argument("--planner-iterations", type=int, default=80)
    parser.add_argument("--planner-damping", type=float, default=0.9)
    # BeliefQ
    parser.add_argument("--q-lr", type=float, default=0.2)
    parser.add_argument("--q-epsilon", type=float, default=0.2)
    parser.add_argument("--q-gamma", type=float, default=0.99)
    # BOSS
    parser.add_argument("--boss-n-samples", type=int, default=4)
    # BAMCP / literal-POMCP (shared)
    parser.add_argument("--bamcp-n-simulations", type=int, default=256)
    parser.add_argument("--bamcp-depth", type=int, default=15,
                        help="BAMCP max tree+rollout depth")
    parser.add_argument("--bamcp-ucb-c", type=float, default=1.0)
    parser.add_argument("--bamcp-gamma", type=float, default=0.99)
    # Leaf-rollout policy, shared by BAMCP / POMCP / BOSS / IDS's BAMCP.
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
    # IDS
    parser.add_argument(
        "--ids-bamcp-n-simulations", type=int, default=256,
        help="BAMCP simulations used by IDS as its Q oracle",
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

    # ---- Environment ----
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--M", type=int, default=None,
                        help="number of probe stations; default = "
                             "len(default_partitions(K))")
    parser.add_argument("--lock-length", type=int, default=1)
    parser.add_argument("--probe-noise", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=5,
                        help="grid is grid_size x grid_size; >= max(5, K+2, M+3)")
    parser.add_argument("--slip-prob", type=float, default=0.1,
                        help="probability the chosen action is replaced by a "
                             "uniform draw over the 6 underlying MiniGrid "
                             "actions (matches StochasticActionWrapper marginal)")
    parser.add_argument("--agent-view-size", type=int, default=3,
                        help="side length of the partial-FOV; must be odd "
                             "and >= 3 (passed to MiniGridEnv.agent_view_size)")

    # ---- Runner ----
    parser.add_argument("--max-steps", type=int, default=25,
                        help="max steps per episode")
    parser.add_argument("--horizon", type=int, default=10,
                        help="model-based planning horizon")
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--learn-on-failure", type=str, choices=["true", "false"], default="false",
    )
    parser.add_argument(
        "--clip-horizon", type=str, choices=["true", "false"], default="true",
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--video-dir", type=Path, default=None,
        help="If set, write per-episode rgb_array videos to "
             "{video_dir}/{outer_loop}/{inner_episode}.mp4. Triggers "
             "render_mode='rgb_array' on the env.",
    )
    parser.add_argument("--video-fps", type=int, default=4)
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
        grid_size=args.grid_size, slip_prob=args.slip_prob,
        agent_view_size=args.agent_view_size,
    )
    if args.M is not None:
        env_kwargs["M"] = args.M
    if args.video_dir is not None:
        env_kwargs["render_mode"] = "rgb_array"

    # M and n_states are needed up-front to size BeliefQ. Construct a
    # throw-away env with layout_seed=0 to read them; the actual runner
    # builds fresh per-seed envs via env_factory.
    from learningpomdp.environments.door_code import default_partitions
    M = args.M if args.M is not None else len(default_partitions(args.K))
    probe_env = MiniGridDoorCodeEnv(**{**env_kwargs, "layout_seed": 0})
    n_states_for_planner = probe_env.n_states
    del probe_env

    planner = _build_planner(args, K=args.K, M=M, L=args.lock_length,
                             n_states=n_states_for_planner)

    learn_on_failure = args.learn_on_failure == "true"
    clip_horizon = args.clip_horizon == "true"
    print(
        f"planner={args.planner}  K={args.K}  M={M}  L={args.lock_length}  "
        f"grid_size={args.grid_size}  slip_prob={args.slip_prob}  "
        f"probe_noise={args.probe_noise}  agent_view_size={args.agent_view_size}  "
        f"max_steps={args.max_steps}  horizon={args.horizon}  "
        f"n_episodes={args.n_episodes}  "
        f"n_seeds={shard_size}/{total_seeds} "
        f"(shard {args.shard_index}/{args.n_shards}, base seed={shard_seed})  "
        f"deterministic={not args.stochastic}  "
        f"learn_on_failure={learn_on_failure}  clip_horizon={clip_horizon}",
        flush=True,
    )

    video_path_for = None
    if args.video_dir is not None:
        vdir = args.video_dir

        def video_path_for(outer_idx: int, inner_idx: int) -> Path:
            return vdir / str(outer_idx) / f"{inner_idx}.mp4"

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
        env_factory=lambda kw: MiniGridDoorCodeEnv(**kw),
        video_path_for=video_path_for,
        video_fps=args.video_fps,
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

    success_curve = summary["curves"]["success"]["mean"]
    mass_curve = summary["curves"]["posterior_mass_on_true_theta"]["mean"]
    probe_curve = summary["curves"]["probe_visits"]["mean"]
    cum_succ = sum(success_curve)
    deaths = sum(
        1 for block in result.blocks for ep in block.episodes
        if ep.terminal_reward < 0
    ) / shard_size
    print(f"\nSummary written to {args.output}", flush=True)
    print(f"  cumulative successes/block: {cum_succ:.1f} / {args.n_episodes}", flush=True)
    print(f"  deaths/block:               {deaths:.2f}", flush=True)
    print(f"  success rate (ep 0  -> ep {args.n_episodes - 1}): "
          f"{success_curve[0]:.2f} -> {success_curve[-1]:.2f}", flush=True)
    print(f"  posterior mass on true theta: "
          f"{mass_curve[0]:.2f} -> {mass_curve[-1]:.2f}", flush=True)
    print(f"  probe visits per ep:         "
          f"{probe_curve[0]:.2f} -> {probe_curve[-1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
