#!/usr/bin/env python
"""Run Reactivity Maze experiments with Active Inference (message-passing) agents.

The Reactivity Maze is the chapter-3 Epistemic Maze ported to the ch5 AIF-MP
contract: the agent must resolve a hidden goal identity θ at an instructional cue
(epistemic) while keeping its reactivity knob at maximum (pragmatic). Reward is
terminal, so the intended mode is ``--terminal-goal-only --receding-horizon``.
"""

import sys
from pathlib import Path
import json
import argparse
import time
import random

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import jax
import jax.numpy as jnp
from environments.reactivity_maze import (
    sample_configs,
    generate_transition_tensor,
    generate_observation_tensor,
    generate_goal,
    ReactivityMazeEnv,
    state_index,
    unpack_state,
    N_LOCATIONS,
    MAX_KNOB,
)
from agents.reactivity_maze_agent import create_agent
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def run_episode(agent, env, seed=None, receding_horizon=False, verbose=False,
                record=True):
    """Run a single Reactivity Maze episode.

    Returns ``(summary, trajectory)``. The trajectory captures the full
    per-timestep record needed to redraw any figure offline from the DVC-tracked
    data without re-running: the true latent goal ``theta_true``, executed
    ``actions``, sampled ``observations`` (one binary row per step), per-step
    ``rewards`` (0 until the terminal step), the true ``states_visited`` (flat
    index + decoded ``locations``/``knobs``), and the agent's posterior over goal
    identities ``q_static`` after each inference step.
    """
    result = env.reset(seed=seed)
    agent = agent.reset()

    total_reward = 0.0
    steps = 0
    max_steps = env.max_steps

    actions: list[int] = []
    rewards: list[float] = []
    observations = [np.asarray(result.obs).round().astype(int).tolist()]
    states_visited = [int(env._state_idx)]
    loc0, knob0 = unpack_state(env._state_idx)
    locations = [int(loc0)]
    knobs = [int(knob0)]
    q_static_hist: list[list[float]] = []
    planning_time_s = 0.0

    while True:
        if receding_horizon:
            time_remaining = max_steps - steps
        else:
            time_remaining = agent.planning_horizon

        obs = jnp.array(result.obs)
        t0 = time.perf_counter()
        action, agent = agent.step(obs, time_remaining)
        planning_time_s += time.perf_counter() - t0

        if record:
            q_static_hist.append(
                [float(v) for v in np.asarray(agent.q_static_state)]
            )

        if verbose:
            print(f"Step {steps}: action={action}, time_remaining={time_remaining}")
            print(env.render_ascii())

        result = env.step(action)
        total_reward += result.reward
        steps += 1

        if record:
            actions.append(int(action))
            rewards.append(float(result.reward))
            observations.append(np.asarray(result.obs).round().astype(int).tolist())
            states_visited.append(int(env._state_idx))
            loc, knob = unpack_state(env._state_idx)
            locations.append(int(loc))
            knobs.append(int(knob))

        if result.terminated or result.truncated:
            break

    outcome = env.get_outcome()
    summary = {
        "total_reward": total_reward,
        "steps": steps,
        "success": bool(outcome == "goal"),
        "outcome": outcome,
        "cue_visited": bool(env.has_seen_cue),
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "theta_true": int(env.theta),
        "planning_time_s": planning_time_s,
    }
    trajectory = {
        "theta_true": int(env.theta),
        "config_idx": int(env.config_idx),
        "actions": actions,
        "observations": observations,
        "rewards": rewards,
        "states_visited": states_visited,
        "locations": locations,
        "knobs": knobs,
        "q_static": q_static_hist,
        "outcome": outcome,
        "cue_visited": bool(env.has_seen_cue),
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "planning_time_s": planning_time_s,
    }
    return summary, trajectory


def main():
    parser = argparse.ArgumentParser(description="Run Reactivity Maze experiment")
    parser.add_argument("--n-theta", type=int, default=5, help="Number of goal identities (n_static, 1..5)")
    parser.add_argument("--cue-accuracy", type=float, default=1.0, help="Cue observation accuracy at the cue")
    parser.add_argument("--pos-noise", type=float, default=0.05, help="Location channel noise")
    parser.add_argument("--cue-dropoff", type=str, default="random",
                        help="Where the agent lands after the cue: 'random' (default, uniform "
                             "over the 5 nav states) or a nav-state index 0..4 for a deterministic "
                             "drop-off. Applies to both the true env and the agent's transition model.")
    parser.add_argument("--goal-temperature", type=float, default=0.25, help="Goal softmax temperature (lower = sharper)")
    parser.add_argument("--cue-cost", type=float, default=1.0,
                        help="VISIT_CUE action prior weight (<1 makes visiting the cue "
                             "pragmatically costly; only a genuine epistemic drive pays it)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=6, help="Steps per episode (terminal reward at last step)")
    parser.add_argument("--planning-horizon", type=int, default=6, help="Planning horizon")
    parser.add_argument("--planning-iterations", type=int, default=150, help="Planning iterations")
    parser.add_argument("--planning-method", type=str, default="active-inference",
                        choices=["loopy-vbp", "loopy", "region-extended",
                                 "dyn-channel", "nuijten", "vbp-channel",
                                 "precise-info-seeking", "active-inference"],
                        help="Planning method")
    parser.add_argument("--damping", type=float, default=0.6, help="Channel update damping (0-1)")
    parser.add_argument("--receding-horizon", action="store_true", help="Use receding horizon")
    parser.add_argument("--terminal-goal-only", action="store_true",
                        help="Apply goal only at the final planning step (intended for this env)")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    set_seed(args.seed)

    # Cue drop-off: 'random' -> None (uniform over nav states), else a fixed nav index.
    if args.cue_dropoff == "random":
        cue_dropoff = None
    else:
        cue_dropoff = int(args.cue_dropoff)

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"\nReactivity Maze (n_theta={args.n_theta})")
    print(f"  Cue accuracy: {args.cue_accuracy}, pos noise: {args.pos_noise}")
    print(f"  Cue drop-off: {'uniform-random' if cue_dropoff is None else f'fixed nav {cue_dropoff}'}")
    print(f"  Goal temperature: {args.goal_temperature}")
    print(f"  Max steps: {args.max_steps}, planning horizon: {args.planning_horizon}")
    print()

    print("Generating tensors...")
    t0 = time.time()
    goals = sample_configs(args.n_theta)
    T = generate_transition_tensor(args.n_theta, cue_dropoff=cue_dropoff)
    B = generate_observation_tensor(
        args.n_theta, cue_accuracy=args.cue_accuracy, pos_noise=args.pos_noise,
    )
    goal = generate_goal(args.n_theta, goal_temperature=args.goal_temperature)
    start_state_idx = state_index(0, MAX_KNOB)

    print(f"  Transition tensor: {T.shape} ({T.nbytes / 1024:.1f} KB)")
    print(f"  Observation tensor: {B.shape} ({B.nbytes / 1024:.1f} KB)")
    print(f"  Generated in {time.time() - t0:.2f}s")
    print()

    METHOD_MAP = {
        "loopy-vbp": "loopy_vbp",
        "loopy": "loopy_bp",
        "region-extended": "region_extended",
        "dyn-channel": "dyn_channel",
        "nuijten": "nuijten",
        "vbp-channel": "vbp_channel",
        "precise-info-seeking": "precise_info_seeking",
        "active-inference": "active_inference",
    }
    method_key = METHOD_MAP[args.planning_method]

    # Action prior: all actions weight 1 except VISIT_CUE (action 7) at cue_cost.
    # cue_cost < 1 makes visiting the cue pragmatically costly, so a method with no
    # epistemic planning term (e.g. plain loopy BP) will avoid the cue and fail to
    # resolve θ, while the AIF-MP scheme still pays the cost for the information.
    action_prior = np.ones(8, dtype=np.float32)
    action_prior[7] = args.cue_cost  # VISIT_CUE
    action_prior = action_prior / action_prior.sum()

    print("Creating agent...")
    agent = create_agent(
        method_key, T, B, goal,
        n_location_channels=N_LOCATIONS,
        start_state_idx=start_state_idx,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        action_prior=action_prior,
        damping=args.damping,
        terminal_goal_only=args.terminal_goal_only,
    )
    print(f"  Method: {args.planning_method}")
    print(f"  Planning horizon: {args.planning_horizon} ({'receding' if args.receding_horizon else 'fixed'})")
    if args.terminal_goal_only:
        print(f"  Terminal goal only: enabled")
    print(f"  Planning iterations: {args.planning_iterations}")
    print()

    env = ReactivityMazeEnv(goals, obs_tensor=B, max_steps=args.max_steps, cue_dropoff=cue_dropoff)

    print(f"Running {args.episodes} episodes...")
    print("-" * 50)

    results = []
    trajectories = []
    successes = 0
    cue_visits = 0
    t0 = time.time()

    pbar = tqdm(range(args.episodes), desc="Episodes")
    for i in pbar:
        seed = args.seed + i
        episode_result, trajectory = run_episode(
            agent, env, seed=seed, receding_horizon=args.receding_horizon,
            verbose=args.verbose,
        )
        results.append(episode_result)
        trajectories.append(trajectory)
        if episode_result["success"]:
            successes += 1
        if episode_result["cue_visited"]:
            cue_visits += 1
        pbar.set_postfix({
            "success": f"{successes / (i + 1):.1%}",
            "cue": f"{cue_visits / (i + 1):.1%}",
        })

    elapsed = time.time() - t0

    success_rate = successes / args.episodes
    cue_visit_rate = cue_visits / args.episodes
    avg_steps = sum(r["steps"] for r in results) / args.episodes
    rewards = np.array([r["total_reward"] for r in results])
    avg_reward = float(rewards.mean())

    # Outcome breakdown
    outcome_counts = {}
    for r in results:
        outcome_counts[r["outcome"]] = outcome_counts.get(r["outcome"], 0) + 1

    print("-" * 50)
    print(f"Success rate (correct goal): {success_rate:.1%}")
    print(f"Cue visit rate: {cue_visit_rate:.1%}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Outcomes: {outcome_counts}")
    print(f"Total time: {elapsed:.2f}s ({elapsed / args.episodes * 1000:.1f}ms/episode)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": {
                "environment": "reactivity_maze",
                "n_theta": args.n_theta,
                "cue_accuracy": args.cue_accuracy,
                "pos_noise": args.pos_noise,
                "cue_dropoff": ("random" if cue_dropoff is None else cue_dropoff),
                "goal_temperature": args.goal_temperature,
                "cue_cost": args.cue_cost,
                "planning_method": args.planning_method,
                "n_episodes": args.episodes,
                "max_steps": args.max_steps,
                "planning_horizon": args.planning_horizon,
                "planning_iterations": args.planning_iterations,
                "damping": args.damping,
                "receding_horizon": args.receding_horizon,
                "terminal_goal_only": args.terminal_goal_only,
                "seed_start": args.seed,
            },
            "env": {
                "environment": "reactivity_maze",
                "n_theta": args.n_theta,
                "n_states": int(T.shape[0]),
                "n_actions": int(T.shape[3]),
                "start_state_idx": int(start_state_idx),
                "goals": np.asarray(goals).tolist(),
            },
            "n_episodes": args.episodes,
            "per_episode": {
                key: [r[key] for r in results]
                for key in (
                    "total_reward", "steps", "success", "outcome", "cue_visited",
                    "terminated", "truncated", "theta_true", "planning_time_s",
                )
            },
            "results": {
                "success_rate": success_rate,
                "cue_visit_rate": cue_visit_rate,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                "outcome_counts": outcome_counts,
                "successes": successes,
                "total_time_s": elapsed,
            },
            "trajectories": trajectories,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
