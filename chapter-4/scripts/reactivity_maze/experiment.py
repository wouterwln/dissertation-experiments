#!/usr/bin/env python3
"""Reactivity (Epistemic) maze experiment using the Nuijten-MP planner (chapter 4).

The chapter-3 "Minimal Reactive Epistemic Maze" is planned in with chapter-5's
Nuijten-MP message-passing planner instead of the gradient-based temporal-VFE
optimizer. Reward is delivered only at the terminal step (get_final_reward), so
the planner uses a terminal-only goal preference by default.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import jax.numpy as jnp
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments.epistemic_maze import (
    EpistemicMaze, N_NAV_STATES, components_to_state,
)
from src.adapters.generative_model import build_reactivity_generative_model
from src.agents.nuijten_agent import NuijtenAgent


def sample_idx(prob_vec, rng) -> int:
    """Sample a categorical outcome index from a probability vector."""
    p = np.asarray(prob_vec, dtype=np.float64)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def encode_obs(location_vec, context_vec, rng) -> jnp.ndarray:
    """Map (noisy location, context) observations to per-channel integer outcomes."""
    return jnp.array([sample_idx(location_vec, rng), sample_idx(context_vec, rng)],
                     dtype=jnp.int32)


def run_experiment(
    n_episodes: int,
    max_steps: int,
    planning_horizon: int,
    planning_iterations: int,
    receding_horizon: bool,
    n_theta: int,
    cue_accuracy: float,
    location_observation_accuracy: float,
    goal_temperature: float,
    cue_cost_epsilon: float,
    terminal_goal_only: bool,
    seed: int,
    verbose: bool = False,
):
    random.seed(seed)
    rng = np.random.default_rng(seed)

    results = []
    outcome_counts = {"goal": 0, "safe_sink": 0, "wrong_goal": 0, "penalty": 0, "other": 0}
    for ep in tqdm(range(n_episodes), desc="Reactivity maze (Nuijten-MP)", disable=verbose):
        theta = int(rng.integers(0, n_theta))
        start_location = int(rng.integers(0, N_NAV_STATES))
        env = EpistemicMaze.create(
            theta=theta, start_location=start_location, n_theta=n_theta,
            cue_accuracy=cue_accuracy, location_accuracy=location_observation_accuracy,
        )
        model = build_reactivity_generative_model(
            n_theta=n_theta, cue_accuracy=cue_accuracy,
            location_observation_accuracy=location_observation_accuracy,
            goal_temperature=goal_temperature, cue_cost_epsilon=cue_cost_epsilon,
            start_location=start_location,
        )
        agent = NuijtenAgent.create(
            model, planning_horizon, planning_iterations,
            terminal_goal_only=terminal_goal_only,
        )

        # Initial observation (no step taken yet).
        location_vec = env._get_location_observation()
        context_vec = env._get_context_observation()
        obs = encode_obs(location_vec, context_vec, rng)

        trajectory = [int(env.location)]
        actions = []
        for step in range(max_steps):
            time_remaining = (max_steps - step) if receding_horizon else planning_horizon
            action, agent = agent.step(obs, time_remaining)
            location_vec, context_vec, _, _ = env.step(action)
            actions.append(int(action))
            trajectory.append(int(env.location))
            obs = encode_obs(location_vec, context_vec, rng)

        final_reward = float(env.get_final_reward())
        outcome = env.get_outcome()
        outcome_counts[outcome] += 1
        results.append({
            "episode": ep,
            "theta": theta,
            "start_location": start_location,
            "final_reward": final_reward,
            "outcome": outcome,
            "n_steps": len(actions),
            "reached_goal": outcome == "goal",
            "cue_visited": bool(env.has_seen_cue),
            "trajectory": trajectory,
            "actions": actions,
            "final_location": int(env.location),
            "final_knob": int(env.knob),
            "q_theta_final": [float(x) for x in np.asarray(agent.q_static)],
        })
        if verbose:
            print(f"ep {ep}: theta={theta} start={start_location} "
                  f"reward={final_reward:+.2f} outcome={outcome} "
                  f"cue={'Y' if env.has_seen_cue else 'N'} traj={trajectory}")

    n = len(results)
    summary = {
        "n_episodes": n,
        "mean_reward": sum(r["final_reward"] for r in results) / n,
        "success_rate": sum(r["reached_goal"] for r in results) / n,
        "cue_visit_rate": sum(r["cue_visited"] for r in results) / n,
        "avg_steps": sum(r["n_steps"] for r in results) / n,
        "outcome_counts": outcome_counts,
    }
    return summary, results


def save(summary, results, config, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "planner": "nuijten_mp",
        "environment": "reactivity_maze",
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump({"metadata": metadata, "summary": summary, "episodes": results}, f, indent=2)
    with open(output_dir / "stats.json", "w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)
    print(f"Saved results to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Reactivity maze experiment (Nuijten-MP planner)")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--planning-horizon", type=int, default=10)
    p.add_argument("--planning-iterations", type=int, default=10)
    p.add_argument("--receding-horizon", dest="receding_horizon", action="store_true", default=True)
    p.add_argument("--no-receding-horizon", dest="receding_horizon", action="store_false")
    p.add_argument("--n-theta", type=int, default=2)
    p.add_argument("--cue-accuracy", type=float, default=1.0)
    p.add_argument("--location-observation-accuracy", type=float, default=0.95)
    p.add_argument("--goal-temperature", type=float, default=5.0)
    p.add_argument("--cue-cost-epsilon", type=float, default=1.0)
    p.add_argument("--terminal-goal-only", dest="terminal_goal_only", action="store_true", default=True)
    p.add_argument("--no-terminal-goal-only", dest="terminal_goal_only", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="data/reactivity_maze/nuijten")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    config = vars(args).copy()
    config.pop("output_dir")
    summary, results = run_experiment(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        planning_horizon=args.planning_horizon,
        planning_iterations=args.planning_iterations,
        receding_horizon=args.receding_horizon,
        n_theta=args.n_theta,
        cue_accuracy=args.cue_accuracy,
        location_observation_accuracy=args.location_observation_accuracy,
        goal_temperature=args.goal_temperature,
        cue_cost_epsilon=args.cue_cost_epsilon,
        terminal_goal_only=args.terminal_goal_only,
        seed=args.seed,
        verbose=args.verbose,
    )
    print(f"Mean reward: {summary['mean_reward']:.3f}  "
          f"Success: {summary['success_rate']*100:.1f}%  "
          f"Cue-visit: {summary['cue_visit_rate']*100:.1f}%  "
          f"Outcomes: {summary['outcome_counts']}")
    save(summary, results, config, Path(args.output_dir))


if __name__ == "__main__":
    main()
