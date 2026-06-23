#!/usr/bin/env python3
"""Stochastic-Maze BP-marginal ablation (chapter 4).

Within-MP ablation: the same region-graph machinery as the Nuijten-MP experiment,
but planning with `region_extended_loopy_bp_planning` (epistemics off). Action
selection is goal-conditioned *marginal* inference against the preference factor
rather than the EFE action-prior fixed point.

With no ambiguity term to avoid high-observation-noise states, the agent is
expected to pursue the goal greedily and gamble through the stochastic interior
(~2/3 success ceiling), in contrast to Nuijten-MP's safe-corridor routing
(~100%). The gap is what the epistemic term buys.

By default the goal is applied as a per-step preference factor (textbook
KL-control). Pass --terminal-goal-only to instead use the marginalised terminal
goal, matching the Nuijten agent's goal spec exactly so the *only* difference is
the planner.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.environments.stochastic_maze import StochasticMaze, DEFAULT_START_STATE
from src.adapters.generative_model import build_stochastic_maze_generative_model
from src.agents.bp_marginal_agent import BPMarginalAgent


def run_experiment(
    n_episodes: int,
    max_steps: int,
    planning_horizon: int,
    planning_iterations: int,
    receding_horizon: bool,
    goal_temperature: float,
    terminal_goal_only: bool,
    damping: float,
    seed: int,
    verbose: bool = False,
):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    model = build_stochastic_maze_generative_model(goal_temperature=goal_temperature)
    goal_state = None

    results = []
    for ep in tqdm(range(n_episodes), desc="Stochastic-Maze (BP-marginal)", disable=verbose):
        env = StochasticMaze.create(start_state=DEFAULT_START_STATE)
        goal_state = env.goal_state
        agent = BPMarginalAgent.create(
            model, planning_horizon, planning_iterations,
            terminal_goal_only=terminal_goal_only, damping=damping,
        )

        # Initial observation: agent localizes at the start state.
        obs = jnp.array([env.agent_state], dtype=jnp.int32)

        trajectory = [int(env.agent_state)]
        actions = []
        total_reward = 0.0
        for step in range(max_steps):
            time_remaining = (max_steps - step) if receding_horizon else planning_horizon
            action, agent = agent.step(obs, time_remaining)
            location_obs, observation, reward, done = env.step(rng, int(action))
            actions.append(int(action))
            trajectory.append(int(env.agent_state))
            total_reward += float(reward)
            obs = jnp.array([observation], dtype=jnp.int32)
            if done:
                break

        results.append({
            "episode": ep,
            "total_reward": total_reward,
            "n_steps": len(actions),
            "reached_goal": total_reward > 0,
            "hit_penalty": total_reward < 0,
            "trajectory": trajectory,
            "actions": actions,
            "final_state": int(env.agent_state),
        })
        if verbose:
            print(f"ep {ep}: reward={total_reward:+.1f} final={env.agent_state} traj={trajectory}")

    n = len(results)
    summary = {
        "n_episodes": n,
        "goal_state": int(goal_state) if goal_state is not None else -1,
        "mean_reward": sum(r["total_reward"] for r in results) / n,
        "success_rate": sum(r["reached_goal"] for r in results) / n,
        "penalty_rate": sum(r["hit_penalty"] for r in results) / n,
        "avg_steps": sum(r["n_steps"] for r in results) / n,
    }
    return summary, results


def save(summary, results, config, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "planner": "region_extended_loopy_bp",
        "environment": "stochastic_maze",
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump({"metadata": metadata, "summary": summary, "episodes": results}, f, indent=2)
    with open(output_dir / "stats.json", "w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)
    print(f"Saved results to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Stochastic-Maze ablation (BP-marginal planner)")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--planning-horizon", type=int, default=10)
    p.add_argument("--planning-iterations", type=int, default=40)
    p.add_argument("--receding-horizon", dest="receding_horizon", action="store_true", default=True)
    p.add_argument("--no-receding-horizon", dest="receding_horizon", action="store_false")
    p.add_argument("--goal-temperature", type=float, default=10.0)
    p.add_argument("--terminal-goal-only", dest="terminal_goal_only", action="store_true",
                   default=False, help="Match the Nuijten goal spec (marginalised terminal goal) "
                                       "instead of a per-step preference factor.")
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="data/stochastic_maze/bp_marginal")
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
        goal_temperature=args.goal_temperature,
        terminal_goal_only=args.terminal_goal_only,
        damping=args.damping,
        seed=args.seed,
        verbose=args.verbose,
    )
    print(f"Mean reward: {summary['mean_reward']:.3f}  "
          f"Success: {summary['success_rate']*100:.1f}%  "
          f"Penalty: {summary['penalty_rate']*100:.1f}%")
    save(summary, results, config, Path(args.output_dir))


if __name__ == "__main__":
    main()
