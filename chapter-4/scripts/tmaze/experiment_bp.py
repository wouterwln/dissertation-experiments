#!/usr/bin/env python3
"""T-Maze BP-marginal ablation (chapter 4).

Within-MP ablation: the same region-graph machinery as the Nuijten-MP T-Maze
experiment, but planning with canonical (undamped) `region_extended_loopy_bp_planning`
(epistemics off). Action selection is goal-conditioned *marginal* inference against
the preference factor rather than the EFE action-prior fixed point.

The T-Maze hides the reward location theta, revealed only at the cue. Reaching the
cue has no pragmatic value -- it pays off only epistemically, by disambiguating
theta. With no EFE drive the marginal planner has no reason to detour to the cue,
so it cannot resolve theta and gambles on the reward arm: cue-visit rate and
success rate both collapse relative to Nuijten-MP. That contrast is the ablation.
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

from src.environments.tmaze import TMaze, State
from src.adapters.generative_model import build_tmaze_generative_model
from src.agents.bp_marginal_agent import BPMarginalAgent


def encode_obs(location_vec, reward_cue_vec, rng) -> jnp.ndarray:
    """Map (location one-hot, cue 2-vector) to per-channel integer outcomes.

    Channel 0: location (perfectly observed) -> argmax.
    Channel 1: cue -> argmax when informative; off-cue (0.5/0.5) -> random {0,1}
    (theta-uninformative, cancels in the posterior).
    """
    loc_idx = int(jnp.argmax(location_vec))
    cue = np.asarray(reward_cue_vec)
    if float(cue.max()) > 0.5:
        cue_idx = int(cue.argmax())
    else:
        cue_idx = int(rng.integers(0, 2))
    return jnp.array([loc_idx, cue_idx], dtype=jnp.int32)


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
    model = build_tmaze_generative_model(goal_temperature=goal_temperature)

    results = []
    for ep in tqdm(range(n_episodes), desc="T-Maze (BP-marginal)", disable=verbose):
        reward_location = random.choice(["left", "right"])
        env = TMaze.create(reward_location=reward_location, start_state=State.MIDDLE)
        agent = BPMarginalAgent.create(
            model, planning_horizon, planning_iterations,
            terminal_goal_only=terminal_goal_only, damping=damping,
        )

        # Initial observation (no step taken yet): at MIDDLE, cue unseen.
        location_vec = jnp.zeros(5).at[env.agent_state].set(1.0)
        reward_cue_vec = jnp.array([0.5, 0.5])
        obs = encode_obs(location_vec, reward_cue_vec, rng)

        trajectory = [int(env.agent_state)]
        actions = []
        total_reward = 0.0
        done = False
        for step in range(max_steps):
            time_remaining = (max_steps - step) if receding_horizon else planning_horizon
            action, agent = agent.step(obs, time_remaining)
            location_vec, reward_cue_vec, reward, done = env.step(action)
            actions.append(int(action))
            trajectory.append(int(env.agent_state))
            total_reward += float(reward)
            obs = encode_obs(location_vec, reward_cue_vec, rng)
            if done:
                break

        results.append({
            "episode": ep,
            "reward_location": reward_location,
            "total_reward": total_reward,
            "n_steps": len(actions),
            "reached_goal": total_reward > 0,
            "cue_visited": bool(env.has_seen_cue),
            "trajectory": trajectory,
            "actions": actions,
            "final_state": int(env.agent_state),
        })
        if verbose:
            print(f"ep {ep}: theta={reward_location} reward={total_reward:+.1f} "
                  f"cue={'Y' if env.has_seen_cue else 'N'} traj={trajectory}")

    n = len(results)
    summary = {
        "n_episodes": n,
        "mean_reward": sum(r["total_reward"] for r in results) / n,
        "success_rate": sum(r["reached_goal"] for r in results) / n,
        "cue_visit_rate": sum(r["cue_visited"] for r in results) / n,
        "avg_steps": sum(r["n_steps"] for r in results) / n,
    }
    return summary, results


def save(summary, results, config, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "planner": "region_extended_loopy_bp",
        "environment": "tmaze",
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump({"metadata": metadata, "summary": summary, "episodes": results}, f, indent=2)
    with open(output_dir / "stats.json", "w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)
    print(f"Saved results to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="T-Maze ablation (BP-marginal planner)")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--planning-horizon", type=int, default=6)
    p.add_argument("--planning-iterations", type=int, default=20)
    p.add_argument("--receding-horizon", dest="receding_horizon", action="store_true", default=True)
    p.add_argument("--no-receding-horizon", dest="receding_horizon", action="store_false")
    p.add_argument("--goal-temperature", type=float, default=4.5)
    p.add_argument("--terminal-goal-only", dest="terminal_goal_only", action="store_true", default=True)
    p.add_argument("--no-terminal-goal-only", dest="terminal_goal_only", action="store_false")
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="data/tmaze/bp_marginal")
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
          f"Cue-visit: {summary['cue_visit_rate']*100:.1f}%")
    save(summary, results, config, Path(args.output_dir))


if __name__ == "__main__":
    main()
