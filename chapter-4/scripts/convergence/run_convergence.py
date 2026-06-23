#!/usr/bin/env python3
"""Bethe Free Energy convergence sweep for the Nuijten-MP planner (chapter 4).

For the chosen environment (T-Maze or Stochastic Maze) this runs episodes with the
Nuijten-MP planner and, at every planning step, records the per-iteration Bethe Free
Energy (BFE) trace produced by `nuijten_mp_convergence`. The collected traces are
saved to JSON for plotting (`plot_convergence.py`).

A FIXED planning horizon is used (no receding horizon) so that every recorded trace
has the same structure and the curves are directly comparable.

Usage:
    uv run python scripts/convergence/run_convergence.py --environment tmaze
    uv run python scripts/convergence/run_convergence.py --environment stochastic_maze
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import jax.numpy as jnp
from jax import nn
from jax.scipy.special import logsumexp
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.messages import safe_log
from src.inference.convergence import nuijten_mp_convergence
from src.adapters.state_inference import infer_state
from src.adapters.generative_model import (
    build_tmaze_generative_model,
    build_stochastic_maze_generative_model,
)
from src.environments.tmaze import TMaze, State
from src.environments.stochastic_maze import (
    StochasticMaze, DEFAULT_START_STATE, coord_to_state, DEFAULT_GRID_X,
)


def planning_goal(goal, q_static, terminal_goal_only):
    """Mirror NuijtenAgent._planning_goal."""
    if terminal_goal_only and goal.ndim == 2:
        log_C = safe_log(goal)
        log_pref = logsumexp(log_C + safe_log(q_static)[None, :], axis=1)
        return nn.softmax(log_pref)
    return goal


# ---------------------------------------------------------------------------
# Environment adapters: each returns (model, terminal_goal_only) and provides
# make_env / initial_obs / step producing per-channel integer observations.
# ---------------------------------------------------------------------------


def encode_tmaze_obs(location_vec, reward_cue_vec, rng):
    loc_idx = int(jnp.argmax(location_vec))
    cue = np.asarray(reward_cue_vec)
    cue_idx = int(cue.argmax()) if float(cue.max()) > 0.5 else int(rng.integers(0, 2))
    return jnp.array([loc_idx, cue_idx], dtype=jnp.int32)


# T-Maze non-reward cells (states 2 and 4 carry the reward) make valid starts.
TMAZE_VALID_STARTS = [int(State.BOTTOM), int(State.MIDDLE), int(State.TOP_MIDDLE)]


def run_tmaze_episode(model, rng, max_steps, horizon, n_iterations, traces):
    reward_location = random.choice(["left", "right"])
    start_state = int(rng.choice(TMAZE_VALID_STARTS))
    env = TMaze.create(reward_location=reward_location, start_state=start_state)
    q_state = jnp.zeros(model.n_states).at[start_state].set(1.0)
    q_static = jnp.asarray(model.theta_prior)
    last_action = -1

    location_vec = jnp.zeros(5).at[env.agent_state].set(1.0)
    obs = encode_tmaze_obs(location_vec, jnp.array([0.5, 0.5]), rng)

    for _ in range(max_steps):
        action_oh = (jnp.zeros(model.n_actions).at[last_action].set(1.0)
                     if last_action >= 0 else jnp.ones(model.n_actions) / model.n_actions)
        q_state, q_static = infer_state(
            q_state, q_static, model.transition_tensor, model.observation_tensor,
            obs, action_oh,
        )
        goal = planning_goal(model.goal, q_static, terminal_goal_only=True)
        action_dist, _, _, vfe_trace = nuijten_mp_convergence(
            q_state, q_static, model.transition_tensor, model.planning_observation_tensor,
            goal, horizon, n_iterations, action_prior=model.action_prior,
        )
        traces.append(np.asarray(vfe_trace).tolist())
        action = int(jnp.argmax(action_dist))
        location_vec, reward_cue_vec, reward, done = env.step(action)
        obs = encode_tmaze_obs(location_vec, reward_cue_vec, rng)
        last_action = action
        if done:
            break


def _stochastic_valid_starts(env):
    """Cells that are neither sinks nor reward states (well-posed planning starts)."""
    sink_states = {coord_to_state(x, y, env.grid_size_x) for (x, y) in env.sink_states}
    reward_states = {s for s, _ in env.reward_states}
    n_states = env.grid_size_x * env.grid_size_y
    return [s for s in range(n_states) if s not in sink_states and s not in reward_states]


def run_stochastic_episode(model, rng, max_steps, horizon, n_iterations, traces):
    env = StochasticMaze.create(start_state=DEFAULT_START_STATE)
    sink_states = {coord_to_state(x, y, env.grid_size_x) for (x, y) in env.sink_states}
    start_state = int(rng.choice(_stochastic_valid_starts(env)))
    env.reset(start_state=start_state)
    q_state = jnp.zeros(model.n_states).at[start_state].set(1.0)
    q_static = jnp.asarray(model.theta_prior)
    last_action = -1

    obs = jnp.array([env.agent_state], dtype=jnp.int32)

    for _ in range(max_steps):
        action_oh = (jnp.zeros(model.n_actions).at[last_action].set(1.0)
                     if last_action >= 0 else jnp.ones(model.n_actions) / model.n_actions)
        q_state, q_static = infer_state(
            q_state, q_static, model.transition_tensor, model.observation_tensor,
            obs, action_oh,
        )
        # Stop once the belief collapses onto an absorbing sink: planning there is
        # trivially degenerate (BFE == 0) and uninformative for the convergence study.
        if int(jnp.argmax(q_state)) in sink_states:
            break
        goal = planning_goal(model.goal, q_static, terminal_goal_only=True)
        action_dist, _, _, vfe_trace = nuijten_mp_convergence(
            q_state, q_static, model.transition_tensor, model.planning_observation_tensor,
            goal, horizon, n_iterations, action_prior=model.action_prior,
        )
        traces.append(np.asarray(vfe_trace).tolist())
        action = int(jnp.argmax(action_dist))
        _, observation, reward, done = env.step(rng, action)
        obs = jnp.array([observation], dtype=jnp.int32)
        last_action = action
        if done:
            break


ENV_RUNNERS = {
    "tmaze": run_tmaze_episode,
    "stochastic_maze": run_stochastic_episode,
}


def build_model(environment, goal_temperature):
    if environment == "tmaze":
        return build_tmaze_generative_model(goal_temperature=goal_temperature)
    return build_stochastic_maze_generative_model(goal_temperature=goal_temperature)


def main():
    p = argparse.ArgumentParser(description="Nuijten-MP BFE convergence sweep")
    p.add_argument("--environment", required=True, choices=list(ENV_RUNNERS.keys()))
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--planning-horizon", type=int, default=6)
    p.add_argument("--n-iterations", type=int, default=60)
    p.add_argument("--goal-temperature", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = build_model(args.environment, args.goal_temperature)
    runner = ENV_RUNNERS[args.environment]

    traces = []
    for _ in tqdm(range(args.n_episodes), desc=f"convergence:{args.environment}"):
        runner(model, rng, args.max_steps, args.planning_horizon, args.n_iterations, traces)

    arr = np.array(traces)  # (n_traces, n_iterations)
    final_deltas = np.abs(arr[:, -1] - arr[:, -2]) if arr.shape[1] >= 2 else np.array([np.nan])
    tol = 1e-4
    summary = {
        "environment": args.environment,
        "n_traces": int(arr.shape[0]),
        "n_iterations": int(args.n_iterations),
        "planning_horizon": args.planning_horizon,
        "median_final_vfe": float(np.median(arr[:, -1])),
        "median_final_delta": float(np.median(final_deltas)),
        # Fraction of traces whose final-step change is below the convergence tolerance.
        "converged_fraction": float(np.mean(final_deltas < tol)),
        "converged_all": bool(np.all(final_deltas < tol)),
        "convergence_tol": tol,
    }

    output_dir = Path(args.output_dir or f"data/{args.environment}/nuijten")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.pop("output_dir")
    with open(output_dir / "convergence.json", "w") as f:
        json.dump({
            "metadata": {
                "planner": "nuijten_mp",
                "environment": args.environment,
                "timestamp": datetime.now().isoformat(),
                "config": config,
            },
            "summary": summary,
            "traces": traces,
        }, f, indent=2)
    with open(output_dir / "convergence_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{args.environment}] {summary['n_traces']} traces  "
          f"median final VFE={summary['median_final_vfe']:.4f}  "
          f"median |Δ|={summary['median_final_delta']:.2e}  "
          f"converged={summary['converged_fraction']*100:.0f}%")
    print(f"Saved to {output_dir}/convergence.json")


if __name__ == "__main__":
    main()
