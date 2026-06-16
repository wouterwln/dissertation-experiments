"""Smoke tests: run one full episode per environment through the Nuijten-MP agent."""

from pathlib import Path
import sys

import numpy as np
import jax.numpy as jnp
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.generative_model import (
    build_tmaze_generative_model,
    build_reactivity_generative_model,
)
from src.adapters.state_inference import infer_state
from src.agents.nuijten_agent import NuijtenAgent
from src.environments.tmaze import TMaze, State
from src.environments.epistemic_maze import EpistemicMaze, N_NAV_STATES

# Keep iterations/horizon tiny so the smoke test is fast.
ITERS = 3


def _check_beliefs(agent):
    assert jnp.all(jnp.isfinite(agent.q_state))
    assert jnp.all(jnp.isfinite(agent.q_static))
    assert jnp.allclose(agent.q_state.sum(), 1.0, atol=1e-4)
    assert jnp.allclose(agent.q_static.sum(), 1.0, atol=1e-4)


def test_tmaze_episode_runs():
    import random
    random.seed(0)
    rng = np.random.default_rng(0)
    model = build_tmaze_generative_model()
    env = TMaze.create(reward_location="left", start_state=State.MIDDLE)
    agent = NuijtenAgent.create(model, planning_horizon=4, planning_iterations=ITERS,
                                terminal_goal_only=False)

    location_vec = jnp.zeros(5).at[env.agent_state].set(1.0)
    cue_vec = jnp.array([0.5, 0.5])
    obs = jnp.array([int(jnp.argmax(location_vec)), int(rng.integers(0, 2))], dtype=jnp.int32)

    total_reward = 0.0
    for step in range(4):
        action, agent = agent.step(obs, 4 - step)
        assert 0 <= action < 4
        location_vec, cue_vec, reward, done = env.step(action)
        total_reward += float(reward)
        cue = np.asarray(cue_vec)
        cue_idx = int(cue.argmax()) if cue.max() > 0.5 else int(rng.integers(0, 2))
        obs = jnp.array([int(jnp.argmax(location_vec)), cue_idx], dtype=jnp.int32)
        _check_beliefs(agent)
        if done:
            break
    assert total_reward in (-1.0, 0.0, 1.0)


def test_offcue_reading_leaves_theta_belief_unchanged():
    """A {0,1} cue reading from an off-cue 0.5/0.5 column must not update q_static."""
    model = build_tmaze_generative_model()
    q_state = jnp.zeros(5).at[State.MIDDLE].set(1.0)
    q_static = jnp.array([0.5, 0.5])
    action_onehot = jnp.ones(4) / 4
    for cue_idx in (0, 1):
        obs = jnp.array([int(State.MIDDLE), cue_idx], dtype=jnp.int32)
        _, q_static_new = infer_state(
            q_state, q_static, model.transition_tensor, model.observation_tensor,
            obs, action_onehot,
        )
        assert jnp.allclose(q_static_new, q_static, atol=1e-5)


def test_reactivity_episode_runs():
    import random
    random.seed(0)
    rng = np.random.default_rng(0)
    n_theta = 2
    env = EpistemicMaze.create(theta=0, start_location=0, n_theta=n_theta,
                               cue_accuracy=1.0, location_accuracy=0.95)
    model = build_reactivity_generative_model(n_theta=n_theta, start_location=0)
    agent = NuijtenAgent.create(model, planning_horizon=6, planning_iterations=ITERS,
                                terminal_goal_only=True)

    def encode(loc_vec, ctx_vec):
        def samp(v):
            p = np.asarray(v, dtype=np.float64); p = p / p.sum()
            return int(rng.choice(len(p), p=p))
        return jnp.array([samp(loc_vec), samp(ctx_vec)], dtype=jnp.int32)

    obs = encode(env._get_location_observation(), env._get_context_observation())
    for step in range(6):
        action, agent = agent.step(obs, 6)
        assert 0 <= action < 8
        loc_vec, ctx_vec, _, _ = env.step(action)
        obs = encode(loc_vec, ctx_vec)
        _check_beliefs(agent)

    assert isinstance(env.get_final_reward(), float)
    assert env.get_outcome() in {"goal", "safe_sink", "wrong_goal", "penalty", "other"}
