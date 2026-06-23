"""Shape and semantics regression tests for the generative-model adapters."""

from pathlib import Path
import sys

import jax.numpy as jnp
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.generative_model import (
    build_tmaze_generative_model,
    build_stochastic_maze_generative_model,
)


def test_tmaze_shapes():
    m = build_tmaze_generative_model()
    assert m.transition_tensor.shape == (5, 5, 2, 4)
    assert m.observation_tensor.shape == (2, 5, 5, 2)
    assert m.planning_observation_tensor.shape == (1, 5, 5, 2)
    assert m.goal.shape == (5, 2)
    assert m.action_prior.shape == (4,)
    assert m.theta_prior.shape == (2,)
    assert m.start_state_idx == 1


def test_tmaze_transition_normalized_and_theta_independent():
    m = build_tmaze_generative_model()
    # next-state distribution sums to 1 for each (current_state, theta, action)
    sums = m.transition_tensor.sum(axis=0)
    assert jnp.allclose(sums, 1.0, atol=1e-5)
    # theta-independent: theta=0 and theta=1 slices identical
    assert jnp.allclose(m.transition_tensor[:, :, 0, :], m.transition_tensor[:, :, 1, :])


def test_tmaze_observation_semantics():
    m = build_tmaze_generative_model()
    loc, cue = m.observation_tensor[0], m.observation_tensor[1]
    # location channel: per (state, theta) outcomes sum to 1
    assert jnp.allclose(loc.sum(axis=0), 1.0, atol=1e-5)
    # location is a perfect one-hot identity (theta-independent)
    assert jnp.allclose(loc[:, :, 0], jnp.eye(5))
    # cue channel: padded outcomes 2..4 are exactly zero
    assert jnp.all(cue[2:] == 0.0)
    # at cue state 0, cue is informative about theta; off-cue it is 0.5/0.5
    assert jnp.allclose(cue[:2, 0, 0], jnp.array([1.0, 0.0]))
    assert jnp.allclose(cue[:2, 0, 1], jnp.array([0.0, 1.0]))
    assert jnp.allclose(cue[:2, 1, 0], jnp.array([0.5, 0.5]))


def test_tmaze_goal_normalized():
    m = build_tmaze_generative_model()
    assert jnp.allclose(m.goal.sum(axis=0), 1.0, atol=1e-5)
    # left arm (state 2) preferred for theta=0, right arm (state 4) for theta=1
    assert int(jnp.argmax(m.goal[:, 0])) == 2
    assert int(jnp.argmax(m.goal[:, 1])) == 4


def test_stochastic_maze_shapes():
    m = build_stochastic_maze_generative_model()
    # default 5x5 = 25 states, 4 actions, no hidden context (n_theta=1)
    assert m.transition_tensor.shape == (25, 25, 1, 4)
    assert m.observation_tensor.shape == (1, 25, 25, 1)
    assert m.planning_observation_tensor.shape == (1, 25, 25, 1)
    assert m.goal.shape == (25, 1)
    assert m.action_prior.shape == (4,)
    assert m.theta_prior.shape == (1,)
    # start_state = Julia 11 (1-indexed) = state 10 (0-indexed) = (0,2)
    assert m.start_state_idx == 10


def test_stochastic_maze_transition_normalized():
    m = build_stochastic_maze_generative_model()
    # next-state distribution sums to 1 for each (current_state, theta, action)
    sums = m.transition_tensor.sum(axis=0)
    assert jnp.allclose(sums, 1.0, atol=1e-5)


def test_stochastic_maze_observation_normalized():
    m = build_stochastic_maze_generative_model()
    obs = m.observation_tensor[0]  # (n_obs, n_states, 1)
    # per (state, theta) the observation distribution sums to 1
    assert jnp.allclose(obs.sum(axis=0), 1.0, atol=1e-5)


def test_stochastic_maze_goal_peaked_on_goal_state():
    m = build_stochastic_maze_generative_model()
    assert jnp.allclose(m.goal.sum(axis=0), 1.0, atol=1e-5)
    # terminal goal is a near-delta on the +1 goal state (linear index 14 = (4,2))
    assert int(jnp.argmax(m.goal[:, 0])) == 14
    assert float(m.goal[14, 0]) > 0.99


def test_stochastic_maze_action_prior_uniform():
    m = build_stochastic_maze_generative_model()
    assert jnp.allclose(m.action_prior, 0.25, atol=1e-6)
