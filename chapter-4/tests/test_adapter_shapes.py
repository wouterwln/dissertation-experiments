"""Shape and semantics regression tests for the generative-model adapters."""

from pathlib import Path
import sys

import jax.numpy as jnp
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.generative_model import (
    build_tmaze_generative_model,
    build_reactivity_generative_model,
)
from src.environments.epistemic_maze import EpistemicAction, components_to_state


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


def test_reactivity_shapes():
    m = build_reactivity_generative_model(n_theta=2)
    assert m.transition_tensor.shape == (35, 35, 2, 8)
    assert m.observation_tensor.shape == (2, 7, 35, 2)
    assert m.planning_observation_tensor.shape == (1, 7, 35, 2)
    assert m.goal.shape == (35, 2)
    assert m.action_prior.shape == (8,)
    assert m.theta_prior.shape == (2,)


def test_reactivity_transition_normalized_and_theta_independent():
    m = build_reactivity_generative_model(n_theta=2)
    sums = m.transition_tensor.sum(axis=0)
    assert jnp.allclose(sums, 1.0, atol=1e-5)
    assert jnp.allclose(m.transition_tensor[:, :, 0, :], m.transition_tensor[:, :, 1, :])


def test_reactivity_observation_semantics():
    m = build_reactivity_generative_model(n_theta=2)
    loc, ctx = m.observation_tensor[0], m.observation_tensor[1]
    # location channel per (state, theta) sums to 1
    assert jnp.allclose(loc.sum(axis=0), 1.0, atol=1e-5)
    # context channel padded outcomes 3..6 are exactly zero (n_theta+1 = 3 used)
    assert jnp.all(ctx[3:] == 0.0)
    # off-cue states fire the neutral outcome (index n_theta == 2) with prob 1
    non_cue_states = [s for s in range(35) if s % 7 != 6]
    assert jnp.allclose(ctx[2, jnp.array(non_cue_states), :], 1.0, atol=1e-5)


def test_reactivity_action_prior_favours_cue():
    m = build_reactivity_generative_model(n_theta=2, cue_cost_epsilon=0.01)
    # cue_cost_epsilon < 1 makes VISIT_CUE more probable than a baseline nav action
    assert jnp.allclose(m.action_prior.sum(), 1.0, atol=1e-5)
    assert m.action_prior[int(EpistemicAction.VISIT_CUE)] > m.action_prior[0]


def test_reactivity_start_state():
    m = build_reactivity_generative_model(n_theta=2, start_location=3)
    assert m.start_state_idx == components_to_state(3, 4)
