"""Adapt chapter-3 environment tensors into the Nuijten-MP planner's generative model.

The chapter-3 environments produce:
  - theta-independent transitions  (S, S, A)
  - heterogeneous observation modalities, each (n_obs, S) or (n_obs, S, n_theta)
  - a 2D goal mapping              (S, n_theta)

The Nuijten-MP planner (chapter-5) expects:
  - theta-dependent transitions    (S, S, n_static, A)
  - a single observation tensor    (n_channels, n_obs_types, S, n_static)  [uniform n_obs_types]
  - a goal                         (S, n_static) per-config preference C(x, theta)

This module bridges the two. theta plays the role of `n_static`.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.environments.tmaze import State, create_tmaze_tensors
from src.environments.epistemic_maze import (
    N_LOCATIONS,
    N_NAV_STATES,
    create_epistemic_maze_tensors,
    components_to_state,
)


@dataclass(frozen=True)
class GenerativeModel:
    """A planner-ready generative model for one environment."""
    transition_tensor: jnp.ndarray   # (S, S, n_theta, A)  p(s'|s,theta,a)
    observation_tensor: jnp.ndarray  # (n_channels, n_obs_types, S, n_theta)  for state inference
    planning_observation_tensor: jnp.ndarray  # theta-informative channels only, for the planner
    goal: jnp.ndarray                # (S, n_theta)  preference C(x, theta)
    action_prior: jnp.ndarray        # (A,)
    theta_prior: jnp.ndarray         # (n_theta,)
    start_state_idx: int
    n_states: int
    n_theta: int
    n_actions: int
    n_channels: int
    n_obs_types: int


def _theta_independent_transition_to_4d(transition: jnp.ndarray, n_theta: int) -> jnp.ndarray:
    """(S, S, A) -> (S, S, n_theta, A) by replicating over the theta axis.

    Axis order is (next_state, current_state, theta, action) — the layout the
    planner expects (it transposes to (x_old, x_new, theta, u) internally).
    """
    s, _, a = transition.shape
    return jnp.broadcast_to(transition[:, :, None, :], (s, s, n_theta, a))


def _pad_outcomes(modality: jnp.ndarray, n_obs_types: int) -> jnp.ndarray:
    """Zero-pad a modality's outcome axis (axis 0) up to n_obs_types.

    modality: (n_obs, S, n_theta). Unused outcomes get probability 0, which is
    the chapter-5 convention for packing heterogeneous-cardinality channels into
    a uniform tensor (see RockSample).
    """
    n_obs = modality.shape[0]
    if n_obs == n_obs_types:
        return modality
    pad = [(0, n_obs_types - n_obs), (0, 0), (0, 0)]
    return jnp.pad(modality, pad, mode="constant", constant_values=0.0)


def _lift_to_theta(modality: jnp.ndarray, n_theta: int) -> jnp.ndarray:
    """(n_obs, S) theta-independent modality -> (n_obs, S, n_theta) by broadcast."""
    n_obs, s = modality.shape
    return jnp.broadcast_to(modality[:, :, None], (n_obs, s, n_theta))


def build_tmaze_generative_model(goal_temperature: float = 4.5) -> GenerativeModel:
    """T-Maze (5 states, 4 actions, n_theta=2 reward locations).

    Channels: [location (perfect, 5 outcomes), cue (theta-dependent, 2 outcomes)].
    theta=0 -> reward 'left', theta=1 -> reward 'right'.
    """
    n_theta = 2
    transition, reward_obs, _ = create_tmaze_tensors()  # (5,5,4), (2,5,2), (5,2)

    # Transition -> theta-dependent (transitions are theta-independent here).
    transition_4d = _theta_independent_transition_to_4d(transition, n_theta)

    # Goal mapping, recomputed so goal_temperature is a tunable knob.
    t = goal_temperature
    goal_logits = jnp.array([
        [0.0, 0.0],   # state 0 (bottom/cue)
        [0.0, 0.0],   # state 1 (middle)
        [t, -t],      # state 2 (top-left) preferred when theta=left
        [0.0, 0.0],   # state 3 (top-middle)
        [-t, t],      # state 4 (top-right) preferred when theta=right
    ])
    goal = jax.nn.softmax(goal_logits, axis=0)  # (5, 2)

    # Observation channels.
    n_states = 5
    location_modality = _lift_to_theta(jnp.eye(n_states), n_theta)  # (5, 5, 2)
    cue_modality = reward_obs  # already (2_obs, 5_states, 2_theta)
    n_obs_types = max(location_modality.shape[0], cue_modality.shape[0])  # 5
    observation_tensor = jnp.stack(
        [_pad_outcomes(location_modality, n_obs_types),
         _pad_outcomes(cue_modality, n_obs_types)],
        axis=0,
    )  # (2, 5, 5, 2)

    return GenerativeModel(
        transition_tensor=transition_4d,
        observation_tensor=observation_tensor,
        planning_observation_tensor=observation_tensor[1:2],  # cue channel (theta-informative)
        goal=goal,
        action_prior=jnp.ones(4) / 4.0,
        theta_prior=jnp.array([0.5, 0.5]),
        start_state_idx=int(State.MIDDLE),
        n_states=n_states,
        n_theta=n_theta,
        n_actions=4,
        n_channels=2,
        n_obs_types=n_obs_types,
    )


def build_reactivity_generative_model(
    n_theta: int = 2,
    cue_accuracy: float = 1.0,
    location_observation_accuracy: float = 0.95,
    goal_temperature: float = 5.0,
    cue_cost_epsilon: float = 0.01,
    start_location: int = 0,
) -> GenerativeModel:
    """Reactivity (Epistemic) maze (35 states, 8 actions).

    Channels: [location (noisy, 7 outcomes), context/theta (theta-dependent,
    n_theta+1 outcomes incl. a neutral outcome at index n_theta)].
    """
    (transition, location_observation, theta_observation,
     goal_mapping, action_prior, theta_prior) = create_epistemic_maze_tensors(
        n_theta=n_theta,
        cue_accuracy=cue_accuracy,
        location_observation_accuracy=location_observation_accuracy,
        goal_temperature=goal_temperature,
        cue_cost_epsilon=cue_cost_epsilon,
    )
    # transition (35,35,8); location_observation (7,35);
    # theta_observation (n_theta+1, 35, n_theta); goal_mapping (35, n_theta)

    transition_4d = _theta_independent_transition_to_4d(transition, n_theta)

    location_modality = _lift_to_theta(location_observation, n_theta)  # (7, 35, n_theta)
    theta_modality = theta_observation                                 # (n_theta+1, 35, n_theta)
    n_obs_types = max(location_modality.shape[0], theta_modality.shape[0])
    observation_tensor = jnp.stack(
        [_pad_outcomes(location_modality, n_obs_types),
         _pad_outcomes(theta_modality, n_obs_types)],
        axis=0,
    )  # (2, n_obs_types, 35, n_theta)

    start_state_idx = components_to_state(start_location, 4)  # knob = 4 (max reactivity)

    return GenerativeModel(
        transition_tensor=transition_4d,
        observation_tensor=observation_tensor,
        planning_observation_tensor=observation_tensor[1:2],  # context channel (theta-informative)
        goal=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        start_state_idx=int(start_state_idx),
        n_states=int(N_LOCATIONS * 5),
        n_theta=n_theta,
        n_actions=8,
        n_channels=2,
        n_obs_types=int(n_obs_types),
    )
