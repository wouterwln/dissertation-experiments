"""Bayesian state inference for POMDPs.

Implements single-step Bayesian filtering: predict via transition model,
then update with observation likelihood. Maintains joint posterior over
hidden state x and (possibly factored) static configuration θ.

Supports theta-aware inference: observation channels marked as
theta-independent (e.g., position) contribute to state inference but not
to theta inference.

Spec-aware: ThetaSpec / FactorSpecs are required. The filter materializes
a rank-(1 + n_components) joint posterior of shape
(n_states, *component_sizes) internally and returns the θ marginal at
that same rank. For the M1/M2 single-component case
(component_sizes=(n_static,) and all factors seeing component 0 or none),
this collapses to a rank-2 joint and a rank-1 θ marginal — the legacy
shapes — and produces byte-identical math.

Reference: MinigridMP-AIF-JAX/agents/frozen_lake_agent.py _infer_state()
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn
from jax.scipy.special import logsumexp

from learningpomdp.factors import FactorSpec, ThetaSpec, align_to_full_components

EPSILON = 1e-12


def reset_episode_theta(
    q_static: jnp.ndarray, theta_spec: ThetaSpec
) -> jnp.ndarray:
    """Re-uniform the per-episode component axes of q(θ).

    Marginalizes out the components listed in
    `theta_spec.episode_reset_components` (preserving any joint distribution
    over the persistent components) and restores a uniform prior on those
    axes. Used between episodes inside a block: the persistent posterior
    carries over, the per-episode prior resets to flat.

    Identity when `episode_reset_components` is empty.
    """
    axes = tuple(theta_spec.episode_reset_components)
    if not axes:
        return q_static
    q = q_static.reshape(theta_spec.component_sizes)
    q_marg = q.sum(axis=axes, keepdims=True)
    uniform_factor = math.prod(theta_spec.component_sizes[i] for i in axes)
    return jnp.broadcast_to(q_marg / uniform_factor, q.shape)


def _condition_on_persistent(
    q: jnp.ndarray,
    persistent_idx_tuple: tuple[int, ...],
    theta_spec: ThetaSpec,
) -> tuple[jnp.ndarray, tuple]:
    """Build conditioned q_static given a sampled persistent value tuple.

    Returns
    -------
    q_cond : (*component_sizes,) tensor with persistent axes one-hot at the
        sampled values and episodic axes equal to q(θ_episodic | θ̃_persistent).
    theta_persistent_full : length-n_components tuple where persistent
        positions hold the sampled int and episodic positions hold None.
    """
    persistent_axes = theta_spec.persistent_components
    persistent_to_value = {
        axis: int(persistent_idx_tuple[i]) for i, axis in enumerate(persistent_axes)
    }
    slicer = tuple(
        persistent_to_value[k] if k in persistent_to_value else slice(None)
        for k in range(theta_spec.n_components)
    )
    q_episodic_slice = q[slicer]
    s = q_episodic_slice.sum()
    # If the slice has zero mass, fall back to a uniform conditional to keep
    # the conditioned q_static a valid distribution. Unreachable when the
    # persistent value was *sampled* from q (PSRL/BOSS: a sampled value has
    # mass by construction); reachable when a caller pins it independently —
    # `PersistentOraclePlanner` pins the truth, which the filter's float32
    # softmax can underflow to 0, and warns before calling in.
    safe_size = max(int(np.prod(q_episodic_slice.shape)), 1)
    q_episodic_cond = jnp.where(
        s > 0, q_episodic_slice / (s + EPSILON), jnp.ones_like(q_episodic_slice) / safe_size
    )

    q_cond = jnp.zeros(theta_spec.component_sizes).at[slicer].set(q_episodic_cond)

    theta_persistent_full = tuple(
        persistent_to_value[k] if k in persistent_to_value else None
        for k in range(theta_spec.n_components)
    )
    return q_cond, theta_persistent_full


def sample_persistent_theta(
    q_static: jnp.ndarray,
    theta_spec: ThetaSpec,
    rng: np.random.Generator,
) -> tuple[jnp.ndarray, tuple]:
    """Flat-sample θ̃_persistent ~ q(θ_persistent); return conditioned q_static.

    Persistent axes of the returned tensor are one-hot at the sampled value;
    episodic axes equal the *true conditional* q(θ_episodic | θ̃_persistent)
    (slice + renormalise of the joint, not just the episodic marginal).

    Returns
    -------
    q_static_cond : (*theta_spec.component_sizes,) tensor.
    theta_persistent_full : length-n_components tuple, ints at persistent
        positions and None at episodic positions.

    When `theta_spec.persistent_components` is empty, returns (q_static, ())
    as the degenerate identity.
    """
    persistent_axes = theta_spec.persistent_components
    q = q_static.reshape(theta_spec.component_sizes)
    if not persistent_axes:
        return q, ()

    episodic_axes = tuple(theta_spec.episode_reset_components)
    if episodic_axes:
        q_persistent = q.sum(axis=episodic_axes)
    else:
        q_persistent = q
    # q_persistent shape = (component_sizes[k] for k in sorted(persistent_axes,))
    # which matches persistent_axes since persistent_components is sorted.

    p_flat = np.asarray(q_persistent, dtype=np.float64).reshape(-1)
    total = p_flat.sum()
    if total <= 0:
        # Degenerate: uniform fallback over the persistent space.
        p_flat = np.ones_like(p_flat) / p_flat.size
    else:
        p_flat = p_flat / total
    flat_idx = int(rng.choice(p_flat.size, p=p_flat))
    persistent_sizes = tuple(theta_spec.component_sizes[k] for k in persistent_axes)
    persistent_idx_tuple = tuple(int(i) for i in np.unravel_index(flat_idx, persistent_sizes))

    return _condition_on_persistent(q, persistent_idx_tuple, theta_spec)


def sample_persistent_theta_factored(
    q_static: jnp.ndarray,
    theta_spec: ThetaSpec,
    rng: np.random.Generator,
) -> tuple[jnp.ndarray, tuple]:
    """Per-component independent sample (Factored BA-POMCP).

    Like ``sample_persistent_theta``, but each persistent axis is drawn
    independently from its 1-D marginal q(θ_persistent_k). On environments
    where the persistent posterior factorises (e.g. RockSample's independent
    rocks) this is distributionally equivalent to flat sampling on the joint
    persistent marginal; otherwise the two diverge.
    """
    persistent_axes = theta_spec.persistent_components
    q = q_static.reshape(theta_spec.component_sizes)
    if not persistent_axes:
        return q, ()

    n_components = theta_spec.n_components
    persistent_idx_list: list[int] = []
    for k_persist in persistent_axes:
        marginal_axes = tuple(j for j in range(n_components) if j != k_persist)
        if marginal_axes:
            q_k = q.sum(axis=marginal_axes)
        else:
            q_k = q
        p_k = np.asarray(q_k, dtype=np.float64)
        total = p_k.sum()
        if total <= 0:
            p_k = np.ones_like(p_k) / p_k.size
        else:
            p_k = p_k / total
        persistent_idx_list.append(int(rng.choice(p_k.size, p=p_k)))
    persistent_idx_tuple = tuple(persistent_idx_list)

    return _condition_on_persistent(q, persistent_idx_tuple, theta_spec)


@functools.partial(
    jax.jit, static_argnames=("theta_spec", "t_factor_spec", "b_factor_specs")
)
def bayesian_state_update(
    q_state: jnp.ndarray,
    q_static: jnp.ndarray,
    T: jnp.ndarray,
    log_B_list: tuple[jnp.ndarray, ...],
    obs: jnp.ndarray,
    action_onehot: jnp.ndarray,
    theta_spec: ThetaSpec,
    t_factor_spec: FactorSpec,
    b_factor_specs: tuple[FactorSpec, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step Bayesian state and theta update.

    1. Predict x_new from x_old via transition.
    2. Compute observation likelihood per channel.
    3. Form joint posterior and marginalize.

    Args:
        q_state: (n_states,) prior belief over hidden state.
        q_static: prior belief over θ. Either rank-1 flat shape
            (n_static,) or rank-(n_components) shape
            (*theta_spec.component_sizes,); reshaped internally either way.
        T: Transition tensor with θ axes laid out per `t_factor_spec`.
            Single-component default: (n_states_new, n_states_old, n_static, n_actions).
        log_B_list: Per-modality log-observation tensors —
            ``jnp.log(B_c + EPSILON)`` precomputed once at episode setup.
            Each shape matches ``B_c``: (n_obs, n_states, *b_factor_spec.shape_for(θ))
            or (n_obs, n_states) when θ-independent.
        obs: Observation vector — concatenation of per-channel obs.
        action_onehot: (n_actions,) one-hot encoding of previous action.
        theta_spec: Required θ-space spec.
        t_factor_spec: Required FactorSpec for T.
        b_factor_specs: Required per-channel FactorSpecs.

    Returns:
        q_state_new: (n_states,) posterior over hidden state.
        q_static_new: rank-(n_components) posterior over θ shaped
            (*theta_spec.component_sizes,). Single-component envs get
            (n_static,) — flat.
    """
    n_states = q_state.shape[0]

    # 1. Predict joint q(x_new, θ | obs_<t, action) under factored prior:
    #    q(x_new, θ) ∝ q(θ) · sum_{x_old} T(x_new|x_old, θ_components_T, u) · q(x_old)
    # T shape: (n_states_new, n_states_old, *t_factor_spec.shape_for(θ), n_actions).
    # Contract last axis (n_actions) with action_onehot:
    T_action = jnp.tensordot(T, action_onehot, axes=([T.ndim - 1], [0]))
    # T_action shape: (n_states_new, n_states_old, *t_factor_spec.shape_for(θ))
    # Contract axis 1 (n_states_old) with q_state:
    predicted_state_per_theta = jnp.tensordot(T_action, q_state, axes=([1], [0]))
    # predicted_state_per_theta shape: (n_states_new, *t_factor_spec.shape_for(θ))

    # Broadcast onto the full (*component_sizes) θ-axes layout:
    predicted_aligned = align_to_full_components(
        predicted_state_per_theta, t_factor_spec, theta_spec, leading_axes=1
    )
    # predicted_aligned shape: (n_states_new, *all_components_or_1)

    # Form joint with the factored θ prior:
    # q_static may arrive flat (n_static,) or already factored
    # (*component_sizes,); reshape works either way.
    q_static_factored = q_static.reshape(theta_spec.component_sizes)
    predicted_joint = predicted_aligned * q_static_factored[None]
    # predicted_joint shape: (n_states_new, *theta_spec.component_sizes)
    predicted_joint = predicted_joint / (predicted_joint.sum() + EPSILON)

    # 2. Observation log-likelihood per channel — accumulate onto the full
    # joint shape via broadcasted alignment.
    log_lik_joint = jnp.zeros((n_states,) + theta_spec.component_sizes)

    offset = 0
    for c, log_B_c in enumerate(log_B_list):
        n_obs = log_B_c.shape[0]
        obs_c = obs[offset:offset + n_obs]
        b_spec = b_factor_specs[c]

        if b_spec.is_theta_independent:
            # log_B_c shape: (n_obs, n_states); contract obs.
            log_lik_x = jnp.tensordot(obs_c, log_B_c, axes=([0], [0]))
            # Broadcast across θ axes: insert size-1 for every component.
            log_lik_aligned = log_lik_x.reshape(
                (n_states,) + (1,) * theta_spec.n_components
            )
        else:
            # log_B_c shape: (n_obs, n_states, *b_spec.shape_for(θ)); contract obs.
            log_lik_x_b = jnp.tensordot(obs_c, log_B_c, axes=([0], [0]))
            # log_lik_x_b shape: (n_states, *b_spec.shape_for(θ))
            log_lik_aligned = align_to_full_components(
                log_lik_x_b, b_spec, theta_spec, leading_axes=1
            )

        log_lik_joint = log_lik_joint + log_lik_aligned
        offset += n_obs

    # 3. Joint posterior: q(x, θ | obs) ∝ B(obs|x,θ) · q(x, θ | obs_<t)
    log_joint = log_lik_joint + jnp.log(predicted_joint + EPSILON)

    # Marginalize.
    theta_axes = tuple(range(1, 1 + theta_spec.n_components))
    log_q_x = logsumexp(log_joint, axis=theta_axes)
    q_state_new = nn.softmax(log_q_x)

    log_q_theta = logsumexp(log_joint, axis=0)  # (*component_sizes,)
    # Joint softmax over all θ axes — done via flat softmax then reshape
    # to preserve the math byte-for-byte vs the legacy flat path.
    q_static_new = nn.softmax(log_q_theta.reshape(-1)).reshape(
        theta_spec.component_sizes
    )

    return q_state_new, q_static_new
