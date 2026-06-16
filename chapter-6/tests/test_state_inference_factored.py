"""Tests for the spec-aware Bayesian filter under multi-component θ.

Validates the per-factor narrow path by constructing a synthetic 2-component
ThetaSpec where each B-channel sees only one component, and comparing the
factored filter output against a brute-force flat-joint reference.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.inference.state_inference import bayesian_state_update


def _flat_reference_update(q_state, q_static_flat, T_flat, B_flat_list, obs, action_onehot):
    """Brute-force reference: flatten the joint θ axis and run the filter
    with single-component specs. Used to verify the spec-aware filter."""
    n_static = q_static_flat.shape[0]
    flat_theta_spec = ThetaSpec(component_sizes=(n_static,))
    flat_t_factor_spec = FactorSpec(theta_components=(0,))
    flat_b_factor_specs = tuple(
        FactorSpec(theta_components=(0,) if B.ndim == 3 else ())
        for B in B_flat_list
    )
    log_B_flat_list = tuple(jnp.log(B + 1e-12) for B in B_flat_list)
    return bayesian_state_update(
        q_state, q_static_flat, T_flat, log_B_flat_list, obs, action_onehot,
        theta_spec=flat_theta_spec,
        t_factor_spec=flat_t_factor_spec,
        b_factor_specs=flat_b_factor_specs,
    )


def test_factored_filter_matches_flat_joint_two_components():
    """Two binary θ components, T full-joint, B-channels each see one component.

    Build a small synthetic env with:
      - n_states = 4, n_actions = 2, K=2 binary components -> n_static = 4.
      - T sees both components (theta_components=(0,1)).
      - B_0 (size 2) sees only component 0; B_1 (size 2) sees only component 1.
      - One step of the filter from a non-uniform prior.

    Compare:
      (a) Spec-aware filter with factored shapes.
      (b) Flat-joint filter where T is shaped (4,4,4,2) and Bs are shaped
          (2, 4, 4) with the per-channel axis broadcast to the full joint.

    The two should agree on q_state_new and q_static_new.
    """
    rng = np.random.default_rng(7)

    n_states, n_actions = 4, 2
    theta_spec = ThetaSpec(component_sizes=(2, 2))
    n_static = theta_spec.n_static  # = 4

    # T narrow: (4, 4, 2, 2, 2) — see both components.
    T_narrow = rng.random((n_states, n_states, 2, 2, n_actions))
    T_narrow = T_narrow / T_narrow.sum(axis=0, keepdims=True)
    T_narrow = jnp.asarray(T_narrow, dtype=jnp.float32)

    # B_0 narrow: (n_obs=2, n_states, 2) — sees only component 0.
    B0_narrow = rng.random((2, n_states, 2))
    B0_narrow = B0_narrow / B0_narrow.sum(axis=0, keepdims=True)
    B0_narrow = jnp.asarray(B0_narrow, dtype=jnp.float32)

    # B_1 narrow: (n_obs=3, n_states, 2) — sees only component 1.
    B1_narrow = rng.random((3, n_states, 2))
    B1_narrow = B1_narrow / B1_narrow.sum(axis=0, keepdims=True)
    B1_narrow = jnp.asarray(B1_narrow, dtype=jnp.float32)

    # Non-uniform priors.
    q_state = jnp.asarray(rng.dirichlet(np.ones(n_states)), dtype=jnp.float32)
    q_static_flat = jnp.asarray(rng.dirichlet(np.ones(n_static)), dtype=jnp.float32)

    # Observations: indicator vectors per channel.
    obs0 = jnp.asarray([1.0, 0.0])
    obs1 = jnp.asarray([0.0, 1.0, 0.0])
    obs = jnp.concatenate([obs0, obs1])

    action_onehot = jnp.asarray([1.0, 0.0])

    # --- Spec-aware path ---
    t_factor_spec = FactorSpec(theta_components=(0, 1))
    b_factor_specs = (
        FactorSpec(theta_components=(0,)),
        FactorSpec(theta_components=(1,)),
    )
    log_B_narrow_list = (
        jnp.log(B0_narrow + 1e-12),
        jnp.log(B1_narrow + 1e-12),
    )
    q_state_new, q_static_new = bayesian_state_update(
        q_state, q_static_flat, T_narrow, log_B_narrow_list,
        obs, action_onehot,
        theta_spec=theta_spec,
        t_factor_spec=t_factor_spec,
        b_factor_specs=b_factor_specs,
    )

    # --- Flat-joint reference: rebuild T, B as if θ were a single n_static axis ---
    # T_flat: (n_states_new, n_states_old, n_static, n_actions). For
    # T_narrow (..., 2, 2, ...), n_static = 4 = ravel(theta_0, theta_1) with
    # row-major ravel order.
    T_flat = jnp.asarray(np.asarray(T_narrow).reshape(n_states, n_states, 4, n_actions))

    # B0_flat: (n_obs, n_states, n_static). B0 sees only component 0; broadcast over component 1.
    # For each ravel index j = (theta_0, theta_1), B0_flat[:, :, j] = B0_narrow[:, :, theta_0].
    B0_flat = np.zeros((2, n_states, 4), dtype=np.float32)
    for theta0 in range(2):
        for theta1 in range(2):
            j = np.ravel_multi_index((theta0, theta1), (2, 2))
            B0_flat[:, :, j] = np.asarray(B0_narrow[:, :, theta0])
    B0_flat = jnp.asarray(B0_flat)

    B1_flat = np.zeros((3, n_states, 4), dtype=np.float32)
    for theta0 in range(2):
        for theta1 in range(2):
            j = np.ravel_multi_index((theta0, theta1), (2, 2))
            B1_flat[:, :, j] = np.asarray(B1_narrow[:, :, theta1])
    B1_flat = jnp.asarray(B1_flat)

    q_state_ref, q_static_ref = _flat_reference_update(
        q_state, q_static_flat, T_flat, [B0_flat, B1_flat], obs, action_onehot,
    )

    np.testing.assert_allclose(np.asarray(q_state_new), np.asarray(q_state_ref), atol=1e-5)
    # Spec-aware path returns rank-K (2, 2); reference returns rank-1 (4,).
    # Compare after flattening.
    np.testing.assert_allclose(
        np.asarray(q_static_new).reshape(-1),
        np.asarray(q_static_ref).reshape(-1),
        atol=1e-5,
    )


def test_factored_filter_single_component_smoke():
    """Single-component specs run end-to-end and return rank-1 q_static."""
    rng = np.random.default_rng(11)
    n_states, n_actions, n_static = 5, 3, 6

    T = rng.random((n_states, n_states, n_static, n_actions))
    T = T / T.sum(axis=0, keepdims=True)
    T = jnp.asarray(T, dtype=jnp.float32)

    B_dep = rng.random((4, n_states, n_static))
    B_dep = B_dep / B_dep.sum(axis=0, keepdims=True)
    B_dep = jnp.asarray(B_dep, dtype=jnp.float32)

    B_indep = rng.random((2, n_states))
    B_indep = B_indep / B_indep.sum(axis=0, keepdims=True)
    B_indep = jnp.asarray(B_indep, dtype=jnp.float32)

    q_state = jnp.asarray(rng.dirichlet(np.ones(n_states)), dtype=jnp.float32)
    q_static = jnp.asarray(rng.dirichlet(np.ones(n_static)), dtype=jnp.float32)
    obs = jnp.concatenate([jnp.asarray([1.0, 0.0, 0.0, 0.0]), jnp.asarray([0.0, 1.0])])
    action_onehot = jnp.asarray([0.0, 1.0, 0.0])

    theta_spec = ThetaSpec(component_sizes=(n_static,))
    t_factor_spec = FactorSpec(theta_components=(0,))
    b_factor_specs = (
        FactorSpec(theta_components=(0,)),
        FactorSpec(theta_components=()),
    )
    log_B_list = (jnp.log(B_dep + 1e-12), jnp.log(B_indep + 1e-12))
    q_state_new, q_static_new = bayesian_state_update(
        q_state, q_static, T, log_B_list, obs, action_onehot,
        theta_spec=theta_spec,
        t_factor_spec=t_factor_spec,
        b_factor_specs=b_factor_specs,
    )

    assert q_state_new.shape == (n_states,)
    assert q_static_new.shape == (n_static,)
    np.testing.assert_allclose(float(q_state_new.sum()), 1.0, atol=1e-5)
    np.testing.assert_allclose(float(q_static_new.sum()), 1.0, atol=1e-5)
