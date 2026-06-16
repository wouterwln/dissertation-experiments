"""Stress tests for bp_core.py under multi-component θ.

Validates the per-factor narrow path against equivalent "lifted" versions
where every factor is declared to see all components (with tensors broadcast
over the components a factor doesn't actually depend on). Under factored
q(θ) and broadcast-equivalent tensors, the two paths must produce
mathematically identical messages and cavities.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from learningpomdp.factors import (
    FactorSpec,
    ThetaSpec,
    align_to_full_components,
)
from learningpomdp.planning.bp_core import (
    compute_log_reduced,
    compute_obs_to_theta_msgs,
    compute_obs_to_x_msgs,
    compute_theta_cavities,
)


def _lift_to_full(arr, factor, theta_spec, leading_axes):
    """Reshape a per-factor narrow tensor to span all components (size-1
    broadcast on missing components), then expand to the full
    component_sizes."""
    aligned = align_to_full_components(arr, factor, theta_spec, leading_axes)
    full_shape = list(arr.shape[:leading_axes]) + list(theta_spec.component_sizes)
    return jnp.broadcast_to(aligned, tuple(full_shape))


def test_compute_log_reduced_narrow_vs_lifted():
    """For T with theta_components=(0,) (sees only component 0), the narrow
    path on (T, x_old, x_new, c_0, u) must match the lifted path on
    (T, x_old, x_new, c_0, c_1, u)."""
    rng = np.random.default_rng(13)
    horizon, n_states, n_actions = 3, 4, 2
    theta_spec = ThetaSpec(component_sizes=(2, 3))

    factor_narrow = FactorSpec(theta_components=(0,))
    factor_full = FactorSpec(theta_components=(0, 1))

    # Narrow kernels: shape (T, x_old, x_new, c_0, u).
    log_kernels_narrow = jnp.asarray(
        rng.standard_normal((horizon, n_states, n_states, 2, n_actions))
    )
    # Lifted kernels: broadcast over component 1 (size 3).
    log_kernels_lifted = jnp.broadcast_to(
        log_kernels_narrow[:, :, :, :, None, :],
        (horizon, n_states, n_states, 2, 3, n_actions),
    )

    # Narrow cavity: shape (T, c_0).
    log_cav_narrow = jnp.asarray(rng.standard_normal((horizon, 2)))
    # Lifted cavity: broadcast over component 1 (size 3); but the lifted
    # cavity must represent the same quantity, so divide log mass uniformly
    # over component 1 (i.e. add log(1/3) on the c_1 axis so the joint
    # marginal back to c_0 equals the narrow cavity).
    log_cav_lifted = jnp.broadcast_to(log_cav_narrow[:, :, None], (horizon, 2, 3))
    log_cav_lifted = log_cav_lifted - jnp.log(3.0)

    out_narrow = compute_log_reduced(
        log_kernels_narrow, log_cav_narrow, n_theta_axes=1
    )
    out_lifted = compute_log_reduced(
        log_kernels_lifted, log_cav_lifted, n_theta_axes=2
    )

    np.testing.assert_allclose(
        np.asarray(out_narrow), np.asarray(out_lifted), atol=1e-5
    )


def test_compute_obs_to_theta_msgs_narrow_vs_lifted():
    """Per-channel messages: a θ-dep channel with theta_components=(k,)
    produces messages over component k only; the lifted version (channel
    sees both components) produces messages over the joint that, when
    marginalized to component k, match the narrow messages."""
    rng = np.random.default_rng(17)
    horizon, n_states, n_obs = 3, 4, 2
    theta_spec = ThetaSpec(component_sizes=(2, 3))
    factor_narrow = FactorSpec(theta_components=(0,))
    factor_full = FactorSpec(theta_components=(0, 1))

    # Narrow kernel: (T+1, n_obs, n_states, c_0)
    log_k_narrow = jnp.asarray(
        rng.standard_normal((horizon + 1, n_obs, n_states, 2))
    )
    # Lifted kernel: broadcast over c_1.
    log_k_lifted = jnp.broadcast_to(
        log_k_narrow[:, :, :, :, None],
        (horizon + 1, n_obs, n_states, 2, 3),
    )

    log_fwd = jnp.asarray(rng.standard_normal((horizon + 1, n_states)))
    log_bwd = jnp.asarray(rng.standard_normal((horizon + 1, n_states)))
    log_obs_to_x = jnp.asarray(rng.standard_normal((horizon + 1, n_states)))

    out_narrow = compute_obs_to_theta_msgs(
        [log_k_narrow], log_fwd, log_bwd, log_obs_to_x,
        b_factor_specs=(factor_narrow,),
    )
    out_lifted = compute_obs_to_theta_msgs(
        [log_k_lifted], log_fwd, log_bwd, log_obs_to_x,
        b_factor_specs=(factor_full,),
    )

    msg_narrow = out_narrow[0]   # (T+1, 2)
    msg_lifted = out_lifted[0]   # (T+1, 2, 3)
    # Marginalize the lifted message over component 1 to compare with narrow.
    # Since log_k was uniform-broadcast over c_1, the narrow result equals
    # logsumexp(lifted, axis=2) - log(3).
    msg_lifted_marginal = jnp.log(jnp.exp(msg_lifted).sum(axis=2))
    np.testing.assert_allclose(
        np.asarray(msg_narrow + jnp.log(3.0)),
        np.asarray(msg_lifted_marginal),
        atol=1e-5,
    )


def test_compute_theta_cavities_narrow_single_channel():
    """End-to-end smoke: one θ-dep channel with narrow factor, run
    compute_theta_cavities, verify shapes and normalization."""
    rng = np.random.default_rng(23)
    horizon = 3
    theta_spec = ThetaSpec(component_sizes=(2, 3))
    t_factor = FactorSpec(theta_components=(0, 1))   # T sees both
    b_factor = FactorSpec(theta_components=(0,))     # B sees only comp 0

    log_prior = (
        jnp.log(jnp.asarray([0.4, 0.6])),
        jnp.log(jnp.asarray([0.2, 0.5, 0.3])),
    )
    log_dyn = jnp.asarray(rng.standard_normal((horizon, 2, 3)))
    log_obs = jnp.asarray(rng.standard_normal((horizon + 1, 2)))

    log_cavity_dyn, log_cavity_obs = compute_theta_cavities(
        log_prior,
        log_dyn, t_factor,
        (log_obs,), (b_factor,),
    )

    # Dyn cavity shape and normalization.
    assert log_cavity_dyn.shape == (horizon, 2, 3)
    sums = jnp.exp(log_cavity_dyn).sum(axis=(1, 2))
    np.testing.assert_allclose(np.asarray(sums), np.ones(horizon), atol=1e-5)

    # Obs cavity (channel 0) shape and normalization.
    assert log_cavity_obs[0].shape == (horizon + 1, 2)
    sums = jnp.exp(log_cavity_obs[0]).sum(axis=1)
    np.testing.assert_allclose(np.asarray(sums), np.ones(horizon + 1), atol=1e-5)


def test_compute_theta_cavities_independent_components():
    """When a B-channel sees only component k, the per-component cavity for
    other components should equal their prior (no obs evidence contributes)."""
    rng = np.random.default_rng(29)
    horizon = 2
    theta_spec = ThetaSpec(component_sizes=(2, 3))
    t_factor = FactorSpec(theta_components=())           # T independent of θ
    b_factor = FactorSpec(theta_components=(0,))         # only sees component 0

    log_prior = (
        jnp.log(jnp.asarray([0.4, 0.6])),
        jnp.log(jnp.asarray([0.2, 0.5, 0.3])),
    )
    log_dyn = jnp.zeros((horizon,))    # T is θ-independent
    log_obs = jnp.asarray(rng.standard_normal((horizon + 1, 2)))

    log_cavity_dyn, log_cavity_obs = compute_theta_cavities(
        log_prior,
        log_dyn, t_factor,
        (log_obs,), (b_factor,),
    )

    # Obs cavity (channel 0) — at component 0 it should equal the per-component
    # total over component 0 minus this channel's marginalized contribution.
    # Total at component 0 = log_prior[0] + obs.sum(0). Cavity = total - obs[t].
    # So cavity[t] = log_prior[0] + obs.sum(0) - obs[t], normalized.
    obs_total_0 = log_obs.sum(axis=0)
    expected = log_prior[0][None, :] + obs_total_0[None, :] - log_obs
    expected = expected - jnp.log(jnp.exp(expected).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(
        np.asarray(log_cavity_obs[0]), np.asarray(expected), atol=1e-5
    )


def test_compute_obs_to_x_msgs_theta_indep_unchanged():
    """A θ-independent channel's contribution must equal logsumexp over obs
    (no cavity used). Check that adding a θ-indep channel leaves the result
    additively decomposable."""
    rng = np.random.default_rng(31)
    horizon, n_obs, n_states = 2, 3, 4
    f_indep = FactorSpec(theta_components=())
    log_k_indep = jnp.asarray(
        rng.standard_normal((horizon + 1, n_obs, n_states))
    )

    out = compute_obs_to_x_msgs(
        [log_k_indep],
        log_cavity_obs=(None,),
        b_factor_specs=(f_indep,),
    )
    expected_unnorm = jnp.log(jnp.exp(log_k_indep).sum(axis=1))   # (T+1, n_states)
    expected = expected_unnorm - jnp.log(jnp.exp(expected_unnorm).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(expected), atol=1e-5
    )
