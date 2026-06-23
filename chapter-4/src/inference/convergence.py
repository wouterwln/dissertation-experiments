"""Convergence variant of the Nuijten-MP planner that returns a per-iteration VFE trace.

`nuijten_mp_convergence` mirrors `nuijten_mp.nuijten_mp_planning` but additionally
accumulates the Bethe free energy (region-based, with EFE prior factors) at each
message-passing iteration. Used to verify that the Nuijten-MP scheme converges.

For pedagogical transparency the EFE drives are booked as explicit prior-factor
energies: the action prior `pi(u)` (singleton factor on u), and the observation
ambiguity prior on the joint `(x, theta)` observation region. The obs->x and
obs->theta EFE messages are the two mirrored cavity-projections of that single
joint factor, so the ambiguity is counted once (not once per leg). See
`compute_nuijten_vfe`.

This is a trimmed copy of chapter-5's `inference/convergence.py` containing only the
Nuijten path (the only method chapter-4 uses). All computation is in log-space.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from .messages import LOG_ZERO, EPSILON, safe_log, safe_log_div
from .region_extended_loopy_bp import (
    compute_log_reduced,
    compute_pref_to_x_msgs,
    compute_pref_to_theta_msgs,
    compute_theta_cavities_extended,
)
from .nuijten_mp import (
    compute_obs_region_beliefs_original,
    compute_efe_action_prior,
    compute_obs_efe_to_x,
    compute_obs_efe_to_theta,
    forward_pass_nuijten,
    backward_pass_nuijten,
    compute_dyn_to_theta_msgs_nuijten,
    compute_dyn_region_beliefs_nuijten,
)


# =============================================================================
# VFE helpers
# =============================================================================


def _entropy(log_q):
    """Shannon entropy H[q] = -sum q log q from normalized log-probs."""
    q = jnp.exp(log_q)
    return -jnp.sum(jnp.where(q > 1e-30, q * log_q, 0.0))


def _entropy_unnorm(log_b):
    """Shannon entropy from unnormalized log beliefs (normalizes over all dims first)."""
    flat = log_b.reshape(-1)
    log_q = flat - logsumexp(flat)
    q = jnp.exp(log_q)
    return -jnp.sum(jnp.where(q > 1e-30, q * log_q, 0.0))


def _energy(log_region, log_factor):
    """Average energy U = -sum b(x) log f(x) from unnormalized log region beliefs."""
    flat_region = log_region.reshape(-1)
    log_q = flat_region - logsumexp(flat_region)
    q = jnp.exp(log_q)
    flat_factor = log_factor.reshape(-1)
    valid = flat_factor > LOG_ZERO / 2
    return -jnp.sum(jnp.where(valid, q * flat_factor, 0.0))


# =============================================================================
# Nuijten VFE (region-extended free energy with EFE action-prior factors)
# =============================================================================


def compute_nuijten_vfe(log_dyn_regions, obs_regions,
                        log_fwd_msgs, log_bwd_msgs, q_u,
                        log_T_kernel_tiled, log_B_flat,
                        log_cavity_dyn, log_cavity_obs,
                        log_prior_theta, action_prior_per_t):
    """VFE for Nuijten methods — region-extended with additional EFE prior factors.

    Two EFE drives are reified as explicit prior factors and booked as energies, each
    a cross-entropy ``-<log factor>`` under the relevant region belief:

    - ``U_prior`` — EFE action prior ``pi_t(u)`` (singleton factor on the action variable).
    - ``U_epi``   — observation ambiguity prior on the *joint* ``(x, theta)`` observation
      region: ``psi_t(x, theta) ~ exp(-sum_k H_k(x, theta))``, with
      ``H_k(x, theta) = H[p_k(y | x, theta)]`` the per-cell likelihood entropy.

    NOTE: ``U_epi`` is counted ONCE on the joint ``(x, theta)`` edge. The ``obs->x`` and
    ``obs->theta`` EFE messages are the two mirrored cavity-projections of this single
    factor (same ``H_k(x, theta)`` field, conditioned on theta vs. on x); booking both as
    separate singleton energies would double-count the same ambiguity. ``U_epi`` equals
    the canonical EFE ambiguity ``E_{q(x,theta)}[H[p(y|x,theta)]]`` and is belief-weighted
    (tracks ``q(x, theta)``), so it is NOT a constant offset.
    """
    horizon = q_u.shape[0]
    n_fov = obs_regions.shape[1]

    # obs_regions arrive in probability space (compute_obs_region_beliefs_original).
    log_obs_regions = safe_log(obs_regions)

    # --- Energy terms ---
    def dyn_energy_t(t):
        return _energy(log_dyn_regions[t], log_T_kernel_tiled[t])

    U_dyn = jax.vmap(dyn_energy_t)(jnp.arange(horizon)).sum()

    def obs_energy_t(t):
        def obs_energy_k(k):
            return _energy(log_obs_regions[t, k], log_B_flat[k])
        return jax.vmap(obs_energy_k)(jnp.arange(n_fov)).sum()

    U_obs = jax.vmap(obs_energy_t)(jnp.arange(horizon + 1)).sum()

    def prior_u_energy(t):
        log_pi = safe_log(action_prior_per_t[t])
        log_q_u = safe_log(q_u[t])
        q_ut = jnp.exp(log_q_u)
        return -jnp.sum(q_ut * log_pi)

    U_prior = jax.vmap(prior_u_energy)(jnp.arange(horizon)).sum()

    # --- Singleton entropies ---
    def x_entropy(t):
        log_q_x = log_fwd_msgs[t] + log_bwd_msgs[t]
        log_q_x = log_q_x - logsumexp(log_q_x)
        return _entropy(log_q_x)

    H_x = jax.vmap(x_entropy)(jnp.arange(horizon + 1)).sum()

    def u_entropy(t):
        return _entropy(safe_log(q_u[t]))

    H_u = jax.vmap(u_entropy)(jnp.arange(horizon)).sum()

    all_cavities = jnp.concatenate([log_cavity_dyn, log_cavity_obs], axis=0)
    log_q_theta = all_cavities.mean(axis=0)
    log_q_theta = log_q_theta - logsumexp(log_q_theta)
    H_theta = _entropy(log_q_theta)

    # --- Epistemic prior factor energy (obs ambiguity on the joint (x,θ) region) ---
    # Single joint factor ψ_t(x,θ) ∝ exp(-Σ_k H_k(x,θ)); counted once (NOT split into
    # obs→x and obs→θ, which are the two mirrored projections of this same field).
    # H_k(x,θ) = -Σ_y p_k(y|x,θ) log p_k(y|x,θ) — fixed property of the likelihood tensor.
    p_B = jnp.exp(log_B_flat)
    plogp = jnp.where(log_B_flat > LOG_ZERO / 2, p_B * log_B_flat, 0.0)
    H_obs = -plogp.sum(axis=1)  # (n_fov, n_states, n_static)

    def epi_energy_t(t):
        def epi_energy_k(k):
            # q(x,θ | k, t): marginalize the obs region belief over y, renormalize.
            b_xtheta = obs_regions[t, k].sum(axis=0)
            b_xtheta = b_xtheta / (b_xtheta.sum() + EPSILON)
            return jnp.sum(b_xtheta * H_obs[k])
        return jax.vmap(epi_energy_k)(jnp.arange(n_fov)).sum()

    U_epi = jax.vmap(epi_energy_t)(jnp.arange(horizon + 1)).sum()

    # --- Intersection terms ---
    def dyn_intersection_t(t):
        log_joint_xu = logsumexp(log_dyn_regions[t], axis=(1, 2))
        H_xu = _entropy_unnorm(log_joint_xu)
        log_joint_xxu = logsumexp(log_dyn_regions[t], axis=2)
        H_xxu = _entropy_unnorm(log_joint_xxu)
        return H_xu - H_xxu

    dyn_intersection = jax.vmap(dyn_intersection_t)(jnp.arange(horizon)).sum()

    def obs_intersection_t(t):
        def obs_intersection_k(k):
            H_yxtheta = _entropy_unnorm(log_obs_regions[t, k])
            log_xtheta = logsumexp(log_obs_regions[t, k], axis=0)
            H_xtheta = _entropy_unnorm(log_xtheta)
            return H_yxtheta - H_xtheta
        return jax.vmap(obs_intersection_k)(jnp.arange(n_fov)).sum()

    obs_intersection = jax.vmap(obs_intersection_t)(jnp.arange(horizon + 1)).sum()

    return (U_dyn + U_obs + U_prior + U_epi) - (H_x + H_u + H_theta) + dyn_intersection + obs_intersection


# =============================================================================
# Nuijten MP convergence (theta inferred) with per-iteration VFE trace
# =============================================================================


@partial(jax.jit, static_argnums=(5, 6))
def nuijten_mp_convergence(
    q_current_state,
    q_static_state,
    transition_tensor,
    observation_tensor,
    goal,
    horizon,
    n_iterations,
    action_prior=None,
):
    """Nuijten MP planning (theta inferred) with per-iteration VFE trace.

    Returns:
        action_dist: (n_actions,)
        log_dyn_region_beliefs: (T, x_old, x_new, theta, u)
        obs_region_beliefs: (T+1, n_channels, n_obs_types, n_states, n_static)
        vfe_trace: (n_iterations,)
    """
    n_states = q_current_state.shape[0]
    n_static = q_static_state.shape[0]
    n_actions = transition_tensor.shape[3]
    n_fov = observation_tensor.shape[0]
    n_obs_types = observation_tensor.shape[1]
    has_pref = goal.ndim == 2

    if action_prior is None:
        action_prior = jnp.ones(n_actions) / n_actions
    action_mask = (action_prior > 0).astype(jnp.float32)

    log_T = safe_log(transition_tensor)
    log_T_kernel = log_T.transpose(1, 0, 2, 3)
    log_T_kernel_tiled = jnp.broadcast_to(
        log_T_kernel[None], (horizon, n_states, n_states, n_static, n_actions)
    )
    log_B_flat = safe_log(observation_tensor)
    log_q0 = safe_log(q_current_state)
    log_prior_theta = safe_log(q_static_state)

    if has_pref:
        log_C = safe_log(goal)                            # (n_states, n_static)
        log_goal = jnp.zeros(n_states)                    # uniform terminal
    else:
        log_C = None
        log_goal = safe_log(goal)

    log_dyn_to_theta = jnp.zeros((horizon, n_static))
    q_u_init = jnp.zeros((horizon, n_actions))
    action_prior_init = jnp.tile(action_prior, (horizon, 1))
    log_dyn_regions_init = jnp.zeros((horizon, n_states, n_states, n_static, n_actions))
    obs_regions_init = jnp.zeros((horizon + 1, n_fov, n_obs_types, n_states, n_static))
    vfe_trace = jnp.zeros(n_iterations)

    if has_pref:
        log_pref_to_theta_init = jnp.zeros((horizon + 1, n_static))

        def body_fn(i, carry):
            log_dyn_to_theta, _, action_prior_per_t, _, obs_regions, log_pref_to_theta, vfe_trace = carry

            log_action_prior_per_t = safe_log(action_prior_per_t)

            log_obs_to_x = compute_obs_efe_to_x(obs_regions)
            log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

            log_cavity_dyn, log_cavity_obs, log_cavity_pref = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta, log_pref_to_theta
            )

            log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

            log_pref_to_x = compute_pref_to_x_msgs(log_C, log_cavity_pref)
            log_local_to_x = log_obs_to_x + log_pref_to_x

            log_fwd_msgs = forward_pass_nuijten(
                log_reduced_per_t, log_q0, log_action_prior_per_t, log_local_to_x, horizon
            )
            log_bwd_msgs, q_u = backward_pass_nuijten(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
                log_local_to_x, horizon
            )

            new_log_dyn_to_theta = compute_dyn_to_theta_msgs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_action_prior_per_t, horizon
            )

            new_log_pref_to_theta = compute_pref_to_theta_msgs(
                log_C, log_fwd_msgs, log_bwd_msgs, log_obs_to_x
            )

            log_dyn_regions = compute_dyn_region_beliefs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_local_to_x,
                log_cavity_dyn, log_action_prior_per_t
            )
            obs_regions = compute_obs_region_beliefs_original(
                log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
            )

            new_action_prior = compute_efe_action_prior(log_dyn_regions, action_mask)

            vfe = compute_nuijten_vfe(
                log_dyn_regions, obs_regions,
                log_fwd_msgs, log_bwd_msgs, q_u,
                log_T_kernel_tiled, log_B_flat,
                log_cavity_dyn, log_cavity_obs,
                log_prior_theta, action_prior_per_t,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return (new_log_dyn_to_theta, q_u, new_action_prior, log_dyn_regions,
                    obs_regions, new_log_pref_to_theta, vfe_trace)

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init,
             log_dyn_regions_init, obs_regions_init, log_pref_to_theta_init, vfe_trace)
        )
        _, q_u, _, log_dyn_region_beliefs, obs_region_beliefs, _, vfe_trace = result
    else:
        def body_fn(i, carry):
            log_dyn_to_theta, _, action_prior_per_t, _, obs_regions, vfe_trace = carry

            log_action_prior_per_t = safe_log(action_prior_per_t)

            log_obs_to_x = compute_obs_efe_to_x(obs_regions)
            log_obs_to_theta = compute_obs_efe_to_theta(obs_regions)

            log_cavity_dyn, log_cavity_obs = compute_theta_cavities_extended(
                log_prior_theta, log_dyn_to_theta, log_obs_to_theta
            )

            log_reduced_per_t = compute_log_reduced(log_T_kernel_tiled, log_cavity_dyn)

            log_fwd_msgs = forward_pass_nuijten(
                log_reduced_per_t, log_q0, log_action_prior_per_t, log_obs_to_x, horizon
            )
            log_bwd_msgs, q_u = backward_pass_nuijten(
                log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior_per_t,
                log_obs_to_x, horizon
            )

            new_log_dyn_to_theta = compute_dyn_to_theta_msgs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_action_prior_per_t, horizon
            )

            log_dyn_regions = compute_dyn_region_beliefs_nuijten(
                log_T_kernel_tiled, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
                log_cavity_dyn, log_action_prior_per_t
            )
            obs_regions = compute_obs_region_beliefs_original(
                log_B_flat, log_fwd_msgs, log_bwd_msgs, log_cavity_obs
            )

            new_action_prior = compute_efe_action_prior(log_dyn_regions, action_mask)

            vfe = compute_nuijten_vfe(
                log_dyn_regions, obs_regions,
                log_fwd_msgs, log_bwd_msgs, q_u,
                log_T_kernel_tiled, log_B_flat,
                log_cavity_dyn, log_cavity_obs,
                log_prior_theta, action_prior_per_t,
            )
            vfe_trace = vfe_trace.at[i].set(vfe)

            return new_log_dyn_to_theta, q_u, new_action_prior, log_dyn_regions, obs_regions, vfe_trace

        result = lax.fori_loop(
            0, n_iterations, body_fn,
            (log_dyn_to_theta, q_u_init, action_prior_init,
             log_dyn_regions_init, obs_regions_init, vfe_trace)
        )
        _, q_u, _, log_dyn_region_beliefs, obs_region_beliefs, vfe_trace = result

    return q_u[0], log_dyn_region_beliefs, obs_region_beliefs, vfe_trace
