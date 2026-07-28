"""
Mean-field temporal factorization of variational free energy.

Posterior:
    q(x_{0:T}, u_{1:T}, y_{1:T}, θ) =
        q(θ) · ∏_t q(u_t) · q(x_t|x_{t-1},θ) · q(y_θ|x_t,θ) · q(y_loc|x_t)

Differs from `temporal_vfe.py` by treating actions as independent of state
and θ in the variational posterior, and dropping the action argument from
the variational transition. This is a strict mean-field assumption on u.

The four epistemic priors are retained. The control prior is recomputed
using the *generative* transition p(x_t|x_{t-1},u_t), since q(x_t|x_{t-1},
u_t,θ) no longer exists. With q(x_{t-1}|u_t) = q(x_{t-1}) under the
mean-field assumption, the Bethe difference
    H[q(x_t,x_{t-1}|u_t)] − H[q(x_{t-1}|u_t)]
collapses to E_{q(x_{t-1})}[ H[p(·|x_{t-1},u)] ].

Inference mode is fixed to "active" — no marginal/planning variants.
"""

from typing import Dict
import jax.numpy as jnp
from jax import Array
from jax.nn import softmax

EPS = 1e-8


def _compute_forward_marginals_mf(
    q_theta: Array,                  # (n_theta,)
    q_x_given_x_theta: Array,        # (horizon, n_states_next, n_states_prev, n_theta)
    initial_state: Array,            # (n_states,)
    horizon: int,
) -> Array:
    """
    Forward pass: q(x_t, θ) under the mean-field posterior.

    q(x_0, θ) = p(x_0) · q(θ)
    q(x_t, θ) = Σ_{x_{t-1}} q(x_t|x_{t-1}, θ) · q(x_{t-1}, θ)

    No action enters the message because q(u_t) is independent.
    """
    n_states = q_x_given_x_theta.shape[1]
    n_theta = q_theta.shape[0]

    q_theta = jnp.clip(q_theta, EPS, 1.0)
    q_theta = q_theta / jnp.sum(q_theta)

    initial_state = jnp.clip(initial_state, EPS, 1.0)
    initial_state = initial_state / jnp.sum(initial_state)

    q_x_theta = jnp.zeros((horizon + 1, n_states, n_theta))
    q_x_theta = q_x_theta.at[0].set(initial_state[:, None] * q_theta[None, :])

    for t in range(horizon):
        var_trans_t = q_x_given_x_theta[t]  # (n_states_next, n_states_prev, n_theta)
        q_x_prev_theta = q_x_theta[t]       # (n_states_prev, n_theta)

        q_x_next_theta = jnp.einsum('xpt,pt->xt', var_trans_t, q_x_prev_theta)

        q_x_next_theta = jnp.clip(q_x_next_theta, EPS, None)
        normalizer = jnp.sum(q_x_next_theta, axis=0, keepdims=True)
        q_x_next_theta = q_x_next_theta / jnp.maximum(normalizer, EPS)

        q_x_theta = q_x_theta.at[t + 1].set(q_x_next_theta)

    return q_x_theta


def _compute_entropy_terms_mf(
    q_theta: Array,                  # (n_theta,)
    q_u: Array,                      # (horizon, n_actions)
    q_x_given_x_theta: Array,        # (horizon, n_states_next, n_states_prev, n_theta)
    q_y_theta_given_x_theta: Array,  # (horizon, n_obs, n_states, n_theta)
    q_y_location_given_x: Array,     # (horizon, n_locations, n_states)
    q_x_theta: Array,                # (horizon+1, n_states, n_theta)
    horizon: int,
) -> Array:
    """
    Total entropy under the mean-field posterior.

    H[q] = H[q(θ)] + Σ_t H[q(u_t)]
         + Σ_t E_{x_{t-1},θ}[H[q(x_t|x_{t-1},θ)]]
         + Σ_t E_{x_t,θ}[H[q(y_θ|x_t,θ)]]
         + Σ_t E_{x_t}[H[q(y_loc|x_t)]]
    """
    total_entropy = 0.0

    # H[q(θ)]
    q_theta_safe = jnp.clip(q_theta, EPS, 1.0)
    total_entropy += -jnp.sum(q_theta_safe * jnp.log(q_theta_safe))

    for t in range(horizon):
        # H[q(u_t)] — marginal entropy, no expectation needed
        q_u_t_safe = jnp.clip(q_u[t], EPS, 1.0)
        total_entropy += -jnp.sum(q_u_t_safe * jnp.log(q_u_t_safe))

        # E_{x_{t-1},θ}[H[q(x_t|x_{t-1},θ)]]
        var_trans_t = q_x_given_x_theta[t]  # (x_next, x_prev, θ)
        var_trans_safe = jnp.clip(var_trans_t, EPS, 1.0)
        h_x_given_x_theta = -jnp.sum(
            var_trans_safe * jnp.log(var_trans_safe),
            axis=0,  # sum over x_next
        )  # (x_prev, θ)
        q_x_prev_theta = q_x_theta[t]  # (x_prev, θ)
        total_entropy += jnp.sum(q_x_prev_theta * h_x_given_x_theta)

        # E_{x_t,θ}[H[q(y_θ|x_t,θ)]]
        var_obs_t = q_y_theta_given_x_theta[t]  # (n_obs, n_states, n_theta)
        var_obs_safe = jnp.clip(var_obs_t, EPS, 1.0)
        h_y_given_x_theta = -jnp.sum(
            var_obs_safe * jnp.log(var_obs_safe),
            axis=0,
        )  # (n_states, n_theta)
        q_x_next_theta = q_x_theta[t + 1]
        total_entropy += jnp.sum(q_x_next_theta * h_y_given_x_theta)

        # E_{x_t}[H[q(y_loc|x_t)]]
        q_loc_t = q_y_location_given_x[t]  # (n_locations, n_states)
        q_loc_safe = jnp.clip(q_loc_t, EPS, 1.0)
        h_loc_given_x = -jnp.sum(
            q_loc_safe * jnp.log(q_loc_safe),
            axis=0,
        )  # (n_states,)
        q_x_t = jnp.sum(q_x_theta[t + 1], axis=1)
        total_entropy += jnp.sum(q_x_t * h_loc_given_x)

    return total_entropy


def _compute_energy_terms_mf(
    q_theta: Array,                  # (n_theta,)
    q_u: Array,                      # (horizon, n_actions)
    q_x_given_x_theta: Array,        # (horizon, n_states_next, n_states_prev, n_theta)
    q_y_theta_given_x_theta: Array,  # (horizon, n_obs, n_states, n_theta)
    q_y_location_given_x: Array,     # (horizon, n_locations, n_states)
    q_x_theta: Array,                # (horizon+1, n_states, n_theta)
    transition_tensor: Array,        # (n_states, n_states, n_actions)
    theta_observation_tensor: Array, # (n_obs, n_states, n_theta)
    location_observation_tensor: Array,  # (n_locations, n_states)
    goal_mapping: Array,             # (n_states, n_theta)
    action_prior: Array,             # (n_actions,)
    theta_prior: Array,              # (n_theta,)
    horizon: int,
) -> Dict[str, Array]:
    """
    Energy terms E_q[-log p(·)] under the mean-field posterior.

    The transition energy uses the factorization
        q(x_t, x_{t-1}, u_t, θ) = q(u_t) · q(x_t|x_{t-1},θ) · q(x_{t-1}, θ)
    against the generative log p(x_t|x_{t-1}, u_t).
    """
    energies = {}

    # 1. Parameter prior energy
    theta_prior_safe = jnp.clip(theta_prior, EPS, 1.0)
    energies['theta'] = -jnp.sum(q_theta * jnp.log(theta_prior_safe))

    action_prior_safe = jnp.clip(action_prior, EPS, 1.0)
    log_action_prior = jnp.log(action_prior_safe)
    log_trans = jnp.log(jnp.clip(transition_tensor, EPS, 1.0))   # (x_n, x_p, u)
    log_obs = jnp.log(jnp.clip(theta_observation_tensor, EPS, 1.0))  # (y, x, θ)
    log_loc = jnp.log(jnp.clip(location_observation_tensor, EPS, 1.0))  # (y_loc, x)

    action_energy = 0.0
    transition_energy = 0.0
    observation_energy = 0.0
    location_energy = 0.0

    for t in range(horizon):
        # 2a. Action prior energy: E_q[-log p(u_t)]
        q_u_t = q_u[t]  # (n_actions,)
        action_energy += -jnp.sum(q_u_t * log_action_prior)

        # 2b. Transition energy: E_q[-log p(x_t|x_{t-1}, u_t)]
        # q(x_t, x_{t-1}, θ) = q(x_t|x_{t-1}, θ) · q(x_{t-1}, θ)
        var_trans_t = q_x_given_x_theta[t]  # (x_n, x_p, θ)
        q_x_prev_theta = q_x_theta[t]       # (x_p, θ)
        q_xx_theta = var_trans_t * q_x_prev_theta[None, :, :]  # (x_n, x_p, θ)

        if transition_tensor.ndim == 3:
            # p(x_t|x_{t-1}, u_t) doesn't depend on θ — sum θ first
            q_xx = jnp.sum(q_xx_theta, axis=2)  # (x_n, x_p)
            # q(x_t, x_{t-1}, u_t) = q(u_t) · q(x_t, x_{t-1})
            transition_energy += -jnp.einsum('xp,u,xpu->', q_xx, q_u_t, log_trans)
        else:
            # θ-dependent transitions: log_trans has shape (x_n, x_p, u, θ)
            transition_energy += -jnp.einsum('xpt,u,xput->', q_xx_theta, q_u_t, log_trans)

        # 2c. Observation energy: E_q[-log p(y_θ|x_t, θ)]
        var_obs_t = q_y_theta_given_x_theta[t]  # (n_obs, n_states, n_theta)
        q_x_next_theta = q_x_theta[t + 1]       # (n_states, n_theta)
        q_yx_theta = var_obs_t * q_x_next_theta[None, :, :]
        observation_energy += -jnp.sum(q_yx_theta * log_obs)

        # 2d. Location observation energy: E_q[-log p(y_loc|x_t)]
        q_loc_t = q_y_location_given_x[t]  # (n_loc, n_states)
        q_x_t = jnp.sum(q_x_theta[t + 1], axis=1)  # (n_states,)
        q_loc_x = q_loc_t * q_x_t[None, :]
        location_energy += -jnp.sum(q_loc_x * log_loc)

    energies['action'] = action_energy
    energies['transition'] = transition_energy
    energies['observation'] = observation_energy
    energies['location_observation'] = location_energy

    # 3. Goal energy
    goal_safe = jnp.clip(goal_mapping, EPS, 1.0)
    log_goal = jnp.log(goal_safe)
    q_x_final_theta = q_x_theta[horizon]
    energies['goal'] = -jnp.sum(q_x_final_theta * log_goal)

    return energies


def _compute_epistemic_priors_mf(
    q_theta: Array,                  # (n_theta,)
    q_u: Array,                      # (horizon, n_actions)
    q_x_theta: Array,                # (horizon+1, n_states, n_theta)
    q_y_theta_given_x_theta: Array,  # (horizon, n_obs, n_states, n_theta)
    q_y_location_given_x: Array,     # (horizon, n_locations, n_states)
    transition_tensor: Array,        # (n_states, n_states, n_actions)
    horizon: int,
) -> Array:
    """
    Four epistemic priors under the mean-field posterior.

    1. Control prior (generative-transition Bethe):
        ũ(u_t) ∝ exp( E_{q(x_{t-1})}[ H[p(·|x_{t-1}, u)] ] )
       Energy: E_{q(u_t)}[-log ũ(u_t)].

    2. State prior (theta): ũ(x_t) ∝ exp(-E_{q(θ|x_t)}[H[q(y_θ|x_t,θ)]])
    3. State prior (location): ũ(x_t) ∝ exp(-H[q(y_loc|x_t)])
    4. Observation prior (theta): ũ(y, x) ∝ exp(KL[q(θ|y,x) || q(θ)])

    Priors 2–4 are identical to the fully-coupled implementation because
    they don't reference q(u) or q(x|x,u).
    """
    total = 0.0

    log_trans = jnp.log(jnp.clip(transition_tensor, EPS, 1.0))  # (x_n, x_p, u)
    p_log_p = transition_tensor * log_trans                      # (x_n, x_p, u)
    h_p_given_xp_u = -jnp.sum(p_log_p, axis=0)                  # (x_p, u)

    for t in range(horizon):
        # ===== 1. Control prior =====
        # q(x_{t-1}) — current marginal
        q_x_prev = jnp.sum(q_x_theta[t], axis=1)  # (n_states,)

        # E_{q(x_{t-1})}[ H[p(·|x_{t-1}, u)] ]: (n_actions,)
        bethe_delta_h = jnp.einsum('p,pu->u', q_x_prev, h_p_given_xp_u)

        # Unnormalized exp prior (matches temporal_vfe): ũ(u) ∝ exp(ΔH)
        control_prior = jnp.exp(bethe_delta_h)
        log_control_prior = jnp.log(control_prior + EPS)
        total += -jnp.sum(q_u[t] * log_control_prior)

        # ===== 2. State prior (theta) =====
        q_y_theta_t = q_y_theta_given_x_theta[t]  # (n_obs, n_states, n_theta)
        q_y_safe = jnp.clip(q_y_theta_t, EPS, 1.0)
        h_y_given_x_theta = -jnp.sum(q_y_safe * jnp.log(q_y_safe), axis=0)  # (x, θ)

        q_x_theta_t = q_x_theta[t + 1]                         # (x, θ)
        q_x_t = jnp.sum(q_x_theta_t, axis=1)                   # (x,)
        q_theta_given_x = q_x_theta_t / jnp.clip(q_x_t[:, None], EPS, None)

        expected_cond_h = jnp.sum(q_theta_given_x * h_y_given_x_theta, axis=1)  # (x,)
        # Unnormalized exp prior (matches temporal_vfe): ũ(x) ∝ exp(-E_θ[H[q(y|x,θ)]])
        state_theta_prior = jnp.exp(-expected_cond_h)
        log_state_theta_prior = jnp.log(state_theta_prior + EPS)
        total += -jnp.sum(q_x_t * log_state_theta_prior)

        # ===== 3. State prior (location) =====
        q_loc_t = q_y_location_given_x[t]                      # (n_loc, n_states)
        q_loc_safe = jnp.clip(q_loc_t, EPS, 1.0)
        h_loc_given_x = -jnp.sum(q_loc_safe * jnp.log(q_loc_safe), axis=0)  # (x,)
        # Unnormalized exp prior (matches temporal_vfe): ũ(x) ∝ exp(-H[q(y_loc|x)])
        loc_info_prior = jnp.exp(-h_loc_given_x)
        log_loc_info_prior = jnp.log(loc_info_prior + EPS)
        total += -jnp.sum(q_x_t * log_loc_info_prior)

        # ===== 4. Observation prior (theta) =====
        # q(θ | y, x) ∝ q(y|x,θ) q(θ)
        q_theta_yx_unnorm = q_y_theta_t * q_theta[None, None, :]
        q_theta_yx_norm = jnp.sum(q_theta_yx_unnorm, axis=2, keepdims=True)
        q_theta_given_yx = q_theta_yx_unnorm / (q_theta_yx_norm + EPS)

        log_ratio = jnp.log(q_theta_given_yx + EPS) - jnp.log(q_theta[None, None, :] + EPS)
        kl_yx = jnp.sum(q_theta_given_yx * log_ratio, axis=2)  # (n_obs, n_states)

        # Unnormalized exp prior (matches temporal_vfe): ũ(y,x) ∝ exp(KL[q(θ|y,x)||q(θ)])
        yx_prior = jnp.exp(kl_yx)
        log_yx_prior = jnp.log(yx_prior + EPS)

        # q(y, x) = Σ_θ q(y|x,θ) q(x,θ)
        q_yx = jnp.sum(q_y_theta_t * q_x_theta_t[None, :, :], axis=2)
        total += -jnp.sum(q_yx * log_yx_prior)

    return total


def temporal_vfe_meanfield(
    q_theta_logits: Array,                   # (n_theta,)
    q_u_logits: Array,                       # (horizon, n_actions)
    q_x_given_x_theta_logits: Array,         # (horizon, n_states_next, n_states_prev, n_theta)
    q_y_theta_given_x_theta_logits: Array,   # (horizon, n_obs_theta, n_states, n_theta)
    q_y_location_given_x_logits: Array,      # (horizon, n_locations, n_states)
    initial_state: Array,                    # (n_states,)
    transition_tensor: Array,                # (n_states, n_states, n_actions)
    theta_observation_tensor: Array,         # (n_obs_theta, n_states, n_theta)
    location_observation_tensor: Array,      # (n_locations, n_states)
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    horizon: int,
) -> Array:
    """
    Mean-field temporal VFE with active-inference epistemic priors.

    VFE = -H[q] + E_q[-log p(...)] + epistemic_energy
    """
    q_theta = softmax(q_theta_logits)
    q_u = softmax(q_u_logits, axis=-1)                                      # over actions
    q_x_given_x_theta = softmax(q_x_given_x_theta_logits, axis=1)           # over x_next
    q_y_theta_given_x_theta = softmax(q_y_theta_given_x_theta_logits, axis=1)  # over y_theta
    q_y_location_given_x = softmax(q_y_location_given_x_logits, axis=1)        # over y_loc

    q_x_theta = _compute_forward_marginals_mf(
        q_theta=q_theta,
        q_x_given_x_theta=q_x_given_x_theta,
        initial_state=initial_state,
        horizon=horizon,
    )

    entropy = _compute_entropy_terms_mf(
        q_theta=q_theta,
        q_u=q_u,
        q_x_given_x_theta=q_x_given_x_theta,
        q_y_theta_given_x_theta=q_y_theta_given_x_theta,
        q_y_location_given_x=q_y_location_given_x,
        q_x_theta=q_x_theta,
        horizon=horizon,
    )

    energies = _compute_energy_terms_mf(
        q_theta=q_theta,
        q_u=q_u,
        q_x_given_x_theta=q_x_given_x_theta,
        q_y_theta_given_x_theta=q_y_theta_given_x_theta,
        q_y_location_given_x=q_y_location_given_x,
        q_x_theta=q_x_theta,
        transition_tensor=transition_tensor,
        theta_observation_tensor=theta_observation_tensor,
        location_observation_tensor=location_observation_tensor,
        goal_mapping=goal_mapping,
        action_prior=action_prior,
        theta_prior=theta_prior,
        horizon=horizon,
    )

    epistemic_energy = _compute_epistemic_priors_mf(
        q_theta=q_theta,
        q_u=q_u,
        q_x_theta=q_x_theta,
        q_y_theta_given_x_theta=q_y_theta_given_x_theta,
        q_y_location_given_x=q_y_location_given_x,
        transition_tensor=transition_tensor,
        horizon=horizon,
    )

    vfe = (
        -entropy
        + energies['action']
        + energies['transition']
        + energies['observation']
        + energies['location_observation']
        + energies['goal']
        + energies['theta']
        + epistemic_energy
    )
    return vfe


def extract_marginals_meanfield(
    q_theta_logits: Array,
    q_u_logits: Array,
    q_x_given_x_theta_logits: Array,
    q_y_theta_given_x_theta_logits: Array,
    q_y_location_given_x_logits: Array,
    initial_state: Array,
    horizon: int,
) -> Dict[str, Array]:
    """Extract all marginal distributions from the mean-field posterior."""
    q_theta = softmax(q_theta_logits)
    q_u = softmax(q_u_logits, axis=-1)
    q_x_given_x_theta = softmax(q_x_given_x_theta_logits, axis=1)
    q_y_theta_given_x_theta = softmax(q_y_theta_given_x_theta_logits, axis=1)
    q_y_location_given_x = softmax(q_y_location_given_x_logits, axis=1)

    q_x_theta = _compute_forward_marginals_mf(
        q_theta=q_theta,
        q_x_given_x_theta=q_x_given_x_theta,
        initial_state=initial_state,
        horizon=horizon,
    )

    return {
        'q_theta': q_theta,
        'q_u': q_u,
        'q_x_given_x_theta': q_x_given_x_theta,
        'q_y_theta_given_x_theta': q_y_theta_given_x_theta,
        'q_y_location_given_x': q_y_location_given_x,
        'q_x_theta': q_x_theta,
    }
