"""Dense Bayesian state inference for the Nuijten-MP agent.

Joint predict-then-correct update over (dynamic state x, latent context theta),
copied from chapter-5's RockSample agent (`agents/rocksample_agent.py::_infer_state`)
and made standalone. Generic over (n_states, n_theta, n_actions, n_channels,
n_obs_types), so it serves both the T-Maze and the stochastic maze unchanged.
"""

import jax.numpy as jnp
from jax import nn

EPSILON = 1e-12


def infer_state(
    q_current: jnp.ndarray,
    q_static: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    observation_tensor: jnp.ndarray,
    obs: jnp.ndarray,
    action_onehot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bayesian state and static-state (theta) update.

    1. Predict x_new from x_old via the transition tensor.
    2. Incorporate the per-channel observation likelihood to update both x and theta.

    Args:
        q_current: (n_states,) prior belief over state
        q_static: (n_static,) belief over latent context theta
        transition_tensor: (n_states, n_states, n_static, n_actions)
        observation_tensor: (n_channels, n_obs_types, n_states, n_static)
        obs: (n_channels,) per-channel categorical outcome indices
        action_onehot: (n_actions,) one-hot previous action

    Returns:
        q_new_state: (n_states,) posterior over state
        q_new_static: (n_static,) posterior over theta
    """
    n_channels = observation_tensor.shape[0]

    # Predict: P(x_new) = Σ_{x_old, θ, a} T(x_new|x_old,θ,a) · q(x_old) · q(θ) · a(a)
    predicted = jnp.einsum(
        "ijkl,j,k,l->i",
        transition_tensor, q_current, q_static, action_onehot,
    )
    predicted = predicted / (predicted.sum() + EPSILON)

    # Observation log-likelihood: Σ_c log P(obs_c | x, θ)
    obs_int = jnp.round(obs).astype(jnp.int32)
    log_lik_per_channel = jnp.log(
        observation_tensor[jnp.arange(n_channels), obs_int, :, :] + EPSILON
    )
    log_likelihood = log_lik_per_channel.sum(axis=0)

    # Joint posterior: P(x, θ | obs) ∝ P(obs | x, θ) · P(x) · P(θ)
    log_joint = (
        log_likelihood
        + jnp.log(predicted + EPSILON)[:, None]
        + jnp.log(q_static + EPSILON)[None, :]
    )

    log_q_x = jnp.logaddexp.reduce(log_joint, axis=1)
    q_new_state = nn.softmax(log_q_x)

    log_q_theta = jnp.logaddexp.reduce(log_joint, axis=0)
    q_new_static = nn.softmax(log_q_theta)

    return q_new_state, q_new_static
