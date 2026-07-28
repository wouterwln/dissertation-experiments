"""
Adam-based optimizer for the mean-field temporal VFE planner.

Mirrors `temporal_optimizer.py` but uses the simpler mean-field posterior
in which q(u) is independent and q(x|x,θ) drops the action argument.
"""

from dataclasses import dataclass
from typing import List, Optional
import jax
import jax.numpy as jnp
from jax import Array
import optax

from ..objectives.temporal_vfe_meanfield import (
    temporal_vfe_meanfield,
    extract_marginals_meanfield,
)


@dataclass
class MeanfieldPlanningConfig:
    """Configuration for mean-field temporal planning."""
    planning_horizon: int = 7
    n_states: int = 32
    n_actions: int = 4
    n_theta: int = 2
    n_obs: int = 3
    n_optimization_steps: int = 2000
    learning_rate: float = 0.01
    init_seed: int = 42


@dataclass
class MeanfieldPlanningResult:
    """Result of mean-field temporal planning with full variational distributions."""
    q_theta: Array                          # (n_theta,)
    q_u: Array                              # (horizon, n_actions)
    q_x_given_x_theta: Array                # (horizon, n_states, n_states, n_theta)
    q_y_theta_given_x_theta: Array          # (horizon, n_obs_theta, n_states, n_theta)
    q_y_location_given_x: Array             # (horizon, n_locations, n_states)
    q_first_action: Array                   # (n_actions,) — equals q_u[0]
    final_loss: float
    loss_history: List[float]
    all_losses: List[float]
    q_x_theta: Array                        # (horizon+1, n_states, n_theta)


def plan_actions_meanfield(
    initial_state: Array,                    # (n_states,)
    transition_tensor: Array,                # (n_states, n_states, n_actions)
    theta_observation_tensor: Array,         # (n_obs_theta, n_states, n_theta)
    goal_mapping: Array,                     # (n_states, n_theta)
    action_prior: Array,                     # (n_actions,)
    theta_prior: Array,                      # (n_theta,)
    config: MeanfieldPlanningConfig,
    prior_theta_logits: Optional[Array] = None,
    location_observation_tensor: Array = None,  # (n_locations, n_states)
) -> MeanfieldPlanningResult:
    """
    Plan actions by minimizing the mean-field temporal VFE.

    Inference mode is fixed to active — the four epistemic priors are always
    included.
    """
    key = jax.random.PRNGKey(config.init_seed)

    log_transition = jnp.log(jnp.clip(transition_tensor, 1e-8, 1.0))   # (x_n, x_p, u)
    log_observation = jnp.log(jnp.clip(theta_observation_tensor, 1e-8, 1.0))

    # Effective theta prior
    if prior_theta_logits is not None:
        q_theta_logits_init = prior_theta_logits
        effective_theta_prior = jax.nn.softmax(prior_theta_logits)
    else:
        q_theta_logits_init = jnp.log(theta_prior + 1e-8)
        effective_theta_prior = theta_prior

    keys = jax.random.split(key, 4)

    # q(u_t): (horizon, n_actions) — small noise around zero (near-uniform)
    q_u_logits_init = jax.random.normal(
        keys[0],
        shape=(config.planning_horizon, config.n_actions),
    ) * 0.01

    # q(x_t|x_{t-1}, θ): (horizon, n_states_next, n_states_prev, n_theta)
    # Initialize close to the action-marginalized log-transition.
    log_trans_action_marg = jnp.mean(log_transition, axis=2)  # (x_n, x_p)
    q_x_given_x_theta_logits_init = jax.random.normal(
        keys[1],
        shape=(config.planning_horizon, config.n_states, config.n_states, config.n_theta),
    ) * 0.01 + log_trans_action_marg[None, :, :, None]

    # q(y_θ_t|x_t, θ): (horizon, n_obs_theta, n_states, n_theta)
    q_y_theta_given_x_theta_logits_init = jax.random.normal(
        keys[2],
        shape=(config.planning_horizon, config.n_obs, config.n_states, config.n_theta),
    ) * 0.01 + log_observation[None, :, :, :]

    # q(y_loc_t|x_t): (horizon, n_locations, n_states)
    n_locations = location_observation_tensor.shape[0]
    log_location_obs = jnp.log(jnp.clip(location_observation_tensor, 1e-8, 1.0))
    q_y_location_given_x_logits_init = jax.random.normal(
        keys[3],
        shape=(config.planning_horizon, n_locations, config.n_states),
    ) * 0.01 + log_location_obs[None, :, :]

    params = {
        'q_theta_logits': q_theta_logits_init,
        'q_u_logits': q_u_logits_init,
        'q_x_given_x_theta_logits': q_x_given_x_theta_logits_init,
        'q_y_theta_given_x_theta_logits': q_y_theta_given_x_theta_logits_init,
        'q_y_location_given_x_logits': q_y_location_given_x_logits_init,
    }

    def loss_fn(params):
        return temporal_vfe_meanfield(
            q_theta_logits=params['q_theta_logits'],
            q_u_logits=params['q_u_logits'],
            q_x_given_x_theta_logits=params['q_x_given_x_theta_logits'],
            q_y_theta_given_x_theta_logits=params['q_y_theta_given_x_theta_logits'],
            q_y_location_given_x_logits=params['q_y_location_given_x_logits'],
            initial_state=initial_state,
            transition_tensor=transition_tensor,
            theta_observation_tensor=theta_observation_tensor,
            location_observation_tensor=location_observation_tensor,
            goal_mapping=goal_mapping,
            action_prior=action_prior,
            theta_prior=effective_theta_prior,
            horizon=config.planning_horizon,
        )

    optimizer = optax.adam(learning_rate=config.learning_rate)
    opt_state = optimizer.init(params)

    num_samples = (config.n_optimization_steps - 1) // 100 + 1
    loss_history_array = jnp.zeros(num_samples)

    def step_fn(carry, step_idx):
        params, opt_state, loss_hist, hist_idx = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        should_record = (step_idx % 100) == 0
        new_hist_idx = hist_idx + jnp.where(should_record, 1, 0)
        loss_hist = jax.lax.cond(
            should_record,
            lambda: loss_hist.at[hist_idx].set(loss),
            lambda: loss_hist,
        )
        return (params, opt_state, loss_hist, new_hist_idx), loss

    initial_carry = (params, opt_state, loss_history_array, 0)
    (params, opt_state, loss_history_array, _), all_losses = jax.lax.scan(
        step_fn,
        initial_carry,
        jnp.arange(config.n_optimization_steps),
    )

    loss_history = [float(x) for x in loss_history_array if x != 0.0]
    all_losses_list = [float(x) for x in all_losses]
    final_loss = float(all_losses[-1])

    marginals = extract_marginals_meanfield(
        q_theta_logits=params['q_theta_logits'],
        q_u_logits=params['q_u_logits'],
        q_x_given_x_theta_logits=params['q_x_given_x_theta_logits'],
        q_y_theta_given_x_theta_logits=params['q_y_theta_given_x_theta_logits'],
        q_y_location_given_x_logits=params['q_y_location_given_x_logits'],
        initial_state=initial_state,
        horizon=config.planning_horizon,
    )

    # q(u_1) is already a marginal under the mean field — no need to sum
    # over states or θ.
    q_first_action = marginals['q_u'][0]
    q_first_action = q_first_action / jnp.sum(q_first_action)

    return MeanfieldPlanningResult(
        q_theta=marginals['q_theta'],
        q_u=marginals['q_u'],
        q_x_given_x_theta=marginals['q_x_given_x_theta'],
        q_y_theta_given_x_theta=marginals['q_y_theta_given_x_theta'],
        q_y_location_given_x=marginals['q_y_location_given_x'],
        q_first_action=q_first_action,
        final_loss=final_loss,
        loss_history=loss_history,
        all_losses=all_losses_list,
        q_x_theta=marginals['q_x_theta'],
    )


def select_action_meanfield(
    result: MeanfieldPlanningResult,
    current_state_idx: int = 0,  # noqa: ARG001 — kept for signature parity
    timestep: int = 0,
) -> int:
    """
    Select action from the mean-field policy.

    Under the mean-field assumption q(u_t) is independent of x and θ, so we
    just argmax the marginal at the requested timestep. `current_state_idx`
    is accepted for signature parity with `select_action_temporal` but
    unused.
    """
    if timestep == 0:
        return int(jnp.argmax(result.q_first_action))
    return int(jnp.argmax(result.q_u[timestep]))
