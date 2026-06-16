"""VBP channel planner — cross-entropy planning with action channel.

Standard Bethe BP with a single action entropy correction reparameterized
into the dynamics kernel via one action channel r(u|x) per timestep.
Theta is inferred via cavity messages.

Modified kernel: kappa_t(x_old, x_new, theta, u) = T(x_old, x_new, theta, u) * r_t(u|x_old)
Observation kernel: raw B (unmodified).

Supports both state preferences C(x, theta) and observation preferences C_obs(y)
via the Goal abstraction.

Spec-aware: when ThetaSpec / FactorSpec metadata is supplied, every θ
operation runs on per-factor narrow tensors. The M1/M2 single-component
case (component_sizes=(n_static,), every θ-conditional factor sees that
single component) collapses to the legacy shapes.

Ported from MinigridMP-AIF-JAX/inference/vbp_channel.py (dense path only).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from learningpomdp.factors import FactorSpec, ThetaSpec

from .goal import Goal, is_goal_object, compute_terminal_goal, resolve_goal
from .messages import safe_log
from .bp_core import (
    damp_log_channel,
    compute_log_reduced,
    forward_pass,
    backward_pass,
    compute_theta_cavities,
    compute_obs_to_x_msgs,
    compute_obs_to_theta_msgs,
    compute_dyn_to_theta_msgs,
    compute_dyn_pair_marginal,
)


def _compute_dyn_kernels_vbp(log_T_kernel, log_r_ux, n_theta_axes):
    """VBP dynamics kernels: action channel enters numerator.

    log_T_kernel: (x_old, x_new, *t_theta_shape, u)
    log_r_ux:     (T, x_old, u)
    Output:       (T, x_old, x_new, *t_theta_shape, u)
    """
    # Reshape r_ux to (T, x_old, 1, *(1,)*n_theta_axes, u) to broadcast.
    horizon = log_r_ux.shape[0]
    r_shape = (horizon, log_r_ux.shape[1], 1) + (1,) * n_theta_axes + (log_r_ux.shape[2],)
    return log_T_kernel[None] + log_r_ux.reshape(r_shape)


def _compute_action_channel(log_pair_marginal):
    """Action channel r(u|x) = q_pair(x, u) / q(x). Pair shape (T, x_old, u);
    cond axis is u (axis 2)."""
    return log_pair_marginal - logsumexp(log_pair_marginal, axis=2, keepdims=True)


def _vbp_planning_impl(
    q_current_state, q_static_state, transition_tensor,
    B_list, goal, horizon, n_iterations,
    theta_spec: ThetaSpec,
    t_factor_spec: FactorSpec,
    b_factor_specs: tuple[FactorSpec, ...],
    damping=1.0,
):
    """Spec-aware VBP planning loop. JIT-compiled by the caller (statics are
    n_iterations + horizon + spec metadata).

    Args:
        q_current_state: (n_states,)
        q_static_state: (n_static,) flat θ marginal.
        transition_tensor: (n_states_new, n_states_old, *t_theta_shape, n_actions).
        B_list: per-channel (per-factor narrow shapes).
        goal: 1D or 2D goal array.
        horizon, n_iterations: BP iteration counts.
        theta_spec, t_factor_spec, b_factor_specs: factor structure.
        damping: action-channel damping.

    Returns:
        action_dist: (n_actions,)
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_tensor.shape[-1]
    n_channels = len(B_list)
    n_theta_axes_T = t_factor_spec.n_components if not t_factor_spec.is_theta_independent else 0
    has_pref = goal.ndim == 2

    log_action_prior = jnp.full(n_actions, -jnp.log(n_actions))

    # log T kernel: (x_old, x_new, *t_theta_shape, u). Move axis 0 (x_new) to position 1.
    log_T = safe_log(transition_tensor)
    log_T_kernel = jnp.moveaxis(log_T, 0, 1)
    log_q0 = safe_log(q_current_state)

    # Per-component log priors over θ.
    q_static_factored = q_static_state.reshape(theta_spec.component_sizes)
    log_prior_components = []
    for k in range(theta_spec.n_components):
        marginal_axes = tuple(j for j in range(theta_spec.n_components) if j != k)
        marg = q_static_factored.sum(axis=marginal_axes) if marginal_axes else q_static_factored
        log_prior_components.append(safe_log(marg))
    log_prior_components = tuple(log_prior_components)

    if has_pref:
        log_C = safe_log(goal)        # (n_states, n_static)
    else:
        log_C = None

    # Tile per-channel B over time.
    log_B_tiled_list = []
    for B_c in B_list:
        log_B_c = safe_log(B_c)
        log_B_tiled = jnp.broadcast_to(log_B_c[None], (horizon + 1,) + log_B_c.shape)
        log_B_tiled_list.append(log_B_tiled)

    # Initialize messages at uniform.
    if not t_factor_spec.is_theta_independent:
        log_dyn_to_theta = jnp.zeros((horizon,) + t_factor_spec.shape_for(theta_spec))
    else:
        log_dyn_to_theta = jnp.zeros((horizon,))
    log_obs_to_theta_init = tuple(
        None if spec.is_theta_independent
        else jnp.zeros((horizon + 1,) + spec.shape_for(theta_spec))
        for spec in b_factor_specs
    )
    q_u = jnp.zeros((horizon, n_actions))
    log_r_ux = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))

    def body_fn(_, carry):
        log_dyn_to_theta, log_obs_to_theta, q_u, log_r_ux = carry

        log_cavity_dyn, log_cavity_obs = compute_theta_cavities(
            log_prior_components,
            log_dyn_to_theta, t_factor_spec,
            log_obs_to_theta, b_factor_specs,
        )

        log_dyn_kernels = _compute_dyn_kernels_vbp(
            log_T_kernel, log_r_ux, n_theta_axes_T
        )
        log_reduced_per_t = compute_log_reduced(
            log_dyn_kernels, log_cavity_dyn, n_theta_axes=n_theta_axes_T
        )

        log_obs_to_x = compute_obs_to_x_msgs(
            log_B_tiled_list, log_cavity_obs, b_factor_specs
        )

        # Compute goal for this iteration.
        if has_pref:
            # log q(θ) joint marginal from per-factor messages.
            # For terminal goal we need the full joint q(θ). Build it from
            # per-component aggregation: q(θ) = ∏_k q(θ_k), with q(θ_k)
            # ∝ exp(prior_k + Σ_factors_seeing_k marg_to_k).
            log_theta_components_post = []
            for k in range(theta_spec.n_components):
                log_total = log_prior_components[k]
                if k in t_factor_spec.theta_components:
                    from learningpomdp.factors import marginalize_to_component
                    dyn_marg = marginalize_to_component(
                        log_dyn_to_theta, t_factor_spec, k, leading_axes=1
                    )
                    log_total = log_total + dyn_marg.sum(axis=0)
                for c, b_spec in enumerate(b_factor_specs):
                    if log_obs_to_theta[c] is None or k not in b_spec.theta_components:
                        continue
                    from learningpomdp.factors import marginalize_to_component
                    obs_marg = marginalize_to_component(
                        log_obs_to_theta[c], b_spec, k, leading_axes=1
                    )
                    log_total = log_total + obs_marg.sum(axis=0)
                log_theta_components_post.append(
                    log_total - logsumexp(log_total)
                )
            # Build joint log q(θ) by outer product (sum in log space) of components.
            log_theta_post = jnp.zeros(theta_spec.component_sizes)
            for k, log_comp in enumerate(log_theta_components_post):
                shape = [1] * theta_spec.n_components
                shape[k] = theta_spec.component_sizes[k]
                log_theta_post = log_theta_post + log_comp.reshape(shape)
            log_theta_post_flat = log_theta_post.reshape(-1)
            log_theta_post_flat = log_theta_post_flat - logsumexp(log_theta_post_flat)
            log_goal = compute_terminal_goal(log_C, log_theta_post_flat)
        else:
            log_goal = safe_log(goal)

        log_fwd_msgs = forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, log_obs_to_x, horizon
        )
        log_bwd_msgs, q_u = backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
            log_obs_to_x, horizon
        )

        log_dyn_to_theta = compute_dyn_to_theta_msgs(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_action_prior, horizon, n_theta_axes=n_theta_axes_T,
        )

        log_obs_to_theta = compute_obs_to_theta_msgs(
            log_B_tiled_list, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            b_factor_specs,
        )

        log_pair = compute_dyn_pair_marginal(
            log_dyn_kernels, log_fwd_msgs, log_bwd_msgs, log_obs_to_x,
            log_cavity_dyn, log_action_prior, n_theta_axes=n_theta_axes_T,
        )
        raw_log_r_ux = _compute_action_channel(log_pair)
        new_log_r_ux = damp_log_channel(log_r_ux, raw_log_r_ux, damping, cond_axis=2)

        return (log_dyn_to_theta, log_obs_to_theta, q_u, new_log_r_ux)

    init_carry = (log_dyn_to_theta, log_obs_to_theta_init, q_u, log_r_ux)
    _, _, q_u, _ = lax.fori_loop(0, n_iterations, body_fn, init_carry)

    action_dist = q_u[0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    return action_dist


# Cache of jitted variants keyed by (horizon, n_iterations, theta_spec,
# t_factor_spec, b_factor_specs). Each unique factor configuration gets its
# own compiled program; for door-code/MiniGrid the cache holds a single entry.
_vbp_jit_cache: dict[tuple, callable] = {}


def _get_jitted_vbp(horizon, n_iterations, theta_spec, t_factor_spec,
                    b_factor_specs):
    key = (horizon, n_iterations, theta_spec, t_factor_spec, b_factor_specs)
    fn = _vbp_jit_cache.get(key)
    if fn is not None:
        return fn

    @partial(jax.jit, static_argnames=("damping",))
    def _impl(q_current_state, q_static_state, transition_tensor,
              B_list, goal, damping=1.0):
        return _vbp_planning_impl(
            q_current_state, q_static_state, transition_tensor,
            B_list, goal, horizon, n_iterations,
            theta_spec, t_factor_spec, b_factor_specs, damping,
        )

    _vbp_jit_cache[key] = _impl
    return _impl


class VBPPlanner:
    """Cross-entropy planner via VBP with action channel reparameterization.

    Uses standard Bethe BP with action entropy correction. Theta is inferred
    via cavity messages. No epistemic terms — exploits known model structure.
    """

    def __init__(self, n_iterations: int = 150, damping: float = 1.0) -> None:
        self._n_iterations = n_iterations
        self._damping = damping

    def plan(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list[jnp.ndarray],
        goal: jnp.ndarray | Goal,
        horizon: int,
        theta_spec: ThetaSpec,
        t_factor_spec: FactorSpec,
        b_factor_specs: tuple[FactorSpec, ...],
        theta_mask: list[bool] | None = None,
    ) -> jnp.ndarray:
        # theta_mask is informational; the per-channel θ axis layout is
        # carried by b_factor_specs.
        # Convert Goal object to raw goal array.
        if is_goal_object(goal):
            goal_array = resolve_goal(
                goal, B_list, q_static,
                theta_spec=theta_spec,
                b_factor_specs=b_factor_specs,
            )
        else:
            goal_array = goal

        jitted = _get_jitted_vbp(
            horizon, self._n_iterations,
            theta_spec, t_factor_spec, b_factor_specs,
        )
        return jitted(
            q_state, q_static, T, B_list, goal_array, self._damping,
        )
