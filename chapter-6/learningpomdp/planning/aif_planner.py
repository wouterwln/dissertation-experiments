"""Active Inference planner — VBP action channels + dyn channels + obs channels.

Combines three channel reparameterizations for explore+exploit behaviour:
- VBP action channel: r(u|x) enters dynamics kernel in NUMERATOR
- Dyn channel: r(x'|x,u) enters dynamics kernel in DENOMINATOR
- Obs channel: obs kernels use r(y|x,theta) and r(y|x,theta)/r(y|x)

Theta is prior-locked during planning — no messages flow toward θ, no
cavities, no posterior updates. Why: the EFE objective rewards mutual
information I[θ; y] = H[q(θ)] - E_y[H[q(θ|y)]]. If we let θ-side messages
update q(θ) inside the planning loop, the optimizer maximizes that
mutual information by pushing q(θ) toward uniform — an uninformative
distribution it never has to commit to. The planner would "discover"
that throwing away its current beliefs makes any observation maximally
informative, which is nonsense: the agent's actual θ posterior was earned
by real evidence and must not drift inside imagined trajectories. So we
block backward (θ-ward) messages and let info-gain over θ enter only via
the obs-channel reparameterization, which carries KL[q(θ|y) || q(θ)] into
the x-side messages without changing q(θ) itself. This is the structural
difference vs VBP, which has no such constraint.

Obs channels can be precomputed outside the planning loop for efficiency.

Spec-aware: per-factor narrow shapes throughout. The single-component
default reduces to the legacy axis layout byte-for-byte.

Ported from MinigridMP-AIF-JAX/inference/active_inference.py (dense path only).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from functools import partial

from learningpomdp.factors import (
    FactorSpec,
    ThetaSpec,
    expand_per_component_to_factor,
)

from .goal import Goal, is_goal_object, compute_terminal_goal, resolve_goal
from .messages import LOG_ZERO, safe_log, safe_log_div
from .bp_core import (
    damp_log_channel,
    channel_tv_residual,
    forward_pass,
    backward_pass,
)


def _theta_for_factor(
    log_theta_components: tuple[jnp.ndarray, ...],
    factor: FactorSpec,
) -> jnp.ndarray:
    """Build log q(θ_components_of_factor) as outer product of per-component
    marginals (additive in log space, exact under independent components)."""
    if factor.is_theta_independent:
        return jnp.zeros(())
    return expand_per_component_to_factor(
        {k: log_theta_components[k] for k in factor.theta_components},
        factor, leading_axes=0,
    )


def _precompute_obs_channels(log_B_list, log_theta_components, b_factor_specs,
                              n_iterations, damping=1.0):
    """Converge obs channels independently and compute obs->x message.

    Channel updates depend only on B, prior θ, and damping — same for all
    timesteps. Each θ-conditional channel has axes (n_obs, n_states, *b_theta_shape).

    Channels with identical shape signatures are batched along a leading
    axis and converged in a single fori_loop, cutting kernel launches from
    O(n_channels) to O(n_shape_classes). For RockSample with K rocks, the
    K θ-dep channels share shape (n_obs, n_states, c_q, c_s) and reduce to
    one batched loop. A single-channel group is shape-equivalent to the
    legacy unbatched code (leading axis of size 1).

    Args:
        log_B_list: list of per-channel log B tensors (per-factor narrow).
        log_theta_components: tuple of per-component log priors.
        b_factor_specs: tuple of FactorSpec.
        n_iterations: number of iterations for channel convergence (static).
        damping: damping factor.

    Returns:
        log_obs_to_x: (n_states,) message from obs factors to x.
    """
    n_states = log_B_list[0].shape[1]

    # Partition channels by shape signature. Done at trace time — keys are
    # static, dict iteration order is insertion order in Python 3.7+.
    indep_groups: dict[int, list[int]] = {}
    dep_groups: dict[tuple, list[int]] = {}
    for c, (log_B_c, spec_c) in enumerate(zip(log_B_list, b_factor_specs)):
        n_obs_c = log_B_c.shape[0]
        if spec_c.is_theta_independent:
            indep_groups.setdefault(n_obs_c, []).append(c)
        else:
            dep_groups.setdefault((n_obs_c, log_B_c.shape[2:]), []).append(c)

    log_obs_to_x = jnp.zeros(n_states)

    # θ-independent groups: stack and run one batched fori_loop per shape.
    for n_obs, idxs in indep_groups.items():
        G = len(idxs)
        log_B_stack = jnp.stack([log_B_list[c] for c in idxs], axis=0)
        # Shape: (G, n_obs, n_states).
        log_obs_ch_init = jnp.full((G, n_obs, n_states), -jnp.log(n_obs))

        def body_indep(_, log_obs_ch):
            log_obs_kernels = log_B_stack + log_obs_ch
            raw_log_obs_ch = log_obs_kernels - logsumexp(
                log_obs_kernels, axis=1, keepdims=True)
            return damp_log_channel(log_obs_ch, raw_log_obs_ch, damping, cond_axis=1)

        log_obs_ch = lax.fori_loop(0, n_iterations, body_indep, log_obs_ch_init)
        log_obs_kernels = log_B_stack + log_obs_ch
        log_per_g = logsumexp(log_obs_kernels, axis=1)  # (G, n_states)
        log_obs_to_x = log_obs_to_x + log_per_g.sum(axis=0)

    # θ-dependent groups: same idea, but carries (log_obs_ch, log_marg_obs_ch).
    for (n_obs, b_theta_shape), idxs in dep_groups.items():
        G = len(idxs)
        n_b_theta = len(b_theta_shape)
        log_B_stack = jnp.stack([log_B_list[c] for c in idxs], axis=0)
        # Shape: (G, n_obs, n_states, *b_theta_shape).

        # Per-channel θ marginals (each channel sees its own components).
        log_theta_stack = jnp.stack([
            _theta_for_factor(log_theta_components, b_factor_specs[c])
            for c in idxs
        ], axis=0)  # (G, *b_theta_shape)
        # Aligned for broadcast as (G, 1, 1, *b_theta_shape).
        log_theta_aligned = log_theta_stack.reshape((G, 1, 1) + b_theta_shape)

        log_obs_ch_init = jnp.full(
            (G, n_obs, n_states) + b_theta_shape, -jnp.log(n_obs)
        )
        log_marg_obs_ch_init = jnp.full((G, n_obs, n_states), -jnp.log(n_obs))

        # Axis map for stacked tensors: 0=G, 1=n_obs, 2=n_states,
        # 3..2+n_b_theta=θ. obs axis shifted by +1 from unstacked.
        theta_axes = tuple(range(3, 3 + n_b_theta))

        def body_dep(_, carry):
            log_obs_ch, log_marg_obs_ch = carry
            log_marg_aligned = log_marg_obs_ch.reshape(
                (G, n_obs, n_states) + (1,) * n_b_theta
            )
            log_obs_kernels = (log_B_stack + log_obs_ch
                               + safe_log_div(log_obs_ch, log_marg_aligned))
            raw_log_obs_ch = log_obs_kernels - logsumexp(
                log_obs_kernels, axis=1, keepdims=True)

            kernel_with_prior = log_obs_kernels + log_theta_aligned
            theta_marg = logsumexp(kernel_with_prior, axis=theta_axes)
            raw_log_marg_obs_ch = theta_marg - logsumexp(
                theta_marg, axis=1, keepdims=True)

            new_log_obs_ch = damp_log_channel(
                log_obs_ch, raw_log_obs_ch, damping, cond_axis=1)
            new_log_marg_obs_ch = damp_log_channel(
                log_marg_obs_ch, raw_log_marg_obs_ch, damping, cond_axis=1)
            return (new_log_obs_ch, new_log_marg_obs_ch)

        log_obs_ch, log_marg_obs_ch = lax.fori_loop(
            0, n_iterations, body_dep,
            (log_obs_ch_init, log_marg_obs_ch_init))

        log_marg_aligned = log_marg_obs_ch.reshape(
            (G, n_obs, n_states) + (1,) * n_b_theta
        )
        log_obs_kernels = (log_B_stack + log_obs_ch
                           + safe_log_div(log_obs_ch, log_marg_aligned))
        terms = log_obs_kernels + log_theta_aligned
        log_per_g = logsumexp(terms, axis=(1,) + theta_axes)  # (G, n_states)
        log_obs_to_x = log_obs_to_x + log_per_g.sum(axis=0)

    log_obs_to_x = log_obs_to_x - logsumexp(log_obs_to_x)
    return log_obs_to_x


def _aif_planning_impl(
    q_current_state, q_static_state, transition_tensor,
    B_list, goal, horizon, n_iterations,
    theta_spec: ThetaSpec,
    t_factor_spec: FactorSpec,
    b_factor_specs: tuple[FactorSpec, ...],
    damping=0.25,
    return_residuals=False,
):
    """Spec-aware AIF planning with precomputed obs channels.

    `return_residuals` is a trace-time Python bool. When False the traced
    computation is exactly the one below with the residual branches
    absent, so the canonical path is unchanged; when True the function
    additionally returns a (n_iterations,) trace of the per-iteration
    fixed-point residual over the two channels updated in the loop (the
    action channel r(u|x) and the dynamics channel r(x'|x,u)), as the
    worst-conditional total variation between each channel and its
    undamped image (`bp_core.channel_tv_residual`). The obs channels
    converge in their own loop inside `_precompute_obs_channels` and are
    not covered by that trace.
    """
    n_states = q_current_state.shape[0]
    n_actions = transition_tensor.shape[-1]
    n_theta_axes_T = (
        t_factor_spec.n_components if not t_factor_spec.is_theta_independent else 0
    )
    has_pref = goal.ndim == 2

    log_action_prior = jnp.full(n_actions, -jnp.log(n_actions))
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
        log_C = safe_log(goal)
        # `q_static_state` may arrive flat (n_static,) or factored
        # (*component_sizes,); compute_terminal_goal expects the
        # flat 1-D vector matching log_C's last axis.
        log_prior_theta_flat = safe_log(q_static_state.reshape(-1))
        log_goal = compute_terminal_goal(log_C, log_prior_theta_flat)
    else:
        log_goal = safe_log(goal)

    # T-side base kernel: marginalize T over θ with the (factored) prior.
    # T shape: (x_new, x_old, *t_theta_shape, u). Move x_new to axis 1:
    log_T = safe_log(transition_tensor)
    log_T_kernel = jnp.moveaxis(log_T, 0, 1)   # (x_old, x_new, *t_theta_shape, u)

    # Build factored log prior over T's components and collapse θ here,
    # before the BP loop. This bakes "θ is prior-locked" into the dyn-side
    # geometry: the loop never sees a θ axis on T, so it has no surface to
    # form dyn-to-θ messages on (see module docstring for why we forbid
    # those). Under factored q(θ) = ∏_k q(θ_k), expanding the prior via
    # _theta_for_factor and logsumexp'ing T's θ axes gives the exact
    # prior-predictive p(x'|x,u) that the AIF dyn channel converges to.
    log_theta_for_T = _theta_for_factor(log_prior_components, t_factor_spec)
    if t_factor_spec.is_theta_independent:
        # No θ axes in T; log_base is just log_T_kernel itself.
        log_base = log_T_kernel
    else:
        # Align prior to broadcast: (1, 1, *t_theta_shape, 1)
        log_theta_aligned = log_theta_for_T.reshape((1, 1) + log_theta_for_T.shape + (1,))
        theta_axes = tuple(range(2, 2 + n_theta_axes_T))
        log_base = logsumexp(log_T_kernel + log_theta_aligned, axis=theta_axes)
    # log_base shape: (x_old, x_new, u)

    # Precompute obs->x message.
    log_B_list = [safe_log(B_c) for B_c in B_list]
    log_obs_to_x_single = _precompute_obs_channels(
        log_B_list, log_prior_components, b_factor_specs, n_iterations, damping
    )
    log_local_to_x = jnp.broadcast_to(
        log_obs_to_x_single[None, :], (horizon + 1, n_states)
    )

    # Initialize.
    q_u = jnp.zeros((horizon, n_actions))
    log_fwd_prev = jnp.zeros((horizon + 1, n_states))
    log_bwd_prev = jnp.zeros((horizon + 1, n_states))
    log_r_ux = jnp.full((horizon, n_states, n_actions), -jnp.log(n_actions))
    log_dyn_channels = jnp.full(
        (horizon, n_states, n_states, n_actions), -jnp.log(n_states))

    def body_fn(i, carry):
        q_u, log_fwd_prev, log_bwd_prev, log_r_ux, log_dyn_channels = carry[:5]

        dyn_valid = log_dyn_channels > LOG_ZERO / 2
        neg_dyn_ch = jnp.where(dyn_valid, -log_dyn_channels, 0.0)
        base_4d = log_base[None] + neg_dyn_ch + log_r_ux[:, :, None, :]
        base_4d = jnp.where(dyn_valid, base_4d, LOG_ZERO)

        log_reduced_per_t = base_4d.transpose(0, 2, 1, 3)

        log_fwd_msgs = forward_pass(
            log_reduced_per_t, log_q0, log_action_prior, log_local_to_x, horizon,
            log_prev_fwd=log_fwd_prev, msg_damping=damping
        )
        log_bwd_msgs, q_u_new = backward_pass(
            log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
            log_local_to_x, horizon,
            log_prev_bwd=log_bwd_prev, msg_damping=damping
        )

        fwd_local = log_fwd_msgs[:-1] + log_local_to_x[:-1]
        bwd_local = log_bwd_msgs[1:] + log_local_to_x[1:]
        log_theta_marg = (base_4d
                          + fwd_local[:, :, None, None]
                          + bwd_local[:, None, :, None]
                          + log_action_prior[None, None, None, :])
        raw_log_dyn_channels = log_theta_marg - logsumexp(
            log_theta_marg, axis=2, keepdims=True)
        log_pair = logsumexp(log_theta_marg, axis=2)
        raw_log_r_ux = log_pair - logsumexp(log_pair, axis=2, keepdims=True)

        new_log_r_ux = damp_log_channel(log_r_ux, raw_log_r_ux, damping, cond_axis=2)
        new_log_dyn_channels = damp_log_channel(
            log_dyn_channels, raw_log_dyn_channels, damping, cond_axis=2)

        out = (q_u_new, log_fwd_msgs, log_bwd_msgs, new_log_r_ux, new_log_dyn_channels)
        if return_residuals:
            # Against the *raw* (undamped) images, so the trace is the
            # residual of the fixed-point map and not of the damped step.
            residual = jnp.maximum(
                channel_tv_residual(log_r_ux, raw_log_r_ux, cond_axis=2),
                channel_tv_residual(
                    log_dyn_channels, raw_log_dyn_channels, cond_axis=2),
            )
            out = out + (carry[5].at[i].set(residual),)
        return out

    init_carry = (q_u, log_fwd_prev, log_bwd_prev, log_r_ux, log_dyn_channels)
    if return_residuals:
        init_carry = init_carry + (jnp.zeros(n_iterations),)
    carry = lax.fori_loop(0, n_iterations, body_fn, init_carry)

    action_dist = carry[0][0]
    action_dist = action_dist / (action_dist.sum() + 1e-10)
    if return_residuals:
        return action_dist, carry[5]
    return action_dist


_aif_jit_cache: dict[tuple, callable] = {}


def _get_jitted_aif(horizon, n_iterations, theta_spec, t_factor_spec,
                    b_factor_specs, return_residuals=False):
    # `return_residuals` is part of the key because it changes the traced
    # output signature. The canonical (False) variant keeps its own entry
    # and compiles the same program as before.
    key = (horizon, n_iterations, theta_spec, t_factor_spec, b_factor_specs,
           return_residuals)
    fn = _aif_jit_cache.get(key)
    if fn is not None:
        return fn

    @partial(jax.jit, static_argnames=("damping",))
    def _impl(q_current_state, q_static_state, transition_tensor,
              B_list, goal, damping=0.25):
        return _aif_planning_impl(
            q_current_state, q_static_state, transition_tensor,
            B_list, goal, horizon, n_iterations,
            theta_spec, t_factor_spec, b_factor_specs, damping,
            return_residuals,
        )

    _aif_jit_cache[key] = _impl
    return _impl


class AIFPlanner:
    """Active Inference planner with three channel reparameterizations."""

    def __init__(self, n_iterations: int = 150, damping: float = 0.9) -> None:
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
        return_residuals: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        """Plan one action distribution.

        return_residuals: if True, also return the (n_iterations,) trace
            of per-iteration channel residuals from the fixed-point loop
            (see `_aif_planning_impl`). Off by default — the runners never
            pass it, so the canonical path is untouched.
        """
        # theta_mask is informational; the per-channel θ axis layout is
        # carried by b_factor_specs.
        if is_goal_object(goal):
            goal_array = resolve_goal(
                goal, B_list, q_static,
                theta_spec=theta_spec,
                b_factor_specs=b_factor_specs,
            )
        else:
            goal_array = goal

        jitted = _get_jitted_aif(
            horizon, self._n_iterations,
            theta_spec, t_factor_spec, b_factor_specs,
            return_residuals,
        )
        return jitted(
            q_state, q_static, T, B_list, goal_array, self._damping,
        )
