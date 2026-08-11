"""Shared belief propagation infrastructure for planning.

Spec-aware: every θ-touching primitive takes ThetaSpec / FactorSpec
metadata and operates on per-factor narrow tensors (only over the
components a factor's `theta_components` declares). The M1/M2
single-component specialization (every θ-conditional factor sees one
component of size n_static) collapses to the legacy shapes byte-for-byte.

Originally ported from MinigridMP-AIF-JAX/inference/region_extended_loopy_bp.py.
All internal computation is in log-space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from learningpomdp.factors import (
    FactorSpec,
    ThetaSpec,
    expand_per_component_to_factor,
    marginalize_to_component,
)

from .messages import LOG_ZERO


# =============================================================================
# Channel damping
# =============================================================================


def damp_log_channel(log_old, log_new, damping, cond_axis):
    """Geometric damping of log-space channels.

    damped = (1-alpha)*log_old + alpha*log_new

    damping=1.0 -> use new channels (no damping)
    damping=0.5 -> geometric mean in probability space
    """
    damped = (1.0 - damping) * log_old + damping * log_new

    valid = (log_old > LOG_ZERO / 2) | (log_new > LOG_ZERO / 2)
    damped = jnp.where(valid, damped, LOG_ZERO)

    normalizer = logsumexp(damped, axis=cond_axis, keepdims=True)
    return jnp.where(valid, damped - normalizer, LOG_ZERO)


def channel_tv_residual(log_old, log_new, cond_axis):
    """Fixed-point residual of a channel update: worst-conditional total
    variation distance, measured in probability space.

    Diagnostic only — nothing in the planners consumes this. It answers
    "how far is this channel from the fixed point of its own update?",
    which the fixed iteration count alone cannot.

    Probability space, not log space. A log-space `max |Δ log r|` is
    dominated by entries whose probability is negligible: `safe_log`
    floors at 1e-30, so a tail entry drifting between p = 1e-25 and
    p = 1e-30 moves its log by ~11 while moving the distribution by
    ~1e-25. Those swings put a floor under the max that has nothing to do
    with convergence, and they hide the decay of the entries that carry
    the mass. Exponentiating first also removes the need for a LOG_ZERO
    mask: structurally zero entries have exp(LOG_ZERO) == 0 exactly, so
    they contribute 0 to the sum on their own.

    Total variation, not max |Δp|, because both arguments are *sets of
    conditional distributions* over `cond_axis` (that is what
    `damp_log_channel` normalizes them to be). TV is then the standard
    distance between the old and new conditional, bounded in [0, 1] and
    independent of how many entries the conditional has, so the number is
    interpretable on its own ("the worst conditional moved by X") rather
    than only relative to itself.

    `log_new` must be the *undamped* image of `log_old` under the channel
    update, not the damped one. Then the residual is that of the
    fixed-point map itself, and its scale does not carry a factor of the
    damping weight, so traces are comparable both across iterations and
    across damping.

    The guarantee runs one way only. If a channel is at a fixed point of
    its own update then this is exactly zero; zero (or near-zero) does
    **not** certify a fixed point of the iteration. Two reasons, both in
    `aif_planner._aif_planning_impl`: `q_u` is rederived from
    `backward_pass` on messages that are still moving, and
    `log_fwd_prev` / `log_bwd_prev` are damped carry state that persists
    across iterations. Neither is in this residual, so a small residual
    says the two traced channels have stopped moving relative to their
    own update — not that the loop as a whole has settled.

    Conditioning slices that are not distributions in both arguments are
    dropped. Such slices exist: where the forward/backward messages have
    already ruled the conditioning index out, the whole slice is LOG_ZERO
    and the update's normalizer is a 0/0, which leaves the slice at
    log 0 = 0 for every entry — an all-ones "conditional" that no
    consumer reads but that would otherwise pin the max at
    (n_cond - 1) / 2 forever. Testing the mass is enough to spot them:
    every slice a consumer reads sums to 1.
    """
    p_old = jnp.exp(log_old)
    p_new = jnp.exp(log_new)
    tv = 0.5 * jnp.sum(jnp.abs(p_new - p_old), axis=cond_axis)
    normalized = (
        (jnp.abs(p_old.sum(axis=cond_axis) - 1.0) < 1e-3)
        & (jnp.abs(p_new.sum(axis=cond_axis) - 1.0) < 1e-3)
    )
    return jnp.max(jnp.where(normalized, tv, 0.0))


# =============================================================================
# Reduced tensors
# =============================================================================


def compute_log_reduced(log_kernels, log_cavity, n_theta_axes=1):
    """Compute per-timestep reduced tensors from dynamics kernels (log-space).

    reduced[t](x_new, x_old, u) = sum_θ kernel_t(x_old, x_new, *θ, u) * cavity[t](*θ)

    Args:
        log_kernels: (T, x_old, x_new, *t_theta_shape, u) log dynamics kernels.
            For single-component T (door-code): (T, x_old, x_new, n_static, u).
        log_cavity: (T, *t_theta_shape) log per-timestep cavity beliefs on θ.
            For single-component: (T, n_static).
        n_theta_axes: number of θ axes in the kernel/cavity (default 1).

    Returns:
        (T, x_new, x_old, u) per-timestep log reduced tensors.
    """
    T_dim = log_cavity.shape[0]
    t_theta_shape = log_cavity.shape[1:]
    # Align cavity onto the kernel's axis layout: insert size-1 at x_old, x_new and u.
    cavity_aligned = log_cavity.reshape(
        (T_dim, 1, 1) + t_theta_shape + (1,)
    )
    terms = log_kernels + cavity_aligned
    theta_axes = tuple(range(3, 3 + n_theta_axes))
    result = logsumexp(terms, axis=theta_axes)  # (T, x_old, x_new, u)
    return result.transpose(0, 2, 1, 3)  # (T, x_new, x_old, u)


# =============================================================================
# Forward/backward passes
# =============================================================================


def forward_pass(log_reduced_per_t, log_q_x0, log_action_prior, log_obs_to_x,
                 horizon, log_prev_fwd=None, msg_damping=1.0):
    """Forward pass with obs->x message injection (log-space).

    Args:
        log_reduced_per_t: (T, x_new, x_old, u)
        log_q_x0: (n_states,) log initial state belief
        log_action_prior: (n_actions,) log prior over actions
        log_obs_to_x: (T+1, n_states) log messages from obs factors to x
        horizon: T
        log_prev_fwd: (T+1, n_states) previous iteration's fwd messages (optional)
        msg_damping: float, inertial damping weight (1.0 = no damping)

    Returns:
        log_fwd_msgs: (T+1, n_states) log forward messages
    """
    n_states = log_q_x0.shape[0]
    log_fwd = jnp.zeros((horizon + 1, n_states))
    log_fwd = log_fwd.at[0].set(log_q_x0)

    def body_fn(t, log_fwd):
        log_x_msg = log_fwd[t] + log_obs_to_x[t]
        terms = (log_reduced_per_t[t]
                 + log_x_msg[None, :, None]
                 + log_action_prior[None, None, :])
        log_q_next = logsumexp(terms, axis=(1, 2))
        log_q_next = log_q_next - logsumexp(log_q_next)
        if log_prev_fwd is not None:
            log_q_next = (1.0 - msg_damping) * log_prev_fwd[t + 1] + msg_damping * log_q_next
            log_q_next = log_q_next - logsumexp(log_q_next)
        return log_fwd.at[t + 1].set(log_q_next)

    return lax.fori_loop(0, horizon, body_fn, log_fwd)


def backward_pass(log_reduced_per_t, log_fwd_msgs, log_goal, log_action_prior,
                  log_obs_to_x, horizon, log_prev_bwd=None, msg_damping=1.0):
    """Backward pass with obs->x message injection (log-space).

    Returns:
        log_bwd_msgs: (T+1, n_states) log backward messages
        q_u: (T, n_actions) action marginals (probability space)
    """
    n_states = log_goal.shape[0]
    n_actions = log_action_prior.shape[0]
    log_bwd = jnp.zeros((horizon + 1, n_states))
    log_bwd = log_bwd.at[horizon].set(log_goal)
    q_u = jnp.zeros((horizon, n_actions))

    def body_fn(carry, t):
        log_bwd, q_u = carry
        log_bwd_t1 = log_bwd[t + 1] + log_obs_to_x[t + 1]

        # Action marginal
        log_fwd_t = log_fwd_msgs[t] + log_obs_to_x[t]
        terms = (log_reduced_per_t[t]
                 + log_bwd_t1[:, None, None]
                 + log_fwd_t[None, :, None])
        log_msg_to_u = logsumexp(terms, axis=(0, 1))
        q_u_t = jax.nn.softmax(log_msg_to_u + log_action_prior)
        q_u = q_u.at[t].set(q_u_t)

        # Backward message to x_t
        terms_bwd = (log_reduced_per_t[t]
                     + log_bwd_t1[:, None, None]
                     + log_action_prior[None, None, :])
        log_bwd_t = logsumexp(terms_bwd, axis=(0, 2))
        log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
        if log_prev_bwd is not None:
            log_bwd_t = (1.0 - msg_damping) * log_prev_bwd[t] + msg_damping * log_bwd_t
            log_bwd_t = log_bwd_t - logsumexp(log_bwd_t)
        log_bwd = log_bwd.at[t].set(log_bwd_t)

        return (log_bwd, q_u), None

    (log_bwd, q_u), _ = lax.scan(
        body_fn, (log_bwd, q_u), jnp.arange(horizon - 1, -1, -1)
    )

    return log_bwd, q_u


# =============================================================================
# Messages to/from theta
# =============================================================================


def compute_dyn_to_theta_msgs(log_dyn_kernels, log_fwd_msgs, log_bwd_msgs,
                               log_obs_to_x, log_action_prior, horizon,
                               n_theta_axes=1):
    """Compute messages from each dynamics factor to θ (log-space).

    Args:
        log_dyn_kernels: (T, x_old, x_new, *t_theta_shape, u).
            For single-component T: (T, x_old, x_new, n_static, u).
        log_fwd_msgs: (T+1, n_states)
        log_bwd_msgs: (T+1, n_states)
        log_obs_to_x: (T+1, n_states)
        log_action_prior: (n_actions,)
        horizon: T
        n_theta_axes: number of θ axes in the kernel (default 1).

    Returns:
        log_dyn_to_theta: (T, *t_theta_shape) log-space messages.
            For single-component T: (T, n_static).
    """
    n_kernel_axes = log_dyn_kernels.ndim
    # log_dyn_kernels axes: 0=T, 1=x_old, 2=x_new, 3..2+n_theta_axes=θ, last=u.
    # Reshape forward / backward to broadcast: (T, x_old, 1, *(1,)*n_theta_axes, 1) etc.
    fwd_shape = (horizon, -1, 1) + (1,) * n_theta_axes + (1,)
    bwd_shape = (horizon, 1, -1) + (1,) * n_theta_axes + (1,)
    action_shape = (1, 1, 1) + (1,) * n_theta_axes + (-1,)

    log_fwd_t = (log_fwd_msgs[:-1] + log_obs_to_x[:-1]).reshape(fwd_shape)
    log_bwd_t1 = (log_bwd_msgs[1:] + log_obs_to_x[1:]).reshape(bwd_shape)
    log_action = log_action_prior.reshape(action_shape)

    terms = (log_dyn_kernels + log_fwd_t + log_bwd_t1 + log_action)
    # Sum over x_old (axis 1), x_new (axis 2), u (last axis); keep θ axes 3..2+n_theta_axes.
    reduce_axes = (1, 2, n_kernel_axes - 1)
    return logsumexp(terms, axis=reduce_axes)


def compute_obs_to_x_msgs(log_obs_kernels, log_cavity_obs, b_factor_specs):
    """Compute messages from observation factors to x variables (log-space).

    Args:
        log_obs_kernels: list of per-channel kernels, shape per spec:
          - θ-independent: (T+1, n_obs, n_states)
          - θ-dependent: (T+1, n_obs, n_states, *b_theta_shape)
        log_cavity_obs: tuple of per-channel cavities. None for θ-independent
            channels; (T+1, *b_theta_shape) for θ-dependent.
        b_factor_specs: tuple of FactorSpec, one per channel.

    Returns:
        log_obs_to_x: (T+1, n_states) log-normalized messages (product over channels).
    """
    T_plus_1 = log_obs_kernels[0].shape[0]
    n_states = log_obs_kernels[0].shape[2]
    log_obs_to_x = jnp.zeros((T_plus_1, n_states))

    for c, log_k_c in enumerate(log_obs_kernels):
        spec_c = b_factor_specs[c]
        if spec_c.is_theta_independent:
            log_per_c = logsumexp(log_k_c, axis=1)  # (T+1, n_states)
        else:
            n_b_theta = spec_c.n_components
            cavity_c = log_cavity_obs[c]   # (T+1, *b_theta_shape)
            cavity_aligned = cavity_c.reshape(
                (T_plus_1, 1, 1) + cavity_c.shape[1:]
            )
            terms = log_k_c + cavity_aligned
            theta_axes = tuple(range(3, 3 + n_b_theta))
            log_per_c = logsumexp(terms, axis=(1,) + theta_axes)

        log_obs_to_x = log_obs_to_x + log_per_c

    log_obs_to_x = log_obs_to_x - logsumexp(log_obs_to_x, axis=1, keepdims=True)
    return log_obs_to_x


def compute_obs_to_theta_msgs(log_obs_kernels, log_fwd_msgs, log_bwd_msgs,
                               log_obs_to_x, b_factor_specs,
                               log_extra_to_x=None):
    """Compute per-channel messages from observation factors to θ (log-space).

    When log_extra_to_x is provided (e.g. pref_to_x), it is included in the
    x belief so that the cavity correctly includes all non-obs messages.

    Args:
        log_obs_kernels: list of per-channel kernels (same shapes as
            compute_obs_to_x_msgs).
        log_fwd_msgs: (T+1, n_states)
        log_bwd_msgs: (T+1, n_states)
        log_obs_to_x: (T+1, n_states)
        b_factor_specs: tuple of FactorSpec, one per channel.
        log_extra_to_x: Optional (T+1, n_states) extra messages.

    Returns:
        log_obs_to_theta: tuple of per-channel messages.
            θ-independent channel: None (no message).
            θ-dependent channel: (T+1, *b_theta_shape).
    """
    log_x_belief = log_fwd_msgs + log_bwd_msgs
    if log_extra_to_x is not None:
        log_x_belief = log_x_belief + log_extra_to_x

    log_per_c_list = []
    for c, log_k_c in enumerate(log_obs_kernels):
        spec_c = b_factor_specs[c]
        if spec_c.is_theta_independent:
            log_per_c_list.append(None)
            continue

        n_b_theta = spec_c.n_components
        # log_k_c: (T+1, n_obs, n_states, *b_theta_shape)
        # log_x_belief: (T+1, n_states); broadcast as (T+1, 1, n_states, *(1,)*n_b_theta)
        T_plus_1, n_states = log_x_belief.shape
        x_belief_aligned = log_x_belief.reshape(
            (T_plus_1, 1, n_states) + (1,) * n_b_theta
        )
        terms = log_k_c + x_belief_aligned
        # Reduce obs (axis 1) and x (axis 2)
        log_msg_c = logsumexp(terms, axis=(1, 2))  # (T+1, *b_theta_shape)
        log_per_c_list.append(log_msg_c)

    return tuple(log_per_c_list)


def compute_obs_pref_to_x_msgs(log_C_obs, log_B_list, log_theta_belief_components,
                                b_factor_specs):
    """Compute state-space message from observation preferences.

    For obs preference C_obs(y) in channel k, the backward message is:
        p(x) ∝ sum_{y, θ_k_components} C_obs(y) · B(y|x, θ_k) · q(θ_k)

    where θ_k denotes the components channel k sees. Under factored q(θ),
    q(θ_k) is reconstructed from per-component marginals via outer product
    in log space.

    Args:
        log_C_obs: dict mapping channel index to (n_obs,) log-preferences.
        log_B_list: list of per-channel log B (per-factor narrow shapes).
        log_theta_belief_components: tuple of per-component log marginals
            (one (component_size,) array per θ component).
        b_factor_specs: tuple of FactorSpec, one per channel.

    Returns:
        log_msg: (n_states,) log-normalized message to state.
    """
    n_states = log_B_list[0].shape[1]
    log_msg = jnp.zeros(n_states)

    for ch, log_c in log_C_obs.items():
        spec_ch = b_factor_specs[ch]
        log_B_ch = log_B_list[ch]

        if spec_ch.is_theta_independent:
            # log_c: (n_obs,) -> (n_obs, 1); log_B_ch: (n_obs, n_states)
            terms = log_c[:, None] + log_B_ch
            log_msg = log_msg + logsumexp(terms, axis=0)
        else:
            n_b_theta = spec_ch.n_components
            # Build q(θ_components_of_ch) as outer product of per-component marginals.
            log_theta_for_ch = expand_per_component_to_factor(
                {k: log_theta_belief_components[k] for k in spec_ch.theta_components},
                spec_ch, leading_axes=0,
            )   # shape: (*b_theta_shape,)
            # Align to (n_obs, n_states, *b_theta_shape) for elementwise sum.
            log_c_aligned = log_c.reshape((-1, 1) + (1,) * n_b_theta)
            log_theta_aligned = log_theta_for_ch.reshape((1, 1) + log_theta_for_ch.shape)
            terms = log_c_aligned + log_B_ch + log_theta_aligned
            theta_axes = tuple(range(2, 2 + n_b_theta))
            log_msg = log_msg + logsumexp(terms, axis=(0,) + theta_axes)

    return log_msg - logsumexp(log_msg)


# =============================================================================
# Theta cavities
# =============================================================================


def compute_theta_cavities(
    log_prior_components,
    log_dyn_to_theta,
    t_factor_spec,
    log_obs_to_theta,
    b_factor_specs,
):
    """Compute per-factor cavity messages for θ under factored q(θ).

    For each factor f and each component k it sees, the per-component cavity
    is the sum (in log space) of all incoming messages to k EXCEPT f's own
    contribution. The factor's full cavity is the outer product of its
    per-component cavities (additive in log space — exact under independent
    components).

    For multiple obs channels, each channel's cavity excludes only its OWN
    message (not all observation messages at the channel's timestep). For
    dynamics, the cavity at timestep t excludes only the T-msg at t (other
    timesteps' T-msgs are included). This is correct loopy BP semantics.

    Args:
        log_prior_components: tuple of per-component log priors,
            shapes (component_sizes[k],) for each k.
        log_dyn_to_theta: (T, *t_theta_shape) log T-to-θ messages
            (per-T-factor narrow).
        t_factor_spec: FactorSpec for T.
        log_obs_to_theta: tuple of per-channel log obs-to-θ messages.
            None for θ-independent channels; (T+1, *b_theta_shape) for θ-dependent.
        b_factor_specs: tuple of FactorSpec, one per channel.

    Returns:
        log_cavity_dyn: (T, *t_theta_shape) per-T-timestep cavity, normalized
            over the θ axes.
        log_cavity_obs: tuple of per-channel cavities. None for θ-independent
            channels; (T+1, *b_theta_shape) for θ-dependent, normalized.
    """
    n_components = len(log_prior_components)

    # Step 1: per-component totals (sum messages from factors that see k).
    log_total_into = []
    for k in range(n_components):
        log_total = log_prior_components[k]
        if k in t_factor_spec.theta_components:
            dyn_marg = marginalize_to_component(
                log_dyn_to_theta, t_factor_spec, k, leading_axes=1
            )   # (T, c_k)
            log_total = log_total + dyn_marg.sum(axis=0)
        for c, b_spec in enumerate(b_factor_specs):
            if log_obs_to_theta[c] is None or k not in b_spec.theta_components:
                continue
            obs_marg = marginalize_to_component(
                log_obs_to_theta[c], b_spec, k, leading_axes=1
            )   # (T+1, c_k)
            log_total = log_total + obs_marg.sum(axis=0)
        log_total_into.append(log_total)

    # Step 2: dyn cavity at each timestep — per-component cavity at k =
    # total_into[k] - dyn-msg-at-t marginalized to k. Reconstruct cavity tensor
    # over T's θ axes via outer product (sum in log space).
    if t_factor_spec.is_theta_independent:
        # T factor doesn't see any θ — degenerate, return zeros.
        T_dim = log_dyn_to_theta.shape[0]
        log_cavity_dyn = jnp.zeros((T_dim,))
    else:
        per_comp_cavity_dyn = {}
        for k in t_factor_spec.theta_components:
            dyn_marg = marginalize_to_component(
                log_dyn_to_theta, t_factor_spec, k, leading_axes=1
            )   # (T, c_k)
            per_comp_cavity_dyn[k] = log_total_into[k][None, :] - dyn_marg
        log_cavity_dyn = expand_per_component_to_factor(
            per_comp_cavity_dyn, t_factor_spec, leading_axes=1,
        )   # (T, *t_theta_shape)
        n_t_theta = t_factor_spec.n_components
        theta_axes = tuple(range(1, 1 + n_t_theta))
        log_cavity_dyn = log_cavity_dyn - logsumexp(
            log_cavity_dyn, axis=theta_axes, keepdims=True
        )

    # Step 3: per-channel obs cavities.
    log_cavity_obs_list = []
    for c, b_spec in enumerate(b_factor_specs):
        if log_obs_to_theta[c] is None:
            log_cavity_obs_list.append(None)
            continue
        per_comp_cavity = {}
        for k in b_spec.theta_components:
            obs_marg = marginalize_to_component(
                log_obs_to_theta[c], b_spec, k, leading_axes=1
            )   # (T+1, c_k)
            per_comp_cavity[k] = log_total_into[k][None, :] - obs_marg
        log_cavity_c = expand_per_component_to_factor(
            per_comp_cavity, b_spec, leading_axes=1,
        )   # (T+1, *b_theta_shape)
        n_b_theta = b_spec.n_components
        theta_axes = tuple(range(1, 1 + n_b_theta))
        log_cavity_c = log_cavity_c - logsumexp(
            log_cavity_c, axis=theta_axes, keepdims=True
        )
        log_cavity_obs_list.append(log_cavity_c)

    return log_cavity_dyn, tuple(log_cavity_obs_list)


# =============================================================================
# Region beliefs
# =============================================================================


def compute_dyn_pair_marginal(log_dyn_kernels, log_fwd_msgs, log_bwd_msgs,
                               log_obs_to_x, log_cavity_dyn, log_action_prior,
                               n_theta_axes=1):
    """Pair marginal log q(x_old, u) per timestep — fused reduction.

    Mathematically: logsumexp over (x_new, *θ) of the unnormalized region
    belief log_dyn_kernels + log_fwd_t + log_bwd_t1 + log_cavity + log_action.
    Contracts x_new and the θ axes inside a single logsumexp instead of
    materializing the full (T, x_old, x_new, *t_theta_shape, u) region tensor
    — the latter scales as T·X²·(∏θ)·U and is pure overhead when the consumer
    only needs the (T, x_old, u) pair marginal (VBP's action-channel update).

    Args:
        log_dyn_kernels: (T, x_old, x_new, *t_theta_shape, u).
        log_fwd_msgs, log_bwd_msgs, log_obs_to_x: as in BP.
        log_cavity_dyn: (T, *t_theta_shape) per-T cavity over θ axes.
        log_action_prior: (n_actions,)
        n_theta_axes: number of θ axes (default 1).

    Returns:
        (T, x_old, u) unnormalized log pair marginal.
    """
    horizon = log_dyn_kernels.shape[0]
    fwd_shape = (horizon, -1, 1) + (1,) * n_theta_axes + (1,)
    bwd_shape = (horizon, 1, -1) + (1,) * n_theta_axes + (1,)
    action_shape = (1, 1, 1) + (1,) * n_theta_axes + (-1,)
    cavity_shape = (horizon, 1, 1) + log_cavity_dyn.shape[1:] + (1,)

    log_fwd_t = (log_fwd_msgs[:-1] + log_obs_to_x[:-1]).reshape(fwd_shape)
    log_bwd_t1 = (log_bwd_msgs[1:] + log_obs_to_x[1:]).reshape(bwd_shape)
    log_action = log_action_prior.reshape(action_shape)
    log_cavity = log_cavity_dyn.reshape(cavity_shape)

    terms = log_dyn_kernels + log_fwd_t + log_bwd_t1 + log_cavity + log_action
    reduce_axes = (2,) + tuple(range(3, 3 + n_theta_axes))
    return logsumexp(terms, axis=reduce_axes)
