"""Information-Directed Sampling (IDS) for the parametric BA-POMDP.

Russo & Van Roy 2018 (bandit IDS); Lu & Van Roy 2019 (RL-IDS).
Information-theoretic exploration: pick the action minimising the
information ratio ``Δ(a)² / I(a)``, where ``Δ(a) = max_a' Q(a') - Q(a)``
is the regret proxy and ``I(a) = E_{o ~ p(o|a)}[KL(q(θ|o,a) || q(θ))]``
is the expected information gain about θ from observing the action's
outcome.

IDS is the canonical non-AIF baseline that explicitly optimises
information gain — making it the methodological foil for AIF's
epistemic-value term. Posterior-sampling baselines (PSRL, BAMCP, BOSS)
explore via random θ samples; IDS exploits info gain directly.

This planner reuses BAMCPPlanner for the Q(a) estimate (BAMCP already
marginalises T over q(θ) at the root, which is the right pragmatic
target). I(a) is computed in closed form by enumerating joint
observations over θ-dependent channels.

Action selection is deterministic argmin over the ratio (Lu & Van Roy
2019 RL convention). The variant ``Δ² − λI`` and Russo–Van Roy 2018's
randomised 2-action support are deliberate omissions; add only if
reviewers ask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
import numpy as np

from learningpomdp.planning.bamcp_planner import BAMCPPlanner

if TYPE_CHECKING:
    from learningpomdp.factors import FactorSpec, ThetaSpec
    from learningpomdp.observations import ModalityRegistry


EPSILON = 1e-12


def _align_full_components_np(
    arr: np.ndarray,
    factor_spec: "FactorSpec",
    theta_spec: "ThetaSpec",
    n_leading: int,
) -> np.ndarray:
    """NumPy mirror of ``learningpomdp.factors.align_to_full_components``.

    Input arr shape: ``(*leading[:n_leading], *factor.shape_for(θ))``.
    Output shape: ``(*leading[:n_leading], *component_sizes)`` where
    axes for components the factor doesn't see are size-1 (broadcast).
    """
    factor_components = factor_spec.theta_components
    full_shape = list(arr.shape[:n_leading])
    for k in range(theta_spec.n_components):
        if k in factor_components:
            full_shape.append(theta_spec.component_sizes[k])
        else:
            full_shape.append(1)

    sorted_components = tuple(sorted(factor_components))
    if factor_components == sorted_components:
        return arr.reshape(full_shape)
    sort_perm = [factor_components.index(c) for c in sorted_components]
    full_perm = list(range(n_leading)) + [n_leading + p for p in sort_perm]
    return arr.transpose(full_perm).reshape(full_shape)


class IDSPlanner:
    """IDS planner: deterministic argmin over Δ²/(I+ε).

    Args:
        bamcp: BAMCPPlanner instance used as the Q-value oracle. It
            must accept ``return_q_values=True`` (added by this
            milestone). All BAMCP knobs (n_simulations, depth, ucb_c,
            gamma) are inherited from this instance.
        epsilon_info: ε in the information ratio. With ε > 0 the ratio
            is finite when ``I(a) = 0``; argmin then reduces to
            Q-greedy in the all-zero-info limit. Default 1e-6.
    """

    def __init__(
        self,
        bamcp: BAMCPPlanner,
        epsilon_info: float = 1e-6,
    ) -> None:
        self._bamcp = bamcp
        self.epsilon_info = float(epsilon_info)

    # ----- runner hook (IDS is stateless across episodes) -----

    def prepare_episode(
        self,
        q_static: jnp.ndarray,
        rng: np.random.Generator,
        theta_spec: "Optional[ThetaSpec]" = None,
    ) -> None:
        # No persistent state. Forward to BAMCP in case it grows one.
        self._bamcp.prepare_episode(q_static, rng)

    # ----- main planning -----

    def plan(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list,
        goal,
        horizon: int,
        theta_mask=None,
        theta_spec: "Optional[ThetaSpec]" = None,
        t_factor_spec: "Optional[FactorSpec]" = None,
        b_factor_specs: "Optional[tuple[FactorSpec, ...]]" = None,
        registry: "Optional[ModalityRegistry]" = None,
        reward_fn: Optional[Callable[..., float]] = None,
        terminal_fn: Optional[Callable[..., bool]] = None,
        reward_table: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        # (a) Q(a) from BAMCP. One tree-search call.
        q_values, _ = self._bamcp.plan(
            q_state, q_static, T, B_list, goal, horizon,
            theta_mask=theta_mask,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
            registry=registry,
            reward_fn=reward_fn,
            terminal_fn=terminal_fn,
            reward_table=reward_table,
            return_q_values=True,
        )
        q_values = np.asarray(q_values, dtype=np.float64)
        n_actions = q_values.shape[0]

        # (b) I(a) closed-form over discrete observations.
        info_gain = self._info_gain_per_action(
            q_state, q_static, T, B_list, n_actions,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
        )

        # (c) Information ratio + argmin.
        delta = q_values.max() - q_values
        ratio = (delta ** 2) / (info_gain + self.epsilon_info)
        a_star = int(np.argmin(ratio))

        dist = np.zeros(n_actions, dtype=np.float64)
        dist[a_star] = 1.0
        return jnp.asarray(dist, dtype=jnp.float32)

    # ----- info-gain computation -----

    def _info_gain_per_action(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list,
        n_actions: int,
        theta_spec: "Optional[ThetaSpec]" = None,
        t_factor_spec: "Optional[FactorSpec]" = None,
        b_factor_specs: "Optional[tuple[FactorSpec, ...]]" = None,
    ) -> np.ndarray:
        """Compute ``I(a) = E_o[KL(q(θ|o,a) || q(θ))]`` per action.

        Vectorized across the inner joint observation space, chunked
        over the highest-cardinality channel (e.g. RockSample's
        ``position`` with card = grid_size²) so the per-action
        intermediate fits in RAM at scale. Each B channel is lifted
        once to its native θ-tail; the product across non-outer
        channels is precomputed; for each outer obs value the inner
        marginals and KL are evaluated in one vectorized pass.

        θ-independent channels stay in the enumeration: under
        θ-dependent ``T`` the predictive p(x_new | a, θ) is a function
        of θ, so an observation that pins down x_new (even through a
        θ-indep channel) still moves q(θ).
        """
        T_np = np.asarray(T, dtype=np.float64)
        q_state_np = np.asarray(q_state, dtype=np.float64)
        q_state_np = q_state_np / (q_state_np.sum() + EPSILON)
        n_states = q_state_np.shape[0]

        if theta_spec is not None:
            component_sizes = theta_spec.component_sizes
        else:
            component_sizes = (int(np.prod(q_static.shape)),)
        n_components = len(component_sizes)
        q_static_np = np.asarray(q_static, dtype=np.float64).reshape(component_sizes)
        q_static_np = q_static_np / (q_static_np.sum() + EPSILON)
        log_q_static = np.log(q_static_np + EPSILON)  # (*component_sizes,)

        # Channel catalogue.
        any_theta_dep_B = False
        channels: list[tuple[np.ndarray, bool]] = []
        for c, B_c in enumerate(B_list):
            B_c_np = np.asarray(B_c, dtype=np.float64)
            if b_factor_specs is not None:
                is_indep = b_factor_specs[c].is_theta_independent
            else:
                is_indep = B_c_np.ndim <= 2
            if not is_indep:
                any_theta_dep_B = True
            channels.append((B_c_np, is_indep))

        t_is_indep = (
            t_factor_spec is not None and t_factor_spec.is_theta_independent
        ) or T_np.ndim == 3

        # With θ-indep T AND no θ-dep B no observation can move q(θ),
        # so I(a) ≡ 0 — short-circuit before lifting B.
        if t_is_indep and not any_theta_dep_B:
            return np.zeros(n_actions, dtype=np.float64)

        # Pre-build predicted x marginal.
        if t_is_indep:
            # T (n_states_new, n_states_old, n_actions).
            predicted_x_a = np.tensordot(T_np, q_state_np, axes=([1], [0]))
            # shape (n_states_new, n_actions)
        else:
            # T (n_states_new, n_states_old, *t_theta, n_actions).
            predicted_x_theta_a = np.tensordot(T_np, q_state_np, axes=([1], [0]))
            # shape (n_states_new, *t_theta, n_actions)

        # Lift each channel B to (card_c, n_states, *θ-tail), where the
        # θ tail has size-1 for components this channel doesn't see and
        # full size for components it does. θ-dep channels use
        # _align_full_components_np with n_leading=2 (obs + state).
        B_lifted: list[np.ndarray] = []
        channel_cards: list[int] = []
        n_channels = len(channels)
        for ci, (B_c, is_indep) in enumerate(channels):
            card_c = B_c.shape[0]
            channel_cards.append(card_c)
            if is_indep:
                B_full = B_c.reshape((card_c, n_states) + (1,) * n_components)
            else:
                B_full = _align_full_components_np(
                    B_c, b_factor_specs[ci], theta_spec, n_leading=2,
                )
            B_lifted.append(B_full)

        # Pick the largest-cardinality channel as the outer chunking
        # axis. For RockSample this is ``position`` (card=grid_size²),
        # which would otherwise multiply the peak per-action tensor by
        # 25× and blow past RAM at K≥4.
        outer_ci = int(np.argmax(channel_cards)) if channel_cards else 0
        outer_card = channel_cards[outer_ci] if channel_cards else 1
        inner_cis = [c for c in range(n_channels) if c != outer_ci]
        n_inner = len(inner_cis)

        # Pre-compute B_inner = Π_{c ∈ inner} B_lifted[c], with each
        # inner channel's obs axis at a distinct leading position.
        # Native θ tails (size-1 for θ-indep, full-size for θ-dep)
        # broadcast against each other for free.
        if n_inner == 0:
            # Single channel total — B_inner is the identity over a
            # length-1 outer "joint" with no inner axes.
            B_inner = np.ones(
                (n_states,) + (1,) * n_components, dtype=np.float64,
            )
        else:
            first = inner_cis[0]
            shape0 = [1] * n_inner
            shape0[0] = channel_cards[first]
            B_inner = B_lifted[first].reshape(
                tuple(shape0) + B_lifted[first].shape[1:]
            )
            for pos_in_inner, ci in enumerate(inner_cis[1:], start=1):
                shape = [1] * n_inner
                shape[pos_in_inner] = channel_cards[ci]
                B_inner = B_inner * B_lifted[ci].reshape(
                    tuple(shape) + B_lifted[ci].shape[1:]
                )
        # B_inner shape: (*inner_cards, n_states, *θ-tail-inner-combined).

        # Per-action loop: build predicted_joint, then sweep over the
        # outer channel's obs values and accumulate info_gain.
        log_q_static_b = log_q_static.reshape(
            (1,) * n_inner + component_sizes
        )
        info_gain = np.zeros(n_actions, dtype=np.float64)
        B_outer_full = B_lifted[outer_ci] if channel_cards else None

        for a in range(n_actions):
            # predicted_joint[x, *θ] = p(x_new, θ | a, beliefs).
            if t_is_indep:
                shape_x = (n_states,) + (1,) * n_components
                shape_t = (1,) + component_sizes
                predicted_joint = (
                    predicted_x_a[:, a].reshape(shape_x)
                    * q_static_np.reshape(shape_t)
                )
            else:
                slice_a = predicted_x_theta_a[..., a]
                predicted_aligned = _align_full_components_np(
                    slice_a, t_factor_spec, theta_spec, n_leading=1,
                )
                predicted_joint = (
                    predicted_aligned
                    * q_static_np.reshape((1,) + component_sizes)
                )
            predicted_joint = predicted_joint / (predicted_joint.sum() + EPSILON)
            # Shape: (n_states, *component_sizes).
            pj_inner_b = predicted_joint.reshape(
                (1,) * n_inner + (n_states,) + component_sizes
            )

            for o_outer in range(outer_card):
                # B_outer_slice shape (n_states, *outer_θ_tail).
                if B_outer_full is not None:
                    B_outer_slice = B_outer_full[o_outer]
                    B_outer_b = B_outer_slice.reshape(
                        (1,) * n_inner + B_outer_slice.shape
                    )
                    B_full = B_inner * B_outer_b
                else:
                    B_full = B_inner

                # unnorm[*inner_obs, x, *θ] = B_full · predicted_joint.
                unnorm = B_full * pj_inner_b
                state_axis = n_inner
                theta_axes = tuple(range(n_inner + 1, n_inner + 1 + n_components))

                p_o = unnorm.sum(axis=(state_axis,) + theta_axes)
                q_theta_unnorm = unnorm.sum(axis=state_axis)

                valid = p_o >= EPSILON
                safe_p_o = np.where(valid, p_o, 1.0)
                q_theta_new = q_theta_unnorm / safe_p_o.reshape(
                    p_o.shape + (1,) * n_components
                )
                log_q_theta_new = np.log(q_theta_new + EPSILON)
                kl_per = (
                    q_theta_new * (log_q_theta_new - log_q_static_b)
                ).sum(axis=tuple(range(n_inner, n_inner + n_components)))
                # Clamp float noise; zero invalid obs (matches the old
                # `if p_o < EPSILON: continue` and `if kl < 0: kl = 0`).
                kl_per = np.maximum(kl_per, 0.0)
                kl_per = np.where(valid, kl_per, 0.0)

                info_gain[a] += (p_o * kl_per).sum()

        return info_gain
