"""BAMCP / BA-POMCP — sample-based Bayes-Adaptive tree search.

Guez, Silver & Dayan 2012 (BAMCP); Katt, Oliehoek & Amato 2017
(BA-POMCP). Online Monte-Carlo tree search where each simulation
samples theta ~ q(theta) at the root and forward-simulates with that
fixed theta in the rollout. For our parametric BA-POMDP the only
persistent latent is theta and the inner state x is well-localised
by q_state, so the BAMCP and BA-POMCP recipes coincide.

Specialised for the BA-POMDP setting:

- The "particle filter" at the root is replaced by the agent's exact
  q(theta) (we have closed-form posterior because the env transitions
  / observations are tabular).
- Each simulation samples a theta ~ q(theta) (BAMCP root resample),
  then forward-simulates the env using the known T(x'|x,theta,u) and
  B(y|x,theta).
- UCB1 picks actions during tree expansion; depth-limited random
  rollouts evaluate leaves.
- Returns are propagated up the tree; after `n_simulations` rollouts
  the root action with the highest visit count is returned.

Same generative model and online setting as AIF/PSRL/VBP/BeliefQ. No
pre-training. Direct sample-based competitor to AIF.

Literal-POMCP ablation: with `resample_theta_per_sim=False`, theta is
sampled once per `plan()` call and reused across all simulations.
This drops the BAMCP-specific theta-aware root resample while keeping
the same MCTS machinery — it isolates the contribution of the
theta-aware sampling step from the contribution of tree search alone.

Notes:
- Successors are sampled from T(.|x, theta, a). For envs with one-hot T
  (door_code) this collapses to argmax; for envs with stochastic T
  (minigrid_door_code with slip_prob>0) this is required for correct
  rollouts.
- The tree history key uses the *view-index* emitted by B_position, not
  the latent next_state. door_code has identity B_position so view ==
  state and behavior is unchanged; minigrid has many-to-one B_position
  so siblings sharing a view collapse to the same branch (matching what
  the agent actually observes).
- The reward signal is the same SUCCESS=+1 / FAILURE=-1 / NEUTRAL=0 we
  already use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from learningpomdp.factors import FactorSpec, ThetaSpec
    from learningpomdp.observations import ModalityRegistry


@dataclass
class _Node:
    """Tree node keyed by (action, observation) history.

    For a discrete observation space, children are stored by (action,
    obs_idx) pairs.
    """

    visit_count: int = 0
    total_value: float = 0.0
    action_visits: dict[int, int] = field(default_factory=dict)
    action_values: dict[int, float] = field(default_factory=dict)
    # children[(action, obs_idx)] = child Node
    children: dict[tuple[int, int], "_Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.total_value / max(self.visit_count, 1)

    def action_mean(self, a: int) -> float:
        n = self.action_visits.get(a, 0)
        if n == 0:
            return 0.0
        return self.action_values[a] / n

    def ucb_score(self, a: int, c: float) -> float:
        n_a = self.action_visits.get(a, 0)
        if n_a == 0:
            return float("inf")
        return self.action_mean(a) + c * math.sqrt(
            math.log(max(self.visit_count, 1)) / n_a
        )


class BAMCPPlanner:
    """Online MCTS for the parametric BA-POMDP.

    Args:
        n_simulations: simulations per call to plan() (root tree budget).
        depth: maximum tree+rollout depth (caps both expansion and
            rollout horizon).
        ucb_c: UCB1 exploration constant.
        gamma: discount factor for returns.
        rollout_seed: RNG seed for theta-sampling and probe noise during
            rollouts.
        resample_theta_per_sim: if True (default; BAMCP/BA-POMCP), sample
            theta ~ q(theta) at the start of every simulation. If False
            (literal-POMCP ablation), sample theta once per plan() call
            and reuse across all simulations — same MCTS machinery, no
            theta-aware root resample.
    """

    def __init__(
        self,
        n_simulations: int = 256,
        depth: int = 6,
        ucb_c: float = 1.0,
        gamma: float = 0.99,
        rollout_seed: int = 0,
        resample_theta_per_sim: bool = True,
        factored_root_sample: bool = False,
    ) -> None:
        self.n_simulations = n_simulations
        self.depth = depth
        self.ucb_c = ucb_c
        self.gamma = gamma
        self.resample_theta_per_sim = resample_theta_per_sim
        self.factored_root_sample = factored_root_sample
        self._rng = np.random.default_rng(rollout_seed)

    # ----- runner hooks (BAMCP is stateless across episodes) -----

    def prepare_episode(
        self, q_static: jnp.ndarray, rng: np.random.Generator,
    ) -> None:
        # No persistent state between episodes; each plan() call builds
        # a fresh tree.
        pass

    # ----- internal: forward-simulate one (state, action, theta) step ----

    @staticmethod
    def _step(
        state: int, theta: int, action: int,
        T: np.ndarray, B_position: np.ndarray,
        B_probe: np.ndarray, B_outcome: np.ndarray,
        probe_state_set: set[int], rng: np.random.Generator,
    ) -> tuple[int, int, float]:
        """Take one env step in the rollout. Returns (next_state,
        obs_idx, reward). obs_idx packs (view_idx, probe_obs,
        outcome_obs) where view_idx is what B_position emits for the
        sampled next_state — distinct latent states that share a view
        share a tree branch (correct under partial observability).
        """
        next_state_dist = T[:, state, theta, action]
        # Sample successor from T. For one-hot T this is argmax-equivalent;
        # for stochastic T (slip_prob>0) we need the actual sample.
        next_state = int(rng.choice(next_state_dist.shape[0], p=next_state_dist))

        # Probe channel: theta-conditional at probe states
        if next_state in probe_state_set:
            probe_dist = B_probe[:, next_state, theta]
            probe_obs = int(rng.choice(len(probe_dist), p=probe_dist))
        else:
            probe_obs = -1  # neutral, not observation-discriminating

        # Outcome channel: theta-independent
        outcome_dist = B_outcome[:, next_state]
        outcome_obs = int(np.argmax(outcome_dist))

        # Reward from the env's terminal map: outcome=1 -> +1, outcome=2 -> -1
        if outcome_obs == 1:
            reward = 1.0
        elif outcome_obs == 2:
            reward = -1.0
        else:
            reward = 0.0

        # Pack (view_idx, probe_obs, outcome_obs) into a single int for
        # the tree key. probe_obs in {-1,0,1} maps to {0,1,2}; outcome in
        # {0,1,2}. view_idx is the position-channel observation: identity
        # in door_code (so view_idx == next_state), many-to-one in
        # minigrid (so aliased states share a branch — what the agent
        # actually observes).
        view_idx = int(np.argmax(B_position[:, next_state]))
        probe_key = probe_obs + 1 if probe_obs >= 0 else 0
        obs_idx = view_idx * 9 + probe_key * 3 + outcome_obs
        return next_state, obs_idx, reward

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
        return_q_values: bool = False,
    ) -> jnp.ndarray:
        # Dispatch: door-code/minigrid use the legacy outcome-channel path
        # (regression-preserved when the runner doesn't pass reward_fn).
        # Rock-sample (or any env that exposes a reward_fn) uses the generic
        # path that takes registry-by-name and a callable reward.
        #
        # return_q_values=True (used by IDSPlanner) returns a
        # ``(q_values, dist)`` tuple — q_values is the per-action mean
        # return at the root (action_values / max(action_visits, 1)).
        # Default False preserves the existing (n_actions,) jnp.float32
        # return contract for all other callers.
        if reward_fn is None:
            return self._plan_legacy(
                q_state, q_static, T, B_list, horizon,
                return_q_values=return_q_values,
            )
        return self._plan_numba(
            q_state, q_static, T, B_list, horizon,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
            registry=registry,
            reward_fn=reward_fn,
            terminal_fn=terminal_fn,
            reward_table=reward_table,
            return_q_values=return_q_values,
        )

    # ----- legacy path: door-code / minigrid (theta-as-flat-int) -----

    def _plan_legacy(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list,
        horizon: int,
        return_q_values: bool = False,
    ) -> jnp.ndarray:
        T_np = np.asarray(T, dtype=np.float64)
        B_position = np.asarray(B_list[0], dtype=np.float64)
        B_probe = np.asarray(B_list[1], dtype=np.float64)
        B_outcome = np.asarray(B_list[2], dtype=np.float64)
        q_state_np = np.asarray(q_state, dtype=np.float64)
        q_static_np = np.asarray(q_static, dtype=np.float64).reshape(-1)
        # Renormalize to guard against float32 -> float64 sum drift; the
        # runner hands us jnp.float32 arrays, and rng.choice's tolerance
        # (~1.49e-8) is tighter than the worst-case round-trip error
        # (e.g. 0.9+0.1 in float32 -> 0.99999998 in float64).
        q_static_np = q_static_np / q_static_np.sum()
        q_state_np = q_state_np / q_state_np.sum()
        T_np = T_np / T_np.sum(axis=0, keepdims=True)
        B_probe = B_probe / B_probe.sum(axis=0, keepdims=True)

        n_states = q_state_np.shape[0]
        n_actions = T_np.shape[3]
        n_static = q_static_np.shape[0]

        probe_state_set = {
            s for s in range(T_np.shape[0])
            if not np.allclose(B_probe[2, s, :], 1.0)
        }

        max_depth = min(self.depth, max(horizon, 1))

        theta_fixed = (
            None
            if self.resample_theta_per_sim
            else int(self._rng.choice(n_static, p=q_static_np))
        )
        root_state_fixed = (
            None
            if self.resample_theta_per_sim
            else int(self._rng.choice(n_states, p=q_state_np))
        )

        root = _Node()
        for _ in range(self.n_simulations):
            if self.resample_theta_per_sim:
                theta = int(self._rng.choice(n_static, p=q_static_np))
                root_state = int(self._rng.choice(n_states, p=q_state_np))
            else:
                theta = theta_fixed
                root_state = root_state_fixed
            self._simulate(
                root, root_state, theta, max_depth,
                T_np, B_position, B_probe, B_outcome,
                probe_state_set, n_actions,
            )

        visits = np.zeros(n_actions, dtype=np.float64)
        for a, n in root.action_visits.items():
            visits[a] = float(n)
        if visits.sum() > 0:
            dist = visits / visits.sum()
        else:
            dist = np.ones(n_actions) / n_actions
        dist_jnp = jnp.asarray(dist, dtype=jnp.float32)
        if return_q_values:
            q_values = np.zeros(n_actions, dtype=np.float64)
            for a in range(n_actions):
                n_a = root.action_visits.get(a, 0)
                if n_a > 0:
                    q_values[a] = root.action_values[a] / n_a
            return jnp.asarray(q_values, dtype=jnp.float32), dist_jnp
        return dist_jnp

    def _simulate(
        self,
        node: _Node,
        state: int,
        theta: int,
        depth: int,
        T: np.ndarray,
        B_position: np.ndarray,
        B_probe: np.ndarray,
        B_outcome: np.ndarray,
        probe_state_set: set[int],
        n_actions: int,
    ) -> float:
        if depth <= 0:
            return 0.0

        # UCB1 over actions: pick the unvisited action first; otherwise
        # the action maximizing UCB.
        unvisited = [a for a in range(n_actions) if a not in node.action_visits]
        if unvisited:
            action = int(self._rng.choice(unvisited))
        else:
            action = max(range(n_actions), key=lambda a: node.ucb_score(a, self.ucb_c))

        next_state, obs_idx, reward = self._step(
            state, theta, action, T, B_position, B_probe, B_outcome,
            probe_state_set, self._rng,
        )

        # Terminal check: outcome observation 1 (success) or 2 (failure).
        outcome = obs_idx % 3
        if outcome == 1 or outcome == 2:
            value = reward
        else:
            child_key = (action, obs_idx)
            if child_key not in node.children:
                node.children[child_key] = _Node()
                # Random rollout from leaf (a la POMCP).
                rollout_value = self._rollout(
                    next_state, theta, depth - 1,
                    T, B_position, B_probe, B_outcome,
                    probe_state_set, n_actions,
                )
                value = reward + self.gamma * rollout_value
                # Bookkeeping for the new child? POMCP bookkeeps only the
                # path up to (and including) the new leaf; the rollout
                # itself doesn't expand the tree.
            else:
                child = node.children[child_key]
                future_value = self._simulate(
                    child, next_state, theta, depth - 1,
                    T, B_position, B_probe, B_outcome,
                    probe_state_set, n_actions,
                )
                value = reward + self.gamma * future_value

        # Update statistics for (node, action).
        node.visit_count += 1
        node.total_value += value
        node.action_visits[action] = node.action_visits.get(action, 0) + 1
        node.action_values[action] = node.action_values.get(action, 0.0) + value
        return value

    def _rollout(
        self,
        state: int,
        theta: int,
        depth: int,
        T: np.ndarray,
        B_position: np.ndarray,
        B_probe: np.ndarray,
        B_outcome: np.ndarray,
        probe_state_set: set[int],
        n_actions: int,
    ) -> float:
        """Random-action rollout to depth-`depth`."""
        total = 0.0
        gamma_t = 1.0
        for _ in range(depth):
            action = int(self._rng.integers(0, n_actions))
            next_state, obs_idx, reward = self._step(
                state, theta, action, T, B_position, B_probe, B_outcome,
                probe_state_set, self._rng,
            )
            total += gamma_t * reward
            outcome = obs_idx % 3
            if outcome == 1 or outcome == 2:
                return total
            state = next_state
            gamma_t *= self.gamma
        return total

    # ----- generic path: env-agnostic via registry + reward_fn -----

    def _batch_sample_root_thetas(
        self,
        q_static: jnp.ndarray,
        theta_spec: "ThetaSpec",
        n_samples: int,
    ) -> np.ndarray:
        """Sample ``n_samples`` root θ̃'s, returning flat θ indices.

        Persistent components are drawn from their (factored or joint)
        marginal; episodic components are drawn from the conditional
        ``q(θ_episodic | θ̃_persistent)``. All draws are vectorised over
        ``n_samples`` so the planner does not pay JAX-dispatch overhead
        per simulation.
        """
        q = np.asarray(q_static, dtype=np.float64).reshape(theta_spec.component_sizes)
        n_components = theta_spec.n_components
        persistent_axes = theta_spec.persistent_components
        episodic_axes = tuple(theta_spec.episode_reset_components)

        if not persistent_axes:
            p_flat = q.reshape(-1)
            total = p_flat.sum()
            p_flat = (p_flat / total) if total > 0 else (np.ones_like(p_flat) / p_flat.size)
            return self._rng.choice(p_flat.size, size=n_samples, p=p_flat).astype(np.int32)

        persistent_sizes = tuple(theta_spec.component_sizes[k] for k in persistent_axes)
        episodic_sizes = tuple(theta_spec.component_sizes[k] for k in episodic_axes)
        n_persistent_flat = int(np.prod(persistent_sizes)) if persistent_sizes else 1
        n_episodic_flat = int(np.prod(episodic_sizes)) if episodic_sizes else 1

        # 1. Persistent draws (per sim).
        if self.factored_root_sample:
            per_pers_idx = np.zeros((n_samples, len(persistent_axes)), dtype=np.int64)
            for ki, k in enumerate(persistent_axes):
                marg_axes = tuple(j for j in range(n_components) if j != k)
                q_k = q.sum(axis=marg_axes) if marg_axes else q
                p_k = q_k.reshape(-1)
                total = p_k.sum()
                p_k = (p_k / total) if total > 0 else (np.ones_like(p_k) / p_k.size)
                per_pers_idx[:, ki] = self._rng.choice(p_k.size, size=n_samples, p=p_k)
            persistent_flat_idx = np.ravel_multi_index(
                tuple(per_pers_idx[:, i] for i in range(len(persistent_axes))),
                persistent_sizes,
            ) if persistent_sizes else np.zeros(n_samples, dtype=np.int64)
        else:
            q_persistent = q.sum(axis=episodic_axes) if episodic_axes else q
            p = q_persistent.reshape(-1)
            total = p.sum()
            p = (p / total) if total > 0 else (np.ones_like(p) / p.size)
            persistent_flat_idx = self._rng.choice(p.size, size=n_samples, p=p)

        # 2. Episodic draws conditional on persistent — vectorised inverse-CDF.
        if not episodic_axes:
            episodic_flat_idx = np.zeros(n_samples, dtype=np.int64)
        else:
            perm = list(persistent_axes) + list(episodic_axes)
            q_perm = q.transpose(perm).reshape(n_persistent_flat, n_episodic_flat)
            row_sums = q_perm.sum(axis=1, keepdims=True)
            safe_sums = np.where(row_sums > 0, row_sums, 1.0)
            q_cond = np.where(
                row_sums > 0, q_perm / safe_sums, 1.0 / n_episodic_flat,
            )
            cdf = np.cumsum(q_cond, axis=1)
            sim_cdfs = cdf[persistent_flat_idx]
            us = self._rng.random(size=n_samples)
            # cdf[:, -1] == 1.0 by construction; us is in [0, 1) so argmax always hits.
            episodic_flat_idx = (sim_cdfs >= us[:, None]).argmax(axis=1)

        # 3. Combine into full θ-flat index in original component order.
        full_components = np.zeros((n_samples, n_components), dtype=np.int64)
        if persistent_sizes:
            pers_components = np.array(
                np.unravel_index(persistent_flat_idx, persistent_sizes)
            ).reshape(len(persistent_axes), n_samples).T
            for i, k in enumerate(persistent_axes):
                full_components[:, k] = pers_components[:, i]
        if episodic_sizes:
            epi_components = np.array(
                np.unravel_index(episodic_flat_idx, episodic_sizes)
            ).reshape(len(episodic_axes), n_samples).T
            for i, k in enumerate(episodic_axes):
                full_components[:, k] = epi_components[:, i]

        return np.ravel_multi_index(
            tuple(full_components[:, k] for k in range(n_components)),
            theta_spec.component_sizes,
        ).astype(np.int32)

    # ----- Numba path: env-agnostic via registry + reward_fn -----

    def _plan_numba(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list,
        horizon: int,
        theta_spec: "ThetaSpec",
        t_factor_spec: "FactorSpec",
        b_factor_specs: tuple,
        registry: "Optional[ModalityRegistry]",
        reward_fn: Callable[..., float],
        terminal_fn: Optional[Callable[..., bool]],
        reward_table: Optional[np.ndarray] = None,
        return_q_values: bool = False,
    ) -> jnp.ndarray:
        from learningpomdp.planning._bamcp_numba import _run_bamcp_numba

        T_np = np.asarray(T, dtype=np.float64)
        B_list_np = [np.asarray(B_c, dtype=np.float64) for B_c in B_list]
        q_state_np = np.asarray(q_state, dtype=np.float64)
        q_state_np = q_state_np / q_state_np.sum()

        n_states = q_state_np.shape[0]
        t_is_theta_independent = (
            t_factor_spec is not None and t_factor_spec.is_theta_independent
        ) or T_np.ndim == 3
        n_actions = T_np.shape[-1]

        pos_idx = (
            registry.get_channel_index("position") if registry is not None else 0
        )
        likelihood_indices = [i for i in range(len(B_list_np)) if i != pos_idx]
        n_likelihood = len(likelihood_indices)

        # T_eff: 3D (n_states_new, n_states_old, n_actions). RockSample's
        # T is θ-independent so we use T directly. The θ-dependent branch
        # marginalises T over q(θ) for the rollout — a conservative
        # approximation; M4 envs don't hit this branch.
        if t_is_theta_independent:
            T_eff = T_np / (T_np.sum(axis=0, keepdims=True) + 1e-30)
        else:
            q_static_flat = np.asarray(q_static, dtype=np.float64).reshape(-1)
            q_static_flat = q_static_flat / q_static_flat.sum()
            T_marg = np.einsum("ijka,k->ija", T_np, q_static_flat)
            T_eff = T_marg / (T_marg.sum(axis=0, keepdims=True) + 1e-30)

        B_pos = B_list_np[pos_idx]
        n_views = B_pos.shape[0]

        # B_lik_all: per-θ effective likelihood per channel, padded
        # along axis 2 to max_obs across channels.
        n_static = theta_spec.n_static
        if n_likelihood > 0:
            lik_obs_card = np.array(
                [B_list_np[c].shape[0] for c in likelihood_indices], dtype=np.int32,
            )
            max_obs = int(lik_obs_card.max())
        else:
            lik_obs_card = np.zeros(1, dtype=np.int32)
            max_obs = 1
        B_lik_all = np.zeros(
            (n_static, max(n_likelihood, 1), max_obs, n_states), dtype=np.float64,
        )
        for theta_flat in range(n_static):
            theta_components = theta_spec.unravel(theta_flat)
            for ci, c in enumerate(likelihood_indices):
                spec = b_factor_specs[c]
                B_c = B_list_np[c]
                if spec.is_theta_independent:
                    obs_dist = B_c
                else:
                    slice_indices = tuple(
                        theta_components[k] for k in spec.theta_components
                    )
                    obs_dist = B_c[(slice(None), slice(None), *slice_indices)]
                norm = obs_dist.sum(axis=0, keepdims=True)
                obs_dist = obs_dist / (norm + 1e-30)
                card = int(lik_obs_card[ci])
                B_lik_all[theta_flat, ci, :card, :] = obs_dist

        # Reward + terminal lookup tables.
        if reward_table is None:
            # Slow fallback — Python triple-loop call to reward_fn.
            # Runners should precompute reward_table and pass it via the
            # `reward_table` kwarg to avoid this O(n_states · n_actions ·
            # n_static) Python overhead per plan() call.
            reward_table = np.zeros((n_states, n_actions, n_static), dtype=np.float64)
            for theta_flat in range(n_static):
                theta_components = theta_spec.unravel(theta_flat)
                for s in range(n_states):
                    for a in range(n_actions):
                        reward_table[s, a, theta_flat] = float(reward_fn(
                            s, a, 0, theta_components,
                        ))
        else:
            reward_table = np.asarray(reward_table, dtype=np.float64)

        terminal_table = np.zeros((n_states, n_actions, n_static), dtype=np.bool_)
        has_terminal = terminal_fn is not None
        if terminal_fn is not None:
            for theta_flat in range(n_static):
                theta_components = theta_spec.unravel(theta_flat)
                for s in range(n_states):
                    for a in range(n_actions):
                        terminal_table[s, a, theta_flat] = bool(terminal_fn(
                            s, a, 0, theta_components,
                        ))

        max_depth = min(self.depth, max(horizon, 1))

        # Batch-sample θ̃ and root state for all simulations in one shot
        # (avoids 1024 JAX dispatches per plan() call).
        if self.resample_theta_per_sim:
            theta_samples = self._batch_sample_root_thetas(
                q_static, theta_spec, self.n_simulations,
            )
            root_states = self._rng.choice(
                n_states, size=self.n_simulations, p=q_state_np,
            ).astype(np.int32)
        else:
            fixed_theta = int(
                self._batch_sample_root_thetas(q_static, theta_spec, 1)[0]
            )
            fixed_state = int(self._rng.choice(n_states, p=q_state_np))
            theta_samples = np.full(self.n_simulations, fixed_theta, dtype=np.int32)
            root_states = np.full(self.n_simulations, fixed_state, dtype=np.int32)

        # obs_stride_total = n_views * prod(card_c).
        obs_stride_total = n_views
        for ci in range(n_likelihood):
            obs_stride_total *= int(lik_obs_card[ci])

        seed = int(self._rng.integers(0, 2**31 - 1))

        visits, totals = _run_bamcp_numba(
            self.n_simulations, max_depth, n_actions,
            theta_samples, root_states,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, has_terminal,
            float(self.ucb_c), float(self.gamma),
            n_likelihood, obs_stride_total,
            seed,
        )

        if visits.sum() > 0:
            dist = visits / visits.sum()
        else:
            dist = np.ones(n_actions) / n_actions
        dist_jnp = jnp.asarray(dist, dtype=jnp.float32)
        if return_q_values:
            q_values = np.where(visits > 0, totals / np.maximum(visits, 1.0), 0.0)
            return jnp.asarray(q_values, dtype=jnp.float32), dist_jnp
        return dist_jnp
