"""Tabular Bayes-adaptive Q-learning baseline.

A standard belief-state RL competitor: Q-table over a coarse summary
of (physical state, agent's belief over theta), updated via temporal-
difference learning with epsilon-greedy exploration. State inference
and q(theta) tracking remain in the runner — this learner only consumes
the belief and emits actions.

Belief feature (low-cardinality so the Q-table is genuinely tabular):
- physical state index (n_states = M + 3)
- MAP pi(0) under q(theta), i.e., argmax over doors d of
  sum_{theta : pi_theta(0) = d} q(theta)  (K values; argmax-tie-broken)
- entropy bin: 0=low (H < log K!/3), 1=medium, 2=high

Total Q-table size: n_states * K * 3 * (M+K). For K=4, M=5: 8*4*3*9 = 864.

Implements the same duck-typed hook interface as PSRLPlanner so the
existing runner (door_code_runner.run_episode) drives it without code
changes:

- prepare_episode(q_static, rng): clears pending TD update.
- plan(...): returns a one-hot action distribution; the runner's
  argmax-when-deterministic action selector recovers the chosen
  action.
- post_step_update(q_state_new, q_static_new, reward, terminated):
  applies the TD update.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from learningpomdp.environments.door_code import enumerate_permutations


N_ENTROPY_BINS = 3


class BeliefStateQLearner:
    """Tabular Q-learning over a discretized belief feature.

    Args:
        K: number of doors / symbols (used for the MAP-pi(stage)
            feature dimension; independent of env state space).
        M: number of probe stations (used as a default for n_states
            when not overridden, matching the door-code shape).
        L: lock length.
        n_states: override env state-space size. Defaults to L + M + 2
            (door_code.py shape). Pass the actual env n_states for
            envs with a different state space (e.g. spatial minigrid).
        n_actions: override env action-space size. Defaults to M + K
            (door_code.py shape). Pass the actual env n_actions for
            envs with a different action space.
        lr: TD learning rate.
        epsilon: exploration probability for epsilon-greedy.
        gamma: discount factor.
        rng_seed: seed for the learner's exploration RNG (independent
            of the runner's action-sampling RNG so this learner stays
            reproducible regardless of how the runner is configured).
    """

    def __init__(
        self,
        K: int,
        M: int,
        L: int = 1,
        n_states: int | None = None,
        n_actions: int | None = None,
        lr: float = 0.1,
        epsilon: float = 0.2,
        gamma: float = 0.99,
        rng_seed: int = 0,
    ) -> None:
        self.K = K
        self.M = M
        self.L = L
        self.n_states = n_states if n_states is not None else L + M + 2
        self.n_actions = n_actions if n_actions is not None else M + K
        self.n_static = math.factorial(K)
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma

        self._perms = enumerate_permutations(K)
        # pi(stage) value for each theta_idx, for stage in [0, L) — used
        # to compute MAP pi(stage) per cell of the lock.
        self._pi_for_theta = np.array(
            [[p[s] for s in range(L)] for p in self._perms],
            dtype=np.int32,
        )  # shape (n_static, L)
        self._max_entropy = math.log(self.n_static)

        # Q-table shape: n_states * K (MAP pi(0)) * K (MAP pi(1)) * ... * K (MAP pi(L-1)) * N_ENTROPY_BINS * n_actions
        # For L=1 this is the original (n_states, K, N_ENTROPY_BINS, n_actions).
        # For L=2, K=4: (9, 4, 4, 3, 9) = 3888 entries. Tabular.
        q_shape = (self.n_states,) + (K,) * L + (N_ENTROPY_BINS, self.n_actions)
        self.Q = np.zeros(q_shape, dtype=np.float64)
        self._rng = np.random.default_rng(rng_seed)
        self._pending: tuple | None = None
        # _pending shape: (state, map_pi0, map_pi1, ..., map_pi{L-1}, ent_bin, action)

    # ----- belief feature -----

    def _belief_feature(
        self, x_idx: int, q_static: np.ndarray,
    ) -> tuple[int, ...]:
        # MAP pi(stage) for stage in [0, L). For each stage, sum the
        # probability mass over thetas grouped by their pi(stage) value
        # and pick the argmax (ties broken at lowest index).
        feature: list[int] = [x_idx]
        for s in range(self.L):
            p_pi_s = np.zeros(self.K, dtype=np.float64)
            for d in range(self.K):
                mask = self._pi_for_theta[:, s] == d
                p_pi_s[d] = q_static[mask].sum()
            feature.append(int(p_pi_s.argmax()))

        # Entropy bin.
        safe = q_static[q_static > 0]
        h = float(-np.sum(safe * np.log(safe))) if safe.size > 0 else 0.0
        frac = h / self._max_entropy
        if frac < 1.0 / 3.0:
            ent_bin = 0
        elif frac < 2.0 / 3.0:
            ent_bin = 1
        else:
            ent_bin = 2
        feature.append(ent_bin)
        return tuple(feature)

    # ----- runner hooks -----

    def prepare_episode(
        self,
        q_static: jnp.ndarray,
        rng: np.random.Generator,
    ) -> None:
        # Reset the pending TD step at the episode boundary so we don't
        # carry a stale (s, a) over a reset.
        self._pending = None

    def plan(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list,
        goal,
        horizon: int,
        theta_mask=None,
        theta_spec=None,
        t_factor_spec=None,
        b_factor_specs=None,
    ) -> jnp.ndarray:
        # Sample x_idx ~ q_state. Identity B_position envs (door_code)
        # produce a one-hot q_state, so this collapses to argmax;
        # many-to-one B_position (minigrid) leaves q_state multimodal,
        # and sampling avoids picking an arbitrary aliased mode.
        q_state_np = np.asarray(q_state, dtype=np.float64)
        q_state_np = q_state_np / q_state_np.sum()
        x_idx = int(self._rng.choice(q_state_np.shape[0], p=q_state_np))
        # q_static may arrive flat or rank-K; the belief feature reads it
        # as flat (mass grouped by per-stage permutation map).
        q_static_np = np.asarray(q_static, dtype=np.float64).reshape(-1)
        feature = self._belief_feature(x_idx, q_static_np)

        if self._rng.random() < self.epsilon:
            action = int(self._rng.integers(0, self.n_actions))
        else:
            action = int(np.argmax(self.Q[feature]))

        self._pending = feature + (action,)

        action_dist = np.zeros(self.n_actions, dtype=np.float32)
        action_dist[action] = 1.0
        return jnp.asarray(action_dist)

    def post_step_update(
        self,
        q_state_new: jnp.ndarray,
        q_static_new: jnp.ndarray,
        reward: float,
        terminated: bool,
    ) -> None:
        if self._pending is None:
            return
        # Match plan()'s sampling treatment so the bootstrap feature is
        # drawn from the same distribution as the on-policy feature.
        q_state_np_new = np.asarray(q_state_new, dtype=np.float64)
        q_state_np_new = q_state_np_new / q_state_np_new.sum()
        x_idx_new = int(self._rng.choice(q_state_np_new.shape[0], p=q_state_np_new))
        q_static_np_new = np.asarray(q_static_new, dtype=np.float64).reshape(-1)
        next_feature = self._belief_feature(x_idx_new, q_static_np_new)
        max_next_q = 0.0 if terminated else float(self.Q[next_feature].max())
        target = float(reward) + self.gamma * max_next_q
        prev_q = float(self.Q[self._pending])
        self.Q[self._pending] = prev_q + self.lr * (target - prev_q)
        # Clear the pending step; runner calls prepare_episode at the
        # next episode boundary, but if it doesn't (single-episode
        # runs), this also keeps state clean.
        self._pending = None
