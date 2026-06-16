"""BOSS — Best of Sampled Set, as a merged-POMDP MCTS.

Asmuth, Li, Littman, Nouri, Wingate 2009. Bayesian RL via optimism in
the face of uncertainty: sample K MDPs from the posterior, plan
greedily on an "optimistic merged MDP" whose action space is augmented
to A x K, where (a, k) means "take real action a, simulate dynamics
T_k of the k-th sample." The Bellman backup is
    Q*(s, (a, k)) = R_k(s, a) + gamma * E_{T_k}[ max_{a', k'} Q*(s', (a', k')) ]
so the agent acts as if the world morphs into whichever sample is most
favourable for its current plan — that is the OFU representative in
the BARL slate.

This implementation realises the merged-MDP construction as MCTS,
sharing the tree / UCB / theta-conditional `_step` machinery with
BAMCP. The differences vs BAMCP:

- K thetas are fixed at episode start (sampled once in
  `prepare_episode`), instead of resampled at the root of each
  simulation.
- At every node, the action set is A x K (augmented). UCB1 selects
  over (a, k) pairs.
- The forward `_step` uses the chosen sample's theta_k to drive
  T(.|x, theta_k, a), B_probe(.|x', theta_k), B_outcome(.|x').
- Tree branches on (real action a, observation key) — k is internal
  to the planner and not part of the tree key, so different sample
  choices that produce the same observation share children. This is
  the deliberate design choice for OFU: optimism lives in the
  per-(a, k) Q statistics at each node, not in the tree branching.
- At the root, the returned action distribution marginalises over k:
  p(a) ~ sum_k visits(a, k).

n_samples=1 reduces to a single-sample MCTS with a fixed theta for
the episode (PSRL-flavoured, but with search instead of analytic
planning).

Same generative model and online setting as AIF/PSRL/VBP/BeliefQ/BAMCP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
import numpy as np

from learningpomdp.inference.state_inference import sample_persistent_theta
from learningpomdp.planning.bamcp_planner import BAMCPPlanner

if TYPE_CHECKING:
    from learningpomdp.factors import FactorSpec, ThetaSpec
    from learningpomdp.observations import ModalityRegistry


@dataclass
class _Node:
    """Tree node for the merged-POMDP MCTS.

    `action_visits` and `action_values` are keyed by the augmented
    action tuple (a, k) -- "take real action a, simulate dynamics of
    sample k". Children are keyed by (a, obs_idx) -- shared across k
    because the real env's downstream observation distribution depends
    on a and the latent state, not on the planner's choice of k.
    """

    visit_count: int = 0
    total_value: float = 0.0
    action_visits: dict[tuple[int, int], int] = field(default_factory=dict)
    action_values: dict[tuple[int, int], float] = field(default_factory=dict)
    children: dict[tuple[int, int], "_Node"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.total_value / max(self.visit_count, 1)

    def action_mean(self, ak: tuple[int, int]) -> float:
        n = self.action_visits.get(ak, 0)
        if n == 0:
            return 0.0
        return self.action_values[ak] / n

    def ucb_score(self, ak: tuple[int, int], c: float) -> float:
        n_ak = self.action_visits.get(ak, 0)
        if n_ak == 0:
            return float("inf")
        return self.action_mean(ak) + c * math.sqrt(
            math.log(max(self.visit_count, 1)) / n_ak
        )


class BOSSPlanner:
    """Faithful BOSS as a merged-POMDP MCTS.

    Args:
        n_samples: number of theta samples per episode (the K of BOSS).
            n=1 reduces to a fixed-theta MCTS for the episode.
        n_simulations: MCTS rollouts per call to plan().
        depth: max tree+rollout depth (caps both expansion and rollout).
        ucb_c: UCB1 exploration constant (over (a, k) pairs).
        gamma: discount factor for returns.
        rollout_seed: RNG seed for theta-sampling and forward sims.
    """

    def __init__(
        self,
        n_samples: int = 4,
        n_simulations: int = 256,
        depth: int = 6,
        ucb_c: float = 1.0,
        gamma: float = 0.99,
        rollout_seed: int = 0,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self.n_samples = n_samples
        self.n_simulations = n_simulations
        self.depth = depth
        self.ucb_c = ucb_c
        self.gamma = gamma
        self._rng = np.random.default_rng(rollout_seed)
        self._sampled_thetas: list[int] | None = None

    @property
    def current_samples(self) -> list[int] | None:
        return None if self._sampled_thetas is None else list(self._sampled_thetas)

    # ----- runner hooks -----

    def prepare_episode(
        self,
        q_static: jnp.ndarray,
        rng: np.random.Generator,
        theta_spec: "Optional[ThetaSpec]" = None,
    ) -> list[int]:
        """Sample n_samples persistent-only thetas ~ q(θ_persistent) and
        freeze them for the episode.

        When ``theta_spec`` declares episodic components, we sample
        θ̃_persistent ~ q(θ_persistent) and θ̃_episodic ~ q(θ_ep | θ̃_p)
        — faithful to BOSS's "sample N posterior thetas" recipe adapted
        to the BA-POMDP framing where only persistent θ is the BA
        parameter. Door-code (no episodic components) hits the legacy
        flat-sample branch byte-identically.

        Returns the sampled list of flat θ̃ indices for inspection /
        logging.
        """
        if theta_spec is None or not theta_spec.episode_reset_components:
            p = np.asarray(q_static, dtype=np.float64).reshape(-1)
            s = p.sum()
            if s <= 0:
                raise ValueError("q_static is not a valid distribution (sum <= 0).")
            p = p / s
            self._sampled_thetas = [
                int(rng.choice(len(p), p=p)) for _ in range(self.n_samples)
            ]
            return list(self._sampled_thetas)

        # Persistent-only path: sample persistent then episodic conditional.
        sampled: list[int] = []
        for _ in range(self.n_samples):
            q_cond, _ = sample_persistent_theta(q_static, theta_spec, rng)
            q_cond_flat = np.asarray(q_cond, dtype=np.float64).reshape(-1)
            s = q_cond_flat.sum()
            if s <= 0:
                q_cond_flat = np.ones_like(q_cond_flat) / q_cond_flat.size
            else:
                q_cond_flat = q_cond_flat / s
            sampled.append(int(rng.choice(q_cond_flat.size, p=q_cond_flat)))
        self._sampled_thetas = sampled
        return list(sampled)

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
        if self._sampled_thetas is None:
            raise RuntimeError(
                "BOSSPlanner.prepare_episode() must be called before plan()."
            )
        if reward_fn is not None:
            return self._plan_numba(
                q_state, q_static, T, B_list, horizon,
                theta_spec=theta_spec,
                t_factor_spec=t_factor_spec,
                b_factor_specs=b_factor_specs,
                registry=registry,
                reward_fn=reward_fn,
                terminal_fn=terminal_fn,
                reward_table=reward_table,
            )
        T_np = np.asarray(T, dtype=np.float64)
        B_position = np.asarray(B_list[0], dtype=np.float64)
        B_probe = np.asarray(B_list[1], dtype=np.float64)
        B_outcome = np.asarray(B_list[2], dtype=np.float64)
        q_state_np = np.asarray(q_state, dtype=np.float64)
        # Renormalize (float32 -> float64 round-trip can drift; rng.choice
        # tolerance is ~1.49e-8). q_static is unused in plan() because
        # samples were drawn at episode start, but normalize T / B_probe
        # for consistency with BAMCP.
        q_state_np = q_state_np / q_state_np.sum()
        T_np = T_np / T_np.sum(axis=0, keepdims=True)
        B_probe = B_probe / B_probe.sum(axis=0, keepdims=True)

        n_states = q_state_np.shape[0]
        n_actions = T_np.shape[3]
        # Dedupe: identical samples collapse to identical dynamics, so
        # (a, k) and (a, k') are the same augmented action when
        # theta_k == theta_k'. Under sharp posterior all K samples
        # collapse to one and BOSS reduces to a single-sample MCTS
        # (BAMCP-equivalent). `current_samples` retains the multiplicity
        # for logging.
        thetas = sorted(set(self._sampled_thetas))
        K = len(thetas)

        probe_state_set = {
            s for s in range(T_np.shape[0])
            if not np.allclose(B_probe[2, s, :], 1.0)
        }

        max_depth = min(self.depth, max(horizon, 1))

        root = _Node()
        for _ in range(self.n_simulations):
            # Per-simulation belief sample: avoids collapsing a multimodal
            # q_state to a single mode (matters under aliased B_position).
            root_state = int(self._rng.choice(n_states, p=q_state_np))
            self._simulate(
                root, root_state, max_depth,
                T_np, B_position, B_probe, B_outcome, probe_state_set,
                n_actions, K, thetas,
            )

        # Marginalize visit counts over k: p(a) ~ sum_k visits(a, k).
        visits = np.zeros(n_actions, dtype=np.float64)
        for (a, _k), n in root.action_visits.items():
            visits[a] += float(n)
        if visits.sum() > 0:
            dist = visits / visits.sum()
        else:
            dist = np.ones(n_actions) / n_actions
        return jnp.asarray(dist, dtype=jnp.float32)

    def _simulate(
        self,
        node: _Node,
        state: int,
        depth: int,
        T: np.ndarray,
        B_position: np.ndarray,
        B_probe: np.ndarray,
        B_outcome: np.ndarray,
        probe_state_set: set[int],
        n_actions: int,
        K: int,
        thetas: list[int],
    ) -> float:
        if depth <= 0:
            return 0.0

        # UCB1 over augmented actions (a, k). Pick an unvisited pair
        # first; otherwise the pair maximizing UCB.
        unvisited = [
            (a, k) for a in range(n_actions) for k in range(K)
            if (a, k) not in node.action_visits
        ]
        if unvisited:
            ak = unvisited[int(self._rng.integers(0, len(unvisited)))]
        else:
            ak = max(
                ((a, k) for a in range(n_actions) for k in range(K)),
                key=lambda ak_: node.ucb_score(ak_, self.ucb_c),
            )
        action, k_idx = ak
        theta = thetas[k_idx]

        next_state, obs_idx, reward = BAMCPPlanner._step(
            state, theta, action, T, B_position, B_probe, B_outcome,
            probe_state_set, self._rng,
        )

        outcome = obs_idx % 3
        if outcome == 1 or outcome == 2:
            value = reward
        else:
            child_key = (action, obs_idx)
            if child_key not in node.children:
                node.children[child_key] = _Node()
                rollout_value = self._rollout(
                    next_state, depth - 1,
                    T, B_position, B_probe, B_outcome, probe_state_set,
                    n_actions, K, thetas,
                )
                value = reward + self.gamma * rollout_value
            else:
                child = node.children[child_key]
                future_value = self._simulate(
                    child, next_state, depth - 1,
                    T, B_position, B_probe, B_outcome, probe_state_set,
                    n_actions, K, thetas,
                )
                value = reward + self.gamma * future_value

        node.visit_count += 1
        node.total_value += value
        node.action_visits[ak] = node.action_visits.get(ak, 0) + 1
        node.action_values[ak] = node.action_values.get(ak, 0.0) + value
        return value

    def _rollout(
        self,
        state: int,
        depth: int,
        T: np.ndarray,
        B_position: np.ndarray,
        B_probe: np.ndarray,
        B_outcome: np.ndarray,
        probe_state_set: set[int],
        n_actions: int,
        K: int,
        thetas: list[int],
    ) -> float:
        """Random-action rollout over augmented (a, k) pairs."""
        total = 0.0
        gamma_t = 1.0
        for _ in range(depth):
            action = int(self._rng.integers(0, n_actions))
            k_idx = int(self._rng.integers(0, K))
            theta = thetas[k_idx]
            next_state, obs_idx, reward = BAMCPPlanner._step(
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

def _plan_numba_impl(
    self: BOSSPlanner,
    q_state, q_static, T, B_list, horizon,
    theta_spec, t_factor_spec, b_factor_specs,
    registry, reward_fn, terminal_fn,
    reward_table=None,
):
    """BOSS Numba dispatcher (kept module-level so it can call BAMCP-style
    preprocess helpers without recursion)."""
    from learningpomdp.planning._boss_numba import _run_boss_numba

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

    if t_is_theta_independent:
        T_eff = T_np / (T_np.sum(axis=0, keepdims=True) + 1e-30)
    else:
        q_static_flat = np.asarray(q_static, dtype=np.float64).reshape(-1)
        q_static_flat = q_static_flat / q_static_flat.sum()
        T_marg = np.einsum("ijka,k->ija", T_np, q_static_flat)
        T_eff = T_marg / (T_marg.sum(axis=0, keepdims=True) + 1e-30)

    B_pos = B_list_np[pos_idx]
    n_views = B_pos.shape[0]

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

    if reward_table is None:
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

    # BOSS thetas were sampled in prepare_episode and frozen for the
    # episode (deduplicated). Convert to int32 array for Numba.
    thetas = np.array(sorted(set(self._sampled_thetas)), dtype=np.int32)
    K_samples = thetas.shape[0]

    # Per-simulation root state sample (BOSS varies root_state per
    # simulation; thetas are episode-frozen).
    root_states = np.zeros(self.n_simulations, dtype=np.int32)
    for sim_idx in range(self.n_simulations):
        root_states[sim_idx] = int(self._rng.choice(n_states, p=q_state_np))

    obs_stride_total = n_views
    for ci in range(n_likelihood):
        obs_stride_total *= int(lik_obs_card[ci])

    seed = int(self._rng.integers(0, 2**31 - 1))

    visits = _run_boss_numba(
        self.n_simulations, max_depth, n_actions,
        thetas, K_samples, root_states,
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
    return jnp.asarray(dist, dtype=jnp.float32)


# Bind the module-level impl to the class.
BOSSPlanner._plan_numba = _plan_numba_impl  # type: ignore[attr-defined]
