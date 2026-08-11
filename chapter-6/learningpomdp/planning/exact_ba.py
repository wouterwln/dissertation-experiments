"""Exact finite-horizon Bayes-adaptive optimum by memoized expectimax.

Reference point for the BA-RL slate: the exact Bayes-adaptive value of the
*episode* problem on a reduced instance, computed without search
approximation, so the gap between AIF-MP (or any sampled-model baseline)
and the per-decision optimum is measurable rather than assumed. A solver
has no lookahead knob, which is why this is also the cleanest
lookahead-parity control available.

Why this terminates
-------------------
The belief simplex is continuous, but the belief ``(q_state, q_static)`` is
a *deterministic function of the action–observation history*. For a bounded
episode with finite actions and finitely many observations the reachable
history set is finite, so a recursion over histories is finite too. At each
node we enumerate actions, then the observations with nonzero
``P(o | b, a)``, update the belief, and back up
``E[r] + gamma * sum_o P(o | b, a) V(b_o)``.

Nodes are memoized on ``(remaining, belief)`` rather than on the raw
history, since ``V*`` depends only on those. That is what makes the
recursion cheap: histories differing only in the *order* of their
observations reach the same posterior, so on door_code at K=3 / L=2 a
depth-6 tree with a 12-wide branching factor collapses from ~3M histories
to ~800 expanded nodes.

Shared-filter protocol
----------------------
Belief updates go through the very same
``inference.state_inference.bayesian_state_update`` the runners call, on the
same ``T`` / ``log_B_list`` the runner hands to ``plan()``. The reference
therefore shares the experiment's one filter — a node's belief is
bit-for-bit the belief the runner would hold after the same history — rather
than measuring a differently-filtered problem.

"Exact" therefore means *exact given the runners' filter*, and that filter
runs at JAX's default float32 — this module does not enable
``jax_enable_x64``. The backup itself is float64, but its inputs are the
float32 beliefs the experiment actually holds, which is the point of sharing
the filter: switching the reference to float64 would make it exact for a
problem no runner solves. So values carry ~float32 relative accuracy
(~1e-7), and anything downstream that compares them must use a *relative*
tolerance scaled to the value magnitude — see ``optimal_action_set`` below,
where Factored Tiger's ±100 values put the float32 noise floor around 1e-5
in absolute terms.

Three approximations sit inside that "exact", all of them pruning or
rounding rather than search truncation:

1. **Observation pruning.** ``_node`` expands only the observation branches
   whose predictive probability exceeds ``_Model.support_tol``, and weights
   the survivors by ``p_i / sum(p over *all* candidates)`` — the dropped
   mass is simply not counted, not redistributed over the survivors. So a
   node's value is low by at most (pruned mass) x max|V|.
2. **Candidate-set pruning.** ``_reachable_observations`` applies the same
   tolerance to the per-channel likelihoods, so an (o, x, θ) cell below it
   never becomes a candidate at all.
3. **Belief-hash rounding.** ``_belief_key`` rounds beliefs to
   ``key_decimals`` places before memoizing, perturbing values by
   O(10**-key_decimals · max|reward|).

(1) and (2) are derived from the reward scale rather than fixed in
probability space, because a *fixed* probability tolerance is a 100x
tighter value bound on door_code (±1) than on Factored Tiger (±100); see
``_PRUNE_VALUE_BUDGET``. That holds the induced value error at or below the
float32 noise the values already carry — ~1e-7 on door_code, ~1e-5 on
Factored Tiger. (3) is deliberately *not* reward-scaled: its whole job is
to absorb that same float32 noise so order-permuted histories collapse onto
one node, and tightening it in step with the reward scale would break the
collapse it exists for. Its value cost on Factored Tiger is therefore
~1e-4, the largest of the three.

Note that this filter is *factored*: it re-forms ``q(x, θ) = q(x) q(θ)`` at every
predict step. On both reduced instances that costs nothing, because the
observation channels pin ``x`` exactly (door_code's ``position`` channel is
the identity; Factored Tiger's transitions are θ-independent and
deterministic, so the opened-door mask is a function of the action history),
which makes the joint exactly a product. On an environment where ``x`` stays
genuinely uncertain the factored filter — and hence this reference — is
mean-field, not exact.

What is optimized
-----------------
The **environment reward table**, not the AIF goal prior, so the value is
directly comparable to the sample-based baselines (which consume env reward)
and to the runners' reported returns. ``reward_table[s, a, θ]`` is the
expected immediate reward of taking ``a`` in ``s`` under ``θ``.

Scope — read this before comparing block returns
------------------------------------------------
The optimum is over the **episode** horizon (``remaining`` decisions from
the current posterior), not over the whole m x n block: a block-level
optimum would have to plan across episode resets and the q(θ) carried
between them. So this is the exact Bayes-adaptive optimum of the decision
problem each planner is actually handed at each step — which is precisely
what an action-agreement measurement needs — and an exact upper bound on
value *within* an episode from a given belief.

It is **not** a proven upper bound on block return, and on Factored Tiger
it demonstrably is not one late in a block. Opening a door reveals θ_k
perfectly but costs -100 against a +10 upside, so the information cannot
repay itself inside one episode and the episode-greedy optimum never buys
it; a planner that takes that hit in episode 0 enters episode 1 with a
sharper q(θ) than this reference ever reaches. The block optimum would
make that trade. Treat the returned value as the exact per-episode
optimum, not as the Bayes-adaptive lifetime optimum.

Ties — the tie-break is a block-level choice, not a formality
-------------------------------------------------------------
Optimal actions tie *structurally* on both instances, not by numerical
accident: on door_code every probe station has the same value, because the
posterior depends only on the multiset of probe results, and on Factored
Tiger LISTEN_0 and LISTEN_1 tie under a symmetric prior. Every member of
``optimal_action_set`` is exactly optimal **for the episode**, so
``action_values`` and the returned ``value`` are tie-break-invariant.

The *block* return is not. Which probe you visit decides which bit of θ you
learn, and that carries into the next episode's prior — so two arms that
are both exactly optimal at every decision can post different block
returns. Hence ``tie_break``: running the arm under both rules brackets the
tie-break spread instead of hiding it inside one arbitrary ``argmax``.
Neither bound is the block optimum, which is above both (see Scope).

Cost guard
----------
The tree grows as ``(|A| · |O|)^depth``. ``max_nodes`` caps the number of
expanded nodes and ``NodeCapExceeded`` reports the count reached, so an
intractable configuration surfaces in seconds instead of hanging a job.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from learningpomdp.factors import FactorSpec, ThetaSpec, align_to_full_components
from learningpomdp.inference.state_inference import bayesian_state_update

# The two probability-space tolerances are quoted as the *value* error they
# buy, then divided by the reward scale, because that is the quantity that
# has to be comparable across benchmarks whose rewards differ 100x. Both
# budgets are set so the induced error stays at or below the float32 noise
# the shared filter already puts on the values (~1e-7 relative).
#
# Below `_PRUNE_VALUE_BUDGET / max|reward|`, an observation / (x, θ) cell is
# treated as unreachable and its mass is dropped from the backup.
_PRUNE_VALUE_BUDGET = 1e-7
# A belief with more than 1 - `_TERMINAL_VALUE_BUDGET / max|reward|` mass on
# absorbing states is treated as absorbed (value 0). Mass strictly between
# the two bounds means the observation channels failed to resolve
# termination, which would make the backup silently wrong — so it raises.
_TERMINAL_VALUE_BUDGET = 1e-6
# Floor on both: the shared filter returns analytically-zero cells as ~1e-12
# round-off (its own `+ EPSILON` log floor), so a tolerance below this floor
# would expand branches the filter itself cannot tell from impossible,
# growing the tree without adding information.
_PROBABILITY_FLOOR = 1e-9
# Tie tolerance for `optimal_action_set`, relative to the node's largest
# |value|. Kept at 1e-6 on measured insensitivity, not on judgement — see the
# sensitivity paragraph in that function's docstring.
_TIE_REL_TOL = 1e-6
# Default node cap. Deliberately small enough that a mis-sized instance
# fails in seconds.
DEFAULT_MAX_NODES = 200_000


class NodeCapExceeded(RuntimeError):
    """The reachable-history tree exceeded the configured node cap."""


def _probability_tol(value_budget: float, reward_scale: float) -> float:
    """Probability tolerance buying ``value_budget`` in reward units."""
    return max(_PROBABILITY_FLOOR, value_budget / reward_scale)


def optimal_action_set(
    action_values: np.ndarray, rel_tol: float = _TIE_REL_TOL
) -> np.ndarray:
    """Indices of the exactly-optimal actions at one node, ascending.

    The single definition of "optimal action" shared by the planner's
    tie-break and by the agreement probe in
    ``scripts/run_exact_reference.py``, so the arm and the measurement
    cannot disagree about what ties.

    ``rel_tol`` is relative to the largest |value| at the node: the ties are
    analytically exact (see the module docstring) but computed through the
    runners' float32 filter, so an absolute threshold is meaningless across
    benchmarks whose values differ by two orders of magnitude (door_code ±1,
    Factored Tiger ±100) and can only under-count ties. The flip side is that
    an action *selected* from this set is optimal to within
    ``rel_tol · max|value|`` — 1e-6 on door_code, 1e-4 on Factored Tiger —
    rather than in exact arithmetic. That is the same order as the float32
    noise on the values themselves, and it is the price of a tie-break that
    does not depend on which of two equal values round-off happened to
    favour.

    Sensitivity — why 1e-6 and not tighter
    --------------------------------------
    ``optimal_set_rate`` is the headline number of
    ``scripts/run_exact_reference.py`` and this tolerance defines it, so the
    tolerance is measured rather than argued: every run of that stage emits
    the rate at 1e-6 / 1e-7 / 1e-8 as
    ``action_agreement.tie_tolerance_sweep``. At smoke scale (5 seeds x 5
    episodes) the rate is *unchanged* from 1e-6 to 1e-7 on both reduced
    instances — 0.893 over 84 door_code decisions, 0.940 over 150 Factored
    Tiger ones — and on Tiger the sets are member-for-member identical (438
    members either way). So the reported rate is not resting on tolerance
    headroom.

    It does fall at 1e-8 (to 0.869 / 0.887), but that fall is round-off
    rather than optimality: re-run both benchmarks under
    ``JAX_ENABLE_X64=1`` and all three tolerances agree exactly (0.899 /
    0.940, mean set sizes equal to three places). 1e-8 relative is below the
    shared filter's float32 noise floor, so it under-counts ties the values
    carry analytically. 1e-6 is therefore the loose end of a plateau, not a
    generous cutoff — and the residual generosity is bounded by the flip
    side above: an action admitted by the tolerance alone forgoes at most
    1e-6 / 1e-4 of exact action value, against the ~2e-2 / ~3e-1 mean regret
    that stage already reports.
    """
    values = np.asarray(action_values, dtype=np.float64)
    tol = rel_tol * max(1.0, float(np.abs(values).max()))
    return np.nonzero(values >= values.max() - tol)[0]


@dataclass(frozen=True)
class ExactSolution:
    """Root value, per-action values, and the expanded-node count."""

    value: float
    action_values: np.ndarray | None
    n_nodes: int


def _belief_key(q: jnp.ndarray, decimals: int) -> bytes:
    """Hashable key for a belief vector.

    Beliefs agreeing to ``decimals`` places share a node. The value function
    is Lipschitz in the belief, so this perturbs the returned value by
    O(10**-decimals); what it actually collapses on these instances is the
    set of order-permuted observation histories, whose posteriors are
    analytically identical and differ only in float32 round-off.
    """
    return np.round(
        np.asarray(q, dtype=np.float64).reshape(-1), decimals
    ).tobytes()


class _Model:
    """Belief-independent precompute for one (T, B_list) pair.

    The expensive part of a node is enumerating observations, so the
    per-channel likelihoods, the reachable observation tuples, and their
    products are all built once here and reduced to a single matrix-vector
    product per (node, action).
    """

    def __init__(
        self,
        T: jnp.ndarray,
        B_list: list[jnp.ndarray],
        theta_spec: ThetaSpec,
        t_factor_spec: FactorSpec,
        b_factor_specs: tuple[FactorSpec, ...],
        reward_table: np.ndarray,
        terminal_states: tuple[int, ...],
    ) -> None:
        self.T = T
        # Rebuilt exactly as the runners do, so the filter call is identical.
        self.log_B_list = tuple(jnp.log(B_c + 1e-12) for B_c in B_list)
        self.theta_spec = theta_spec
        self.t_factor_spec = t_factor_spec
        self.b_factor_specs = b_factor_specs

        self.n_states = int(T.shape[0])
        self.n_static = theta_spec.n_static
        self.n_actions = int(T.shape[-1])
        self.T_np = np.asarray(T, dtype=np.float64)
        self.action_onehots = jnp.eye(self.n_actions, dtype=jnp.float32)

        expected = (self.n_states, self.n_actions, self.n_static)
        reward_table = np.asarray(reward_table, dtype=np.float64)
        if reward_table.shape != expected:
            raise ValueError(
                f"reward_table shape {reward_table.shape} != expected {expected}"
            )
        self.reward_table = reward_table

        # Reward scale of *this* instance, so the two pruning tolerances are
        # a value budget rather than a bare probability — see the module
        # docstring. Floored at 1.0 so a near-zero reward table cannot blow
        # the tolerances up.
        self.reward_scale = max(1.0, float(np.abs(reward_table).max()))
        self.support_tol = _probability_tol(_PRUNE_VALUE_BUDGET, self.reward_scale)
        self.terminal_tol = _probability_tol(
            _TERMINAL_VALUE_BUDGET, self.reward_scale
        )

        self.terminal_mask = np.zeros(self.n_states, dtype=bool)
        for s in terminal_states:
            self.terminal_mask[int(s)] = True
        self.has_terminal = bool(self.terminal_mask.any())

        lik = self._channel_likelihoods(B_list)
        self.candidates = self._reachable_observations(lik)
        self.lik_prod = np.ones(
            (len(self.candidates), self.n_states * self.n_static), dtype=np.float64
        )
        for i, tup in enumerate(self.candidates):
            for c, o in enumerate(tup):
                self.lik_prod[i] *= lik[c][o]

        dims = [int(m.shape[0]) for m in lik]
        offsets = np.cumsum([0] + dims)
        obs_vectors = np.zeros((len(self.candidates), int(offsets[-1])), dtype=np.float32)
        for i, tup in enumerate(self.candidates):
            for c, o in enumerate(tup):
                obs_vectors[i, int(offsets[c]) + o] = 1.0
        self.obs_vectors = jnp.asarray(obs_vectors)

    def _channel_likelihoods(self, B_list: list[jnp.ndarray]) -> list[np.ndarray]:
        """Per-channel ``(n_obs_c, n_states * n_static)`` likelihoods."""
        mats: list[np.ndarray] = []
        for B_c, spec in zip(B_list, self.b_factor_specs):
            B_np = np.asarray(B_c, dtype=np.float64)
            n_obs = int(B_np.shape[0])
            aligned = align_to_full_components(
                B_np, spec, self.theta_spec, leading_axes=2
            )
            full = np.broadcast_to(
                aligned, (n_obs, self.n_states) + self.theta_spec.component_sizes
            )
            mats.append(np.ascontiguousarray(full).reshape(n_obs, -1))
        return mats

    def _reachable_observations(
        self, lik: list[np.ndarray]
    ) -> list[tuple[int, ...]]:
        """Observation tuples with nonzero likelihood at some (x, θ) cell."""
        supports = [m > self.support_tol for m in lik]
        found: set[tuple[int, ...]] = set()
        for cell in range(self.n_states * self.n_static):
            per_channel = [np.nonzero(s[:, cell])[0].tolist() for s in supports]
            if any(not p for p in per_channel):
                continue
            found.update(itertools.product(*per_channel))
        return sorted(found)

    def predict(
        self, q_state: np.ndarray, q_static: np.ndarray, action: int
    ) -> np.ndarray:
        """Normalized q(x', θ) after ``action``, flattened over (x', θ)."""
        T_a = self.T_np[..., action]
        pred = np.tensordot(T_a, q_state, axes=([1], [0]))
        aligned = align_to_full_components(
            pred, self.t_factor_spec, self.theta_spec, leading_axes=1
        )
        joint = aligned * q_static.reshape((1,) + self.theta_spec.component_sizes)
        flat = joint.reshape(-1)
        total = flat.sum()
        if total <= 0:
            raise ValueError(f"action {action} has zero predictive mass")
        return flat / total


class _Solver:
    """Memoized expectimax over reachable action-observation histories."""

    def __init__(
        self,
        model: _Model,
        gamma: float,
        max_nodes: int,
        key_decimals: int,
        memoize: bool = True,
    ) -> None:
        self._model = model
        self._gamma = float(gamma)
        self._max_nodes = int(max_nodes)
        self._key_decimals = int(key_decimals)
        self._memoize = memoize
        self._memo: dict[tuple, tuple[float, np.ndarray | None]] = {}
        self.n_nodes = 0

    def solve(
        self, q_state: jnp.ndarray, q_static: jnp.ndarray, remaining: int
    ) -> ExactSolution:
        value, action_values = self._node(q_state, q_static, int(remaining))
        return ExactSolution(
            value=value, action_values=action_values, n_nodes=self.n_nodes
        )

    def _node(
        self, q_state: jnp.ndarray, q_static: jnp.ndarray, remaining: int
    ) -> tuple[float, np.ndarray | None]:
        model = self._model
        q_state_np = np.asarray(q_state, dtype=np.float64).reshape(-1)

        if model.has_terminal:
            terminal_mass = float(q_state_np[model.terminal_mask].sum())
            if terminal_mass > 1.0 - model.terminal_tol:
                return 0.0, None
            if terminal_mass > model.terminal_tol:
                raise ValueError(
                    f"belief carries {terminal_mass:.6g} mass on absorbing states "
                    "without resolving termination; the observation channels must "
                    "separate absorbed from non-absorbed states for the expectimax "
                    "backup to be exact"
                )
        if remaining <= 0:
            return 0.0, None

        key = (
            remaining,
            _belief_key(q_state, self._key_decimals),
            _belief_key(q_static, self._key_decimals),
        )
        if self._memoize:
            hit = self._memo.get(key)
            if hit is not None:
                return hit

        self.n_nodes += 1
        if self.n_nodes > self._max_nodes:
            raise NodeCapExceeded(
                f"exact expectimax expanded {self.n_nodes} nodes, exceeding "
                f"max_nodes={self._max_nodes} (hit at remaining={remaining}). "
                f"Shrink the instance (K / max_steps) or raise the cap."
            )

        q_static_np = np.asarray(q_static, dtype=np.float64).reshape(-1)
        joint_prior = np.outer(q_state_np, q_static_np)
        action_values = np.empty(model.n_actions, dtype=np.float64)

        for action in range(model.n_actions):
            reward = float(np.sum(joint_prior * model.reward_table[:, action, :]))
            continuation = 0.0
            if remaining > 1:
                predicted = model.predict(q_state_np, q_static_np, action)
                probs = model.lik_prod @ predicted
                # Normalising by the full sum, not by the surviving branches:
                # pruned mass is dropped rather than redistributed. See the
                # module docstring's approximation (1).
                total = probs.sum()
                for i in np.nonzero(probs > model.support_tol)[0]:
                    q_state_child, q_static_child = bayesian_state_update(
                        q_state,
                        q_static,
                        model.T,
                        model.log_B_list,
                        model.obs_vectors[i],
                        model.action_onehots[action],
                        theta_spec=model.theta_spec,
                        t_factor_spec=model.t_factor_spec,
                        b_factor_specs=model.b_factor_specs,
                    )
                    child_value, _ = self._node(
                        q_state_child, q_static_child, remaining - 1
                    )
                    continuation += (float(probs[i]) / total) * child_value
            action_values[action] = reward + self._gamma * continuation

        out = (float(action_values.max()), action_values)
        if self._memoize:
            self._memo[key] = out
        return out


def solve_exact(
    q_state: jnp.ndarray,
    q_static: jnp.ndarray,
    T: jnp.ndarray,
    B_list: list[jnp.ndarray],
    remaining: int,
    *,
    theta_spec: ThetaSpec,
    t_factor_spec: FactorSpec,
    b_factor_specs: tuple[FactorSpec, ...],
    reward_table: np.ndarray,
    terminal_states: tuple[int, ...] = (),
    gamma: float = 1.0,
    max_nodes: int = DEFAULT_MAX_NODES,
    key_decimals: int = 6,
    memoize: bool = True,
) -> ExactSolution:
    """One-shot exact expectimax from ``(q_state, q_static)``.

    ``remaining`` is the number of decisions left in the episode. ``gamma``
    defaults to 1.0 so the solver optimizes exactly the undiscounted return
    the runners report.
    """
    model = _Model(
        T, B_list, theta_spec, t_factor_spec, b_factor_specs,
        reward_table, terminal_states,
    )
    solver = _Solver(model, gamma, max_nodes, key_decimals, memoize=memoize)
    return solver.solve(q_state, q_static, remaining)


class ExactBAPlanner:
    """Planner returning the exact Bayes-adaptive optimal action.

    Drop-in for the runners' ``plan()`` protocol. The memo persists across
    the decisions of one episode: the first call expands the whole reachable
    tree, and every later decision of that episode is a memo hit, so the
    per-episode cost is a single tree. ``prepare_episode`` drops the memo,
    since a new episode starts from a different q(θ) and so has a disjoint
    reachable set.

    ``horizon`` is taken as the number of decisions *remaining*, which is a
    contract the caller can get wrong silently: driven with a canonical
    ``horizon`` shorter than ``max_steps``, the runner's
    ``min(horizon, max_steps - step)`` hands over a constant lookahead and
    this becomes a receding-horizon arm still labelled "exact". So
    ``max_steps`` is required at construction and every ``plan()`` call
    asserts ``horizon == max_steps - decisions taken this episode``, which
    holds exactly under ``clip_horizon=True`` with ``horizon >= max_steps``.
    A caller who genuinely wants a shorter exact lookahead says so by
    constructing with the smaller ``max_steps``.

    Args:
        max_steps: the runner's per-episode step budget. Enforced, not
            assumed — see above.
        reward_table: ``(n_states, n_actions, n_static)`` expected immediate
            env reward. Optional — the RockSample / Factored Tiger runners
            already pass one into ``plan()``; the door_code runner does not,
            so inject it here for that env.
        terminal_states: absorbing states after which no reward accrues
            (door_code's SUCCESS / FAILURE). Empty for bounded-horizon envs.
        gamma: return discount. 1.0 matches the runners' undiscounted
            ``cumulative_reward`` metric. It is deliberately *not* the 0.99
            the BAMCP baseline discounts its rollouts at, so a value
            comparison against that arm carries a small objective mismatch on
            top of the approximation gap — flagged, not silently equalized.
        max_nodes: cap on expanded nodes per episode.
        key_decimals: belief-hash precision — see ``_belief_key``.
        tie_break: which member of ``optimal_action_set`` to act on,
            ``"lowest"`` or ``"highest"`` index. Every member is equally
            optimal for the episode, so this cannot change ``action_values``
            — but it can change the *block* return, because it decides which
            part of θ gets learned first. See the module docstring's Ties
            section; run both to bracket that spread.
    """

    def __init__(
        self,
        max_steps: int,
        reward_table: np.ndarray | None = None,
        terminal_states: tuple[int, ...] = (),
        gamma: float = 1.0,
        max_nodes: int = DEFAULT_MAX_NODES,
        key_decimals: int = 6,
        tie_break: str = "lowest",
    ) -> None:
        if tie_break not in ("lowest", "highest"):
            raise ValueError(
                f"tie_break must be 'lowest' or 'highest', got {tie_break!r}"
            )
        self._max_steps = int(max_steps)
        self._reward_table = reward_table
        self._terminal_states = tuple(int(s) for s in terminal_states)
        self._gamma = float(gamma)
        self._max_nodes = int(max_nodes)
        self._key_decimals = int(key_decimals)
        self._tie_break = tie_break

        self._solver: _Solver | None = None
        self._decisions_this_episode = 0

        # Diagnostics consumed by scripts/run_exact_reference.py.
        self.last_action_values: np.ndarray | None = None
        self.max_nodes_expanded = 0
        self.n_plan_calls = 0

    def prepare_episode(self, q_static: jnp.ndarray, rng) -> None:
        self._solver = None
        self._decisions_this_episode = 0

    def plan(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list[jnp.ndarray],
        goal,
        horizon: int,
        theta_mask=None,
        theta_spec: ThetaSpec | None = None,
        t_factor_spec: FactorSpec | None = None,
        b_factor_specs: tuple[FactorSpec, ...] | None = None,
        reward_table: np.ndarray | None = None,
    ) -> jnp.ndarray:
        """One-hot at an exact optimal action (``tie_break`` picks which).
        ``goal`` is ignored: the reference optimizes env reward, not the AIF
        goal prior."""
        expected_horizon = self._max_steps - self._decisions_this_episode
        if int(horizon) != expected_horizon:
            raise ValueError(
                f"ExactBAPlanner got horizon={int(horizon)} on decision "
                f"{self._decisions_this_episode} of the episode, but the exact "
                f"backup needs the remaining budget {expected_horizon} = "
                f"max_steps({self._max_steps}) - decisions taken. Drive it with "
                "clip_horizon=True and horizon >= max_steps, or construct it "
                "with max_steps equal to the lookahead you want; otherwise "
                "this is a receding-horizon arm, not an exact one."
            )
        self._decisions_this_episode += 1

        table = self._reward_table if self._reward_table is not None else reward_table
        if table is None:
            raise ValueError(
                "ExactBAPlanner needs an env reward table — pass reward_table "
                "to __init__ (door_code / minigrid) or let the runner supply "
                "it via plan() (rock_sample / factored_tiger)."
            )
        if self._solver is None:
            # Built once per episode (prepare_episode drops the solver), from
            # the tensors of that episode's *first* plan() call — not on every
            # call. Sound because T / B_list are fixed within an episode for
            # every runner here; a runner that re-estimated them mid-episode
            # would have to drop the solver whenever they changed.
            model = _Model(
                T, B_list, theta_spec, t_factor_spec, b_factor_specs,
                table, self._terminal_states,
            )
            self._solver = _Solver(
                model, self._gamma, self._max_nodes, self._key_decimals,
            )

        solution = self._solver.solve(q_state, q_static, horizon)
        self.last_action_values = solution.action_values
        self.max_nodes_expanded = max(self.max_nodes_expanded, solution.n_nodes)
        self.n_plan_calls += 1

        n_actions = int(T.shape[-1])
        if solution.action_values is None:
            return jnp.ones(n_actions, dtype=jnp.float32) / n_actions
        optimal = optimal_action_set(solution.action_values)
        best = int(optimal[0] if self._tie_break == "lowest" else optimal[-1])
        return jnp.zeros(n_actions, dtype=jnp.float32).at[best].set(1.0)
