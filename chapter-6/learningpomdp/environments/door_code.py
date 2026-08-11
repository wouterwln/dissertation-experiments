"""Door-code POMDP for system identification experiments.

Discrete environment with a persistent permutation parameter theta. The
agent must enter the correct door, but which door is correct depends on
the hidden code. Probe stations reveal partial, optionally noisy
information about the permutation; many probes across many episodes are
typically needed to identify theta.

Probes are **binary partitions** of the permutation space S_K. Each
probe station PROBE_i implements a fixed predicate p_i: S_K -> {0, 1}
and emits the bit p_i(theta), optionally flipped with probability
probe_noise. M partition probes are designed so the M-bit codeword
uniquely identifies every theta in S_K (so full identification is
achievable with enough visits), but each individual visit only carries
~1 - H(probe_noise) bits.

State space (n_states = M + 3):
    0          : START
    1..M       : PROBE_0..PROBE_{M-1}
    M+1        : SUCCESS  (absorbing)
    M+2        : FAILURE  (absorbing)

Action space (n_actions = M + K):
    0..M-1     : VISIT_PROBE_i  (move to PROBE_i)
    M..M+K-1   : OPEN_DOOR_i    (terminate; SUCCESS iff i = pi_theta(0))

Theta: theta_idx in [0, K!) indexes a permutation pi in S_K
       (lexicographic order). The correct door for theta is pi(0).

Modalities:
    position : theta-INDEPENDENT, identity emission, cardinality M+3
    probe    : theta-CONDITIONAL at PROBE_i states; cardinality 3
               (bit_0, bit_1, neutral)
    outcome  : theta-INDEPENDENT given state; cardinality 3

Tensor conventions (4D, matching MinigridMP-AIF-JAX):
    T          : (M+3, M+3, K!, M+K)         theta-dependent (door slice)
    B_position : (M+3, M+3)                  theta-independent
    B_probe    : (3, M+3, K!)                theta-dependent at probe states
    B_outcome  : (3, M+3)                    theta-independent
"""

from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np

from learningpomdp.environments.base import SAFE_LOG_FLOOR, StepResult
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.observations import (
    ModalityRegistry,
    ObservationModality,
    ThetaDependence,
)


# Probe observation indices
PROBE_BIT_0 = 0
PROBE_BIT_1 = 1
PROBE_NEUTRAL = 2
N_PROBE_OBS = 3

# Outcome observation indices
OUTCOME_NEUTRAL = 0
OUTCOME_SUCCESS = 1
OUTCOME_FAILURE = 2
N_OUTCOME_OBS = 3


# --- Permutations and parity ------------------------------------------------

def enumerate_permutations(K: int) -> list[tuple[int, ...]]:
    """Lexicographic enumeration of permutations of range(K)."""
    return [tuple(p) for p in itertools.permutations(range(K))]


def permutation_parity(perm: tuple[int, ...]) -> int:
    """Return 0 for even permutations, 1 for odd. Uses cycle decomposition."""
    n = len(perm)
    visited = [False] * n
    transposition_count = 0
    for i in range(n):
        if visited[i]:
            continue
        j = i
        cycle_len = 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        transposition_count += cycle_len - 1
    return transposition_count % 2


# --- Partition predicates ---------------------------------------------------

def default_partitions(K: int) -> list[Callable[[tuple[int, ...]], int]]:
    """Default binary partition predicates that uniquely identify S_K.

    For any K >= 2 we use bits of the lexicographic rank: with
    M = ceil(log2(K!)) predicates p_b(perm) = bit_b(rank(perm)), the
    M-bit codeword is rank-by-construction and therefore unique across
    all K! permutations. Bit 0 (LSB) is always balanced; higher bits
    become progressively more imbalanced and so leak less information
    per probe visit. This is the canonical reference scheme; pass
    `partitions=...` to MiniGridDoorCodeEnv / DoorCodeEnv to inject a
    hand-tuned alternative when better per-visit information matters.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2, got {K}")
    n_perms = math.factorial(K)
    M = math.ceil(math.log2(n_perms))
    rank_lookup = {p: i for i, p in enumerate(enumerate_permutations(K))}
    return [
        (lambda p, b=b: (rank_lookup[p] >> b) & 1) for b in range(M)
    ]


def build_partition_table(
    K: int,
    partitions: list[Callable[[tuple[int, ...]], int]],
) -> np.ndarray:
    """Return a (M, K!) int array where row i is the bit for partition i."""
    perms = enumerate_permutations(K)
    M = len(partitions)
    table = np.zeros((M, len(perms)), dtype=np.int32)
    for i, predicate in enumerate(partitions):
        for j, perm in enumerate(perms):
            bit = int(predicate(perm))
            if bit not in (0, 1):
                raise ValueError(
                    f"Partition {i} returned {bit} for perm {perm}; must be 0 or 1."
                )
            table[i, j] = bit
    return table


def partitions_uniquely_identify(table: np.ndarray) -> bool:
    """Check that every column of the partition table is distinct."""
    M, n_perms = table.shape
    seen = set()
    for j in range(n_perms):
        codeword = tuple(table[:, j].tolist())
        if codeword in seen:
            return False
        seen.add(codeword)
    return True


# --- Layout helpers ---------------------------------------------------------

def n_states_for(M: int, L: int = 1) -> int:
    """Total state count: START + (L-1) STAGE_k + M PROBE + SUCCESS + FAILURE."""
    return L + M + 2


def n_actions_for(M: int, K: int) -> int:
    return M + K


def n_static_for_K(K: int) -> int:
    return math.factorial(K)


def state_indices(M: int, L: int = 1) -> dict[str, int]:
    """State layout for an L-step lock with M probe stations.

    For L=1 this matches the single-step layout exactly:
    START=0, PROBE_i=1+i, SUCCESS=M+1, FAILURE=M+2.

    For L>1, intermediate STAGE_k states sit between START and PROBE_i:
    START=0 (= STAGE_0), STAGE_1=1, ..., STAGE_{L-1}=L-1,
    PROBE_i=L+i, SUCCESS=L+M, FAILURE=L+M+1.
    """
    layout = {"START": 0}
    for k in range(1, L):
        layout[f"STAGE_{k}"] = k
    for i in range(M):
        layout[f"PROBE_{i}"] = L + i
    layout["SUCCESS"] = L + M
    layout["FAILURE"] = L + M + 1
    return layout


def action_indices(M: int, K: int) -> dict[str, int]:
    return {
        **{f"VISIT_PROBE_{i}": i for i in range(M)},
        **{f"OPEN_DOOR_{i}": M + i for i in range(K)},
    }


# --- Tensor builders --------------------------------------------------------

def default_lock_doors(perm: tuple[int, ...], L: int) -> tuple[int, ...]:
    """Default lock-door rule: open the first L cells of pi in order.

    For L=1 this is just (pi(0),) — the single-cell case from earlier
    versions of the env. For L>1, the agent must open pi(0), pi(1), ...,
    pi(L-1) in that order without error.

    Open knob (see DoorCodeEnv docstring): swap this for any function
    that returns a length-L tuple of door indices.
    """
    if not 1 <= L <= len(perm):
        raise ValueError(f"L must be in [1, {len(perm)}], got {L}")
    return tuple(perm[:L])


def create_transition_tensor(
    K: int,
    M: int,
    L: int = 1,
    lock_doors_fn=None,
) -> np.ndarray:
    """T with shape (n_states, n_states, K!, M+K).

    Probe phase (START or any PROBE_i):
      VISIT_PROBE_j -> PROBE_j
      OPEN_DOOR_X   -> STAGE_1 if X = lock[0] and L>1
                       SUCCESS if X = lock[0] and L=1
                       FAILURE otherwise

    Commit phase (STAGE_k for k>=1):
      VISIT_PROBE_j -> FAILURE   (probing during commit phase = death)
      OPEN_DOOR_X   -> STAGE_{k+1} if X = lock[k] and k+1 < L
                       SUCCESS    if X = lock[k] and k+1 == L
                       FAILURE    otherwise
    """
    if lock_doors_fn is None:
        lock_doors_fn = lambda perm: default_lock_doors(perm, L)

    n_states = n_states_for(M, L)
    n_actions = n_actions_for(M, K)
    n_static = n_static_for_K(K)
    perms = enumerate_permutations(K)
    layout = state_indices(M, L)

    START = layout["START"]
    SUCCESS = layout["SUCCESS"]
    FAILURE = layout["FAILURE"]
    PROBE = [layout[f"PROBE_{i}"] for i in range(M)]
    STAGE = [START] + [layout[f"STAGE_{k}"] for k in range(1, L)]  # STAGE[k] for k=0..L-1
    probe_phase = [START] + PROBE  # states from which probing is allowed
    commit_phase = STAGE[1:]       # STAGE_1..STAGE_{L-1} only

    T = np.zeros((n_states, n_states, n_static, n_actions))

    for theta_idx, perm in enumerate(perms):
        lock = lock_doors_fn(perm)
        if len(lock) != L:
            raise ValueError(
                f"lock_doors_fn({perm}) returned length {len(lock)}; "
                f"must equal lock_length L={L}."
            )
        for d in lock:
            if not 0 <= d < K:
                raise ValueError(
                    f"lock_doors_fn({perm}) returned door {d}; "
                    f"must be in [0, {K})."
                )

        # Probe phase: probing always allowed; first door commit
        # transitions to STAGE_1 (or SUCCESS for L=1).
        for s_old in probe_phase:
            for j in range(M):
                T[PROBE[j], s_old, theta_idx, j] = 1.0
            for i in range(K):
                door_action = M + i
                if i == lock[0]:
                    target = STAGE[1] if L > 1 else SUCCESS
                else:
                    target = FAILURE
                T[target, s_old, theta_idx, door_action] = 1.0

        # Commit phase: probing leads to FAILURE; door commits advance
        # if correct, else FAILURE.
        for k in range(1, L):
            s_old = STAGE[k]
            for j in range(M):
                T[FAILURE, s_old, theta_idx, j] = 1.0
            for i in range(K):
                door_action = M + i
                if i == lock[k]:
                    target = STAGE[k + 1] if k + 1 < L else SUCCESS
                else:
                    target = FAILURE
                T[target, s_old, theta_idx, door_action] = 1.0

        # Terminal states absorb.
        for a in range(n_actions):
            T[SUCCESS, SUCCESS, theta_idx, a] = 1.0
            T[FAILURE, FAILURE, theta_idx, a] = 1.0

    return T


def create_observation_tensors(
    K: int,
    partition_table: np.ndarray,
    probe_noise: float = 0.0,
    L: int = 1,
) -> list[np.ndarray]:
    """Build [B_position, B_probe, B_outcome] for a multi-step lock of
    length L (L=1 reduces to the single-step env).

    Shapes:
      B_position : (n_states, n_states)        theta-independent
      B_probe    : (3, n_states, K!)           theta-dependent at PROBE_i states
      B_outcome  : (3, n_states)               theta-independent

    where n_states = L + M + 2.
    """
    if not 0.0 <= probe_noise <= 0.5:
        raise ValueError(
            f"probe_noise must be in [0, 0.5]; got {probe_noise}. "
            "Values > 0.5 are equivalent to 1 - probe_noise with a "
            "flipped predicate (and worse than 0.5)."
        )
    M, n_static = partition_table.shape
    n_states = n_states_for(M, L)
    layout = state_indices(M, L)

    PROBE = [layout[f"PROBE_{i}"] for i in range(M)]
    PROBE_SET = set(PROBE)
    SUCCESS = layout["SUCCESS"]
    FAILURE = layout["FAILURE"]

    B_position = np.eye(n_states)

    B_probe = np.zeros((N_PROBE_OBS, n_states, n_static))
    p_correct = 1.0 - probe_noise
    p_flipped = probe_noise
    for theta_idx in range(n_static):
        for i in range(M):
            true_bit = int(partition_table[i, theta_idx])
            B_probe[true_bit, PROBE[i], theta_idx] = p_correct
            B_probe[1 - true_bit, PROBE[i], theta_idx] = p_flipped
            # Neutral never emitted at probe stations.
        for s in range(n_states):
            if s not in PROBE_SET:
                B_probe[PROBE_NEUTRAL, s, theta_idx] = 1.0

    B_outcome = np.zeros((N_OUTCOME_OBS, n_states))
    for s in range(n_states):
        if s == SUCCESS:
            B_outcome[OUTCOME_SUCCESS, s] = 1.0
        elif s == FAILURE:
            B_outcome[OUTCOME_FAILURE, s] = 1.0
        else:
            B_outcome[OUTCOME_NEUTRAL, s] = 1.0

    return [B_position, B_probe, B_outcome]


def create_goal(
    K: int,
    M: int,
    success_logit: float = 4.0,
    failure_logit: float = -4.0,
    L: int = 1,
) -> np.ndarray:
    """(n_states, n_static) softmax preference: SUCCESS preferred, FAILURE not.

    Unlike the factored envs, the logit span here is
    ``success_logit - failure_logit`` and does not grow with K, so the
    swept ``door_code_preference`` grid clears the ``safe_log`` floor by
    ~20 decades. The guard is still applied so the three preference sweeps
    fail the same way if a future grid pushes the span far enough.
    """
    n_states = n_states_for(M, L)
    n_static = n_static_for_K(K)
    layout = state_indices(M, L)

    logits = np.zeros((n_states, n_static))
    logits[layout["SUCCESS"], :] = success_logit
    logits[layout["FAILURE"], :] = failure_logit

    exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
    goal = exp_logits / exp_logits.sum(axis=0, keepdims=True)
    # The planners consume float32 log C via messages.safe_log, which
    # clamps at log(max(C, 1e-30)); below that the goal is silently
    # truncated onto one floor value.
    if goal.min() < SAFE_LOG_FLOOR:
        raise ValueError(
            f"goal prior underflows the safe_log floor at "
            f"(success_logit, failure_logit)=({success_logit}, "
            f"{failure_logit}): min entry {goal.min():.3e} < "
            f"{SAFE_LOG_FLOOR:.0e}. The logits span "
            f"success_logit - failure_logit nats; narrow the gap."
        )
    return goal


def create_modality_registry(K: int, M: int, L: int = 1) -> ModalityRegistry:
    """Three-channel registry: position (INDEP), probe (CONDITIONAL),
    outcome (INDEP)."""
    layout = state_indices(M, L)
    probe_states = frozenset(layout[f"PROBE_{i}"] for i in range(M))
    return ModalityRegistry(modalities=(
        ObservationModality(
            name="position",
            cardinality=n_states_for(M, L),
            theta_dependence=ThetaDependence.INDEPENDENT,
        ),
        ObservationModality(
            name="probe",
            cardinality=N_PROBE_OBS,
            theta_dependence=ThetaDependence.CONDITIONAL,
            informative_states=probe_states,
        ),
        ObservationModality(
            name="outcome",
            cardinality=N_OUTCOME_OBS,
            theta_dependence=ThetaDependence.INDEPENDENT,
        ),
    ))


# --- Environment class ------------------------------------------------------

class DoorCodeEnv:
    """Discrete door-code POMDP with a persistent permutation theta,
    binary-partition probes (optionally noisy), and an optional
    multi-step lock.

    Open knobs:
    - `partitions`: which predicates the probe stations report. Default
      uniquely identifies S_K for K up to 5 (cell-based) and any K via
      lex-rank bits for K>=5.
    - `lock_length` and `lock_doors_fn`: how many doors must be opened
      (in order) per episode, and which doors. Default lock_length=1
      with lock_doors_fn(perm) = (perm[0],) — single-step task,
      backward-compatible with earlier versions of the env. Increase
      lock_length to break baseline planners that rely on "one lucky
      success identifies pi(0)".
    - `goal`: the (n_states, n_static) preference. Default prefers
      SUCCESS uniformly across theta.

    Args:
        K: number of doors / symbols (>= 2).
        M: number of probe stations. Defaults to len(default_partitions(K)).
        probe_noise: per-probe bit-flip probability in [0, 0.5].
        partitions: optional list of M predicates, one per probe.
            If None, uses default_partitions(K).
        lock_length: number of doors that must be opened correctly in
            sequence per episode (>= 1).
        lock_doors_fn: optional callable (perm) -> tuple[int, ...] of
            length lock_length, giving the door sequence the agent
            must open under each permutation. Default opens the first
            lock_length cells of pi in order.
        goal: optional override for the goal preference tensor with
            shape (n_states, n_static). If None, uses create_goal.
        success_logit: goal-prior logit on the SUCCESS state (preference
            strength). Ignored when `goal` is given.
        failure_logit: goal-prior logit on the FAILURE state (cost of
            failure). Ignored when `goal` is given.
        obs_seed: seed for the env's stochastic-emission RNG.
    """

    def __init__(
        self,
        K: int = 4,
        M: int | None = None,
        probe_noise: float = 0.0,
        partitions: list[Callable[[tuple[int, ...]], int]] | None = None,
        lock_length: int = 1,
        lock_doors_fn: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
        goal: np.ndarray | None = None,
        success_logit: float = 4.0,
        failure_logit: float = -4.0,
        obs_seed: int | None = 0,
    ) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        if not 1 <= lock_length <= K:
            raise ValueError(f"lock_length must be in [1, {K}], got {lock_length}")

        if partitions is None:
            partitions = default_partitions(K)
        if M is None:
            M = len(partitions)
        if len(partitions) != M:
            raise ValueError(
                f"partitions has {len(partitions)} entries but M={M}."
            )
        if lock_doors_fn is None:
            lock_doors_fn = lambda perm: default_lock_doors(perm, lock_length)

        self.K = K
        self.M = M
        self.L = lock_length
        self.lock_length = lock_length
        self.probe_noise = probe_noise
        self.n_states = n_states_for(M, lock_length)
        self.n_actions = n_actions_for(M, K)
        self.n_static = n_static_for_K(K)
        # M1 degenerate single-component factor structure: theta = S_K is
        # one component of size K!; T and the probe channel see it,
        # position and outcome are theta-independent.
        self.theta_spec = ThetaSpec(component_sizes=(self.n_static,))
        self.t_factor_spec = FactorSpec(theta_components=(0,))
        self.b_factor_specs: tuple[FactorSpec, ...] = (
            FactorSpec(theta_components=()),       # B_position
            FactorSpec(theta_components=(0,)),     # B_probe
            FactorSpec(theta_components=()),       # B_outcome
        )
        self._lock_doors_fn = lock_doors_fn

        self._layout = state_indices(M, lock_length)
        self._START = self._layout["START"]
        self._SUCCESS = self._layout["SUCCESS"]
        self._FAILURE = self._layout["FAILURE"]
        self._PROBE = [self._layout[f"PROBE_{i}"] for i in range(M)]
        self._STAGE = [self._START] + [
            self._layout[f"STAGE_{k}"] for k in range(1, lock_length)
        ]
        self._terminals = {self._SUCCESS, self._FAILURE}

        self._perms = enumerate_permutations(K)
        self._partition_table = build_partition_table(K, partitions)
        self._unique_codewords = partitions_uniquely_identify(self._partition_table)

        self._T = create_transition_tensor(
            K, M, L=lock_length, lock_doors_fn=lock_doors_fn,
        )
        self._B_list = create_observation_tensors(
            K, self._partition_table, probe_noise=probe_noise, L=lock_length,
        )
        if goal is None:
            self._goal = create_goal(
                K, M, success_logit=success_logit,
                failure_logit=failure_logit, L=lock_length,
            )
        else:
            expected_shape = (self.n_states, self.n_static)
            if goal.shape != expected_shape:
                raise ValueError(
                    f"goal shape {goal.shape} != expected {expected_shape}"
                )
            self._goal = np.asarray(goal)
        self._registry = create_modality_registry(K, M, L=lock_length)

        self._rng = np.random.default_rng(obs_seed)
        self._agent_state: int = self._START
        self._theta: int = 0

    def reset(
        self,
        seed: int | None = None,
        theta: int | None = None,
    ) -> StepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if theta is None:
            self._theta = int(self._rng.integers(0, self.n_static))
        else:
            if not 0 <= theta < self.n_static:
                raise ValueError(
                    f"theta must be in [0, {self.n_static}), got {theta}"
                )
            self._theta = theta
        self._agent_state = self._START
        obs, obs_per_channel = self._make_obs()
        return StepResult(
            obs=obs, reward=0.0, terminated=False, truncated=False,
            obs_per_channel=obs_per_channel,
        )

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.n_actions:
            raise ValueError(
                f"action must be in [0, {self.n_actions}), got {action}"
            )
        next_dist = self._T[:, self._agent_state, self._theta, action]
        next_state = int(np.argmax(next_dist))
        self._agent_state = next_state

        terminated = next_state in self._terminals
        if next_state == self._SUCCESS:
            reward = 1.0
        elif next_state == self._FAILURE:
            reward = -1.0
        else:
            reward = 0.0

        obs, obs_per_channel = self._make_obs()
        return StepResult(
            obs=obs, reward=reward, terminated=terminated, truncated=False,
            obs_per_channel=obs_per_channel,
        )

    def get_transition_tensor(self) -> np.ndarray:
        return self._T

    def get_observation_tensors(self) -> list[np.ndarray]:
        return self._B_list

    def get_goal(self) -> np.ndarray:
        return self._goal

    def get_modality_registry(self) -> ModalityRegistry:
        return self._registry

    def get_partition_table(self) -> np.ndarray:
        return self._partition_table

    @property
    def theta(self) -> int:
        return self._theta

    @property
    def theta_permutation(self) -> tuple[int, ...]:
        return self._perms[self._theta]

    @property
    def lock_doors(self) -> tuple[int, ...]:
        """The full door sequence the agent must open under the current theta."""
        return self._lock_doors_fn(self._perms[self._theta])

    @property
    def correct_door(self) -> int:
        """The next door the agent must open. For lock_length=1, this is
        the only door; for L>1, it depends on the agent's current stage."""
        if self._agent_state in self._STAGE:
            stage = self._STAGE.index(self._agent_state)
        else:
            stage = 0  # at START or PROBE -> next door is lock_doors[0]
        return self.lock_doors[stage]

    @property
    def agent_state(self) -> int:
        return self._agent_state

    @property
    def start_state_index(self) -> int:
        return self._START

    @property
    def success_state_index(self) -> int:
        return self._SUCCESS

    @property
    def failure_state_index(self) -> int:
        return self._FAILURE

    @property
    def partitions_uniquely_identify_theta(self) -> bool:
        return self._unique_codewords

    def state_to_cell(self, state: int) -> int:
        """Identity helper for runner-level cell-counting.

        For graph envs each state already corresponds to a single
        cell-equivalent (the agent's location is the state itself), so
        the mapping is identity. Spatial sibling envs (e.g.
        MiniGridDoorCodeEnv) override this to collapse heading/stage
        replicas of one cell down to a single index.
        """
        return state

    def _make_obs(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        position = np.zeros(self.n_states)
        position[self._agent_state] = 1.0

        probe = np.zeros(N_PROBE_OBS)
        if self._agent_state in self._PROBE:
            probe_idx = self._PROBE.index(self._agent_state)
            true_bit = int(self._partition_table[probe_idx, self._theta])
            if self.probe_noise > 0:
                flipped = self._rng.random() < self.probe_noise
            else:
                flipped = False
            emitted_bit = (1 - true_bit) if flipped else true_bit
            probe[emitted_bit] = 1.0
        else:
            probe[PROBE_NEUTRAL] = 1.0

        outcome = np.zeros(N_OUTCOME_OBS)
        if self._agent_state == self._SUCCESS:
            outcome[OUTCOME_SUCCESS] = 1.0
        elif self._agent_state == self._FAILURE:
            outcome[OUTCOME_FAILURE] = 1.0
        else:
            outcome[OUTCOME_NEUTRAL] = 1.0

        obs = np.concatenate([position, probe, outcome])
        obs_per_channel = {
            "position": position, "probe": probe, "outcome": outcome,
        }
        return obs, obs_per_channel
