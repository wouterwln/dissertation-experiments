"""Factored Tiger BA-POMDP — canonical multi-door tiger for factored θ.

Following Katt, Oliehoek, Amato 2019 (FBA-POMDP, AAMAS). K independent
doors; behind each is either a tiger (θ_k=0) or treasure (θ_k=1). The
agent LISTENs to disambiguate doors, then OPENs safe doors and avoids
tiger doors.

θ is purely persistent across episodes within a block — no episode-reset
components. This pairs with M4 RockSample (persistent + episodic θ) to
demonstrate that the same ThetaSpec/FactorSpec paradigm handles
persistent-only and persistent+episodic with no machinery change.

State (n_states = 2^K · (2K + 1)):
    flat index of (opened_mask, last_event)
        opened_mask : bitmask of doors opened so far ∈ [0, 2^K)
        last_event  : OTHER=0, LISTEN_0=1, ..., LISTEN_{K-1}=K,
                      OPEN_0=K+1, ..., OPEN_{K-1}=2K

Action (n_actions = 2K):
    0..K-1     : LISTEN_k (no mask change; emits noisy bit on door k)
    K..2K-1    : OPEN_k   (if not opened, flips opened_mask[k]=1)

Theta (component_sizes = (2,) * K, episode_reset_components = ()):
    components 0..K-1 : θ_k ∈ {tiger=0, treasure=1}, persistent

Modalities:
    last_event : θ-INDEPENDENT identity over the 2K+1 event values,
                 cardinality 2K+1. Analogous to RockSample's `position`
                 channel — gives the BAMCP tree-branching observation and
                 a re-grounding signal for the Bayesian filter.
    door_k for each k in [0, K): θ-CONDITIONAL on (θ_k,), cardinality 3
        {bit_0, bit_1, no_info}
        - On LISTEN_k: emit θ_k with prob listen_accuracy, flipped with
          prob 1 - listen_accuracy.
        - On OPEN_k with opened_mask[k] == 1 (the just-flipped reveal):
          emit θ_k deterministically.
        - Otherwise: no_info.

Tensor shapes:
    T            : (n_states, n_states, n_actions)        θ-independent
    B_last_event : (2K+1, n_states)                       θ-independent
    B_door_k     : (3, n_states, 2)                       θ-conditional on (θ_k,)

Reward:
    OPEN_k on unopened door (opened_mask[k] == 0):
        +open_safe_reward      if θ_k == 1
        -open_tiger_penalty    if θ_k == 0
    LISTEN_k:                  -listen_cost
    Otherwise (re-OPEN, etc.): 0

Default ``listen_cost = 0.0`` — Katt 2019's canonical schedule uses
``listen_cost = 1.0``, but AIF/VBP's goal prior is over states (not
actions), so a per-action LISTEN cost would put exact-BP planners at
a structural disadvantage vs sample-based BAMCP/PSRL (which see env
reward directly). Override via CLI / params for canonical-numerics runs.

The AIF/VBP goal prior is a softmax preference over states parameterised
by good_logit (per opened safe door) and bad_logit (per opened tiger door).
The tree-search planners (BAMCP/Factored-BAMCP/BOSS/IDS) ignore the goal
prior and consume the env reward directly; PSRL and Oracle wrap VBP and
therefore do read it.

Termination: bounded horizon (max_steps; runner-controlled, no env-side
              terminal absorbing state).
"""

from __future__ import annotations

import numpy as np

from learningpomdp.environments.base import SAFE_LOG_FLOOR, StepResult
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.observations import (
    ModalityRegistry,
    ObservationModality,
    ThetaDependence,
)


# Door-channel observation indices.
DOOR_BIT_0 = 0
DOOR_BIT_1 = 1
DOOR_NO_INFO = 2
N_DOOR_OBS = 3

# last_event values.
EVENT_OTHER = 0


# --- Sizing helpers ---------------------------------------------------------


def n_events_for(K: int) -> int:
    """OTHER + K LISTEN_k + K OPEN_k."""
    return 2 * K + 1


def n_states_for(K: int) -> int:
    return (2 ** K) * n_events_for(K)


def n_actions_for(K: int) -> int:
    return 2 * K  # K LISTEN + K OPEN


# --- Encoding / decoding ----------------------------------------------------


def encode_state(mask: int, event: int, K: int) -> int:
    return mask * n_events_for(K) + event


def decode_state(state: int, K: int) -> tuple[int, int]:
    n_events = n_events_for(K)
    return state // n_events, state % n_events


def action_role(action: int, K: int) -> tuple[str, int]:
    """Return ('listen', k) | ('open', k)."""
    if action < 0 or action >= 2 * K:
        raise ValueError(f"action out of range: {action}")
    if action < K:
        return ("listen", action)
    return ("open", action - K)


# --- Tensor builders --------------------------------------------------------


def create_transition_tensor(K: int) -> np.ndarray:
    """T with shape (n_states, n_states, n_actions); θ-independent."""
    n_states = n_states_for(K)
    n_actions = n_actions_for(K)
    T = np.zeros((n_states, n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        mask, _ = decode_state(state, K)
        for action in range(n_actions):
            role, payload = action_role(action, K)
            if role == "listen":
                new_mask = mask
                new_event = 1 + payload  # LISTEN_payload
            else:  # open
                if not ((mask >> payload) & 1):
                    new_mask = mask | (1 << payload)
                    new_event = 1 + K + payload  # OPEN_payload
                else:
                    new_mask = mask
                    new_event = EVENT_OTHER
            new_state = encode_state(new_mask, new_event, K)
            T[new_state, state, action] = 1.0
    return T


def create_observation_tensors(
    K: int, listen_accuracy: float,
) -> list[np.ndarray]:
    """Return [B_last_event, B_door_0, ..., B_door_{K-1}].

    B_last_event is θ-independent, shape (2K+1, n_states), identity on
    the event axis.
    Each B_door_r has shape (3, n_states, 2) — θ-conditional on (θ_r,).
    """
    if not 0.5 <= listen_accuracy <= 1.0:
        raise ValueError(
            f"listen_accuracy must be in [0.5, 1.0], got {listen_accuracy}"
        )
    n_states = n_states_for(K)
    n_events = n_events_for(K)
    states = np.arange(n_states)
    mask = states // n_events
    event = states % n_events

    # B_last_event: identity on event index.
    B_last_event = np.zeros((n_events, n_states), dtype=np.float64)
    B_last_event[event, states] = 1.0

    B_doors: list[np.ndarray] = []
    for r in range(K):
        B_r = np.zeros((N_DOOR_OBS, n_states, 2), dtype=np.float64)
        listen_event_r = 1 + r
        open_event_r = 1 + K + r
        mask_r = (mask >> r) & 1

        is_listen_r = (event == listen_event_r)
        is_open_revealed = (event == open_event_r) & (mask_r == 1)
        is_other = ~is_listen_r & ~is_open_revealed

        # no_info default for non-trigger states (constant across θ).
        B_r[DOOR_NO_INFO, is_other, :] = 1.0

        # LISTEN_r: emit θ_r with prob listen_accuracy.
        for theta_k in (0, 1):
            B_r[theta_k, is_listen_r, theta_k] = listen_accuracy
            B_r[1 - theta_k, is_listen_r, theta_k] = 1.0 - listen_accuracy
            # OPEN_r reveal: deterministic θ_r emission.
            B_r[theta_k, is_open_revealed, theta_k] = 1.0

        B_doors.append(B_r)
    return [B_last_event] + B_doors


def create_goal(
    K: int,
    good_logit: float = 2.0,
    bad_logit: float = 4.0,
) -> np.ndarray:
    """(n_states, n_static) softmax over states for each θ.

    Higher preference for states with more "safe doors opened" (mask
    bit set with θ_k=1); lower for "tiger doors opened". Asymmetric
    ``bad_logit > good_logit`` so the *expected* log C of OPEN under
    uniform θ belief is strictly negative — preventing AIF/VBP from
    selecting OPEN purely for info gain.

    Note: the default ``(good_logit, bad_logit) = (2.0, 4.0)`` is
    hand-picked to satisfy the negative-expected-log-C property; its
    1:2 ratio is intentionally *decoupled* from the reward function's
    ``open_safe_reward : open_tiger_penalty`` ratio (1:10 by default).
    Treat these as design knobs, not tuned values.

    What the transformation preserves
    --------------------------------
    For one door k with belief ``p = P(θ_k = 1)``, the two objectives the
    paper compares are both affine and strictly increasing in ``p``::

        E[Δ log C | OPEN_k]  = p·good_logit       - (1-p)·bad_logit
        E[Δ reward | OPEN_k] = p·open_safe_reward - (1-p)·open_tiger_penalty

    The softmax normaliser is per-θ and state-independent, so it cancels
    exactly in every state-to-state difference and hence in the
    expectation above. Both objectives are sums of these per-door terms,
    so additivity across doors is preserved as well.

    Preserved for *any* positive (good_logit, bad_logit):

    - **Monotonicity in the belief.** Both objectives induce a threshold
      rule on the same sufficient statistic ``p``: if OPEN is preferred at
      ``p`` it is preferred at every ``p' > p``.
    - **The ranking of doors.** At a fixed belief both objectives prefer
      opening the door with the largest ``p``, so the commitment *target*
      never differs between them.

    Not preserved:

    - **The threshold location.** ``E[Δ log C]`` crosses zero at
      ``p = bad_logit / (good_logit + bad_logit)``; ``E[Δ reward]`` at
      ``p = open_tiger_penalty / (open_safe_reward + open_tiger_penalty)``.
      These coincide iff ``good_logit : bad_logit == open_safe_reward :
      open_tiger_penalty``. The canonical (2, 4) puts the goal prior's
      threshold at 0.67 against the reward's 0.91 — i.e. it *under*-
      penalises tigers relative to the reward the results are scored on,
      so AIF/VBP commit at lower confidence than reward-optimality and
      pay for it in the reported return. The asymmetry is not a margin
      granted to the exact-BP planners here. The affine form and the
      ``bad/(good+bad)`` crossing are checked numerically in
      ``tests/test_preference_sweeps.py``, not just asserted here.
    - **Magnitude.** ``log C`` is in nats, reward in env units; only the
      sign/threshold structure above can be aligned, never the scale.
    - **Per-action costs.** ``listen_cost`` has no representation at all:
      the goal prior is a distribution over *states*, so no choice of
      logits can encode a cost attached to the LISTEN *action*. This is
      why the default is 0.0 rather than Katt 2019's 1.0 — at
      ``listen_cost > 0`` the two objectives are not sign-equivalent for
      any (good_logit, bad_logit), and the gap is a structural limit of
      the transformation rather than a tuning choice. Because "we could
      not represent it, so we set it to zero" is not an answer, the
      ``factored_tiger_listen_cost`` DVC stage runs one AIF arm at the
      canonical goal prior and Katt's ``listen_cost = 1.0``, and the
      preference figure reports it against the (2, 4) cell — so the size
      of the deviation is measured rather than assumed.

    Finally, the argument above covers only the pragmatic (expected
    log-preference) term of the Expected Free Energy. The epistemic term
    has no reward counterpart — it is an addition to the objective, not a
    re-encoding of it. That is precisely why the ratio matters: with
    ``good_logit >= bad_logit`` the epistemic term can make OPEN
    attractive purely for the information the reveal yields.

    The ``factored_tiger_preference`` DVC stage sweeps
    ``good_logit × bad_logit`` at reduced seed count and includes the
    reward-ratio-matched cell (1, 10), so the sensitivity of the reported
    results to this knob is measured rather than assumed. (1, 10) rather
    than (2, 20) because only the *ratio* sets the commit threshold,
    while the pair's overall scale is bounded from above by float32: with
    ``bad_logit >= good_logit`` the logits span ``bad_logit·K`` nats, so
    the smallest normalised entry is ≈ ``exp(-bad_logit·K)`` (6.7e-28 at
    K=6, bad=10; zero in float32 at bad=20), and ``messages.safe_log``
    clamps everything below ``1e-30`` onto a single floor value. Past
    that clamp the objective can no longer tell "4 tigers opened" from
    "6", which is why this function refuses to return such a goal.
    """
    n_states = n_states_for(K)
    n_static = 2 ** K
    component_sizes = (2,) * K
    n_events = n_events_for(K)

    states = np.arange(n_states)
    mask = states // n_events
    mask_bits = ((mask[:, None] >> np.arange(K)) & 1).astype(np.int32)

    goal_logits = np.zeros((n_states, n_static), dtype=np.float64)
    for theta_flat in range(n_static):
        components = np.unravel_index(theta_flat, component_sizes)
        theta_bits = np.array(components, dtype=np.int32)
        n_safe = (mask_bits * theta_bits[None, :]).sum(axis=1)
        n_tiger = (mask_bits * (1 - theta_bits[None, :])).sum(axis=1)
        goal_logits[:, theta_flat] = good_logit * n_safe - bad_logit * n_tiger

    logits_centered = goal_logits - goal_logits.max(axis=0, keepdims=True)
    g = np.exp(logits_centered)
    goal = g / g.sum(axis=0, keepdims=True)
    # The planners consume float32 log C via messages.safe_log, which
    # clamps at log(max(C, 1e-30)); below that the goal is silently
    # truncated onto one floor value. Checked in float64 — the float32
    # cast is exact to ~1e-7 relative this far above the subnormal range.
    if goal.min() < SAFE_LOG_FLOOR:
        raise ValueError(
            f"goal prior underflows the safe_log floor at K={K}, "
            f"(good_logit, bad_logit)=({good_logit}, {bad_logit}): min "
            f"entry {goal.min():.3e} < {SAFE_LOG_FLOOR:.0e}. The logits "
            f"span max(good_logit, bad_logit)*K nats; lower them (keeping "
            f"the ratio) or lower K."
        )
    return goal


def informative_states_for_door(K: int, r: int) -> frozenset[int]:
    """States where door_r's emission is θ-informative (not no_info)."""
    n_events = n_events_for(K)
    n_states = n_states_for(K)
    listen_event_r = 1 + r
    open_event_r = 1 + K + r

    out: list[int] = []
    for state in range(n_states):
        mask, event = decode_state(state, K)
        mask_r = (mask >> r) & 1
        if event == listen_event_r:
            out.append(state)
        elif event == open_event_r and mask_r == 1:
            out.append(state)
    return frozenset(out)


def create_modality_registry(K: int) -> ModalityRegistry:
    # last_event channel name is "position" to match BAMCP's hardcoded
    # lookup of the tree-branching channel (see _bamcp_numba.py and the
    # Numba dispatch in bamcp_planner.py). It serves the same role: a
    # θ-independent identity-on-state-axis observation used for
    # observation-keyed tree branching.
    modalities: list[ObservationModality] = [
        ObservationModality(
            name="position",
            cardinality=n_events_for(K),
            theta_dependence=ThetaDependence.INDEPENDENT,
        )
    ]
    for r in range(K):
        modalities.append(
            ObservationModality(
                name=f"door_{r}",
                cardinality=N_DOOR_OBS,
                theta_dependence=ThetaDependence.CONDITIONAL,
                informative_states=informative_states_for_door(K, r),
            )
        )
    return ModalityRegistry(modalities=tuple(modalities))


# --- Environment class ------------------------------------------------------


class FactoredTigerEnv:
    """Discrete Factored Tiger BA-POMDP with factored persistent θ.

    Args:
        K: number of doors (>= 2).
        listen_accuracy: P(LISTEN_k emits correct bit | θ_k) ∈ [0.5, 1.0].
        open_safe_reward: reward for OPEN_k on unopened door with θ_k=1.
        open_tiger_penalty: positive magnitude; reward is the negation
            for OPEN_k on unopened door with θ_k=0.
        listen_cost: positive magnitude; reward is the negation per LISTEN.
        good_logit / bad_logit: AIF goal-prior asymmetry. ``bad_logit >
            good_logit`` keeps OPEN-for-info-gain off the table; the
            tree-search planners (BAMCP/Factored-BAMCP/BOSS/IDS) consume
            env reward directly and ignore these, while PSRL and Oracle
            read them via VBP. Defaults are hand-picked design knobs
            (1:2 ratio), intentionally decoupled from the reward
            magnitudes — see ``create_goal`` docstring for the rationale.
        obs_seed: stochastic emission RNG seed.
        max_steps: informational; runner enforces actual termination.
    """

    def __init__(
        self,
        K: int = 4,
        listen_accuracy: float = 0.85,
        open_safe_reward: float = 10.0,
        open_tiger_penalty: float = 100.0,
        listen_cost: float = 0.0,
        good_logit: float = 2.0,
        bad_logit: float = 4.0,
        obs_seed: int | None = 0,
        max_steps: int = 20,
    ) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")

        self.K = K
        self.listen_accuracy = float(listen_accuracy)
        self.open_safe_reward = float(open_safe_reward)
        self.open_tiger_penalty = float(open_tiger_penalty)
        self.listen_cost = float(listen_cost)
        self.max_steps = int(max_steps)

        self.n_states = n_states_for(K)
        self.n_actions = n_actions_for(K)
        self.theta_spec = ThetaSpec(
            component_sizes=(2,) * K,
            episode_reset_components=(),
        )
        self.n_static = self.theta_spec.n_static
        # T is θ-independent.
        self.t_factor_spec = FactorSpec(theta_components=())
        # B_last_event is θ-independent; each B_door_r depends only on θ_r.
        self.b_factor_specs: tuple[FactorSpec, ...] = (
            FactorSpec(theta_components=()),
        ) + tuple(
            FactorSpec(theta_components=(r,)) for r in range(K)
        )

        self._T = create_transition_tensor(K)
        self._B_list = create_observation_tensors(K, listen_accuracy)
        self._goal = create_goal(K, good_logit=good_logit, bad_logit=bad_logit)
        self._registry = create_modality_registry(K)

        self._rng = np.random.default_rng(obs_seed)
        self._theta: int = 0
        self._theta_components: tuple[int, ...] = (0,) * K
        self._agent_state: int = 0
        self._step_count: int = 0

    # --- Environment protocol ----------------------------------------------

    @property
    def start_state_index(self) -> int:
        return encode_state(mask=0, event=EVENT_OTHER, K=self.K)

    @property
    def success_state_index(self) -> int:
        # Sentinel: Factored Tiger has no terminal SUCCESS state, so any
        # ``agent_state == success_state_index`` check is permanently False.
        # ``rock_sample.py`` has the same shape — both could be harmonized
        # to this sentinel in a follow-up pass.
        return -1

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
        self._theta_components = tuple(
            int(c) for c in np.unravel_index(
                self._theta, self.theta_spec.component_sizes,
            )
        )
        self._agent_state = self.start_state_index
        self._step_count = 0
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
        next_dist = self._T[:, self._agent_state, action]
        next_state = int(np.argmax(next_dist))

        # Compute reward against the pre-transition mask.
        mask, _ = decode_state(self._agent_state, self.K)
        role, payload = action_role(action, self.K)
        reward = 0.0
        if role == "listen":
            reward = -self.listen_cost
        else:  # open
            k = payload
            if not ((mask >> k) & 1):
                theta_k = self._theta_components[k]
                reward = (
                    self.open_safe_reward if theta_k == 1
                    else -self.open_tiger_penalty
                )

        self._agent_state = next_state
        self._step_count += 1
        obs, obs_per_channel = self._make_obs()
        return StepResult(
            obs=obs, reward=float(reward),
            terminated=False, truncated=False,
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

    # --- Accessors ---------------------------------------------------------

    @property
    def theta(self) -> int:
        return self._theta

    @property
    def theta_components(self) -> tuple[int, ...]:
        return self._theta_components

    @property
    def agent_state(self) -> int:
        return self._agent_state

    # --- Observation generation -------------------------------------------

    def _make_obs(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        K = self.K
        n_events = n_events_for(K)
        mask, event = decode_state(self._agent_state, K)

        position = np.zeros(n_events)
        position[event] = 1.0

        per_channel: dict[str, np.ndarray] = {"position": position}
        door_obs_concat: list[np.ndarray] = []
        for r in range(K):
            door = np.zeros(N_DOOR_OBS)
            listen_event_r = 1 + r
            open_event_r = 1 + K + r
            mask_r = (mask >> r) & 1
            theta_k = self._theta_components[r]

            if event == listen_event_r:
                emit_truth = self._rng.random() < self.listen_accuracy
                bit = theta_k if emit_truth else 1 - theta_k
                door[bit] = 1.0
            elif event == open_event_r and mask_r == 1:
                door[theta_k] = 1.0
            else:
                door[DOOR_NO_INFO] = 1.0

            per_channel[f"door_{r}"] = door
            door_obs_concat.append(door)

        obs = np.concatenate([position] + door_obs_concat)
        return obs, per_channel
