"""Reactivity Maze environment (chapter-3 Epistemic Maze, ported to the ch5 AIF-MP contract).

The Minimal Reactive Epistemic Maze: the agent must both (a) learn a hidden goal
identity θ by visiting an instructional cue, and (b) keep its "reactivity knob" at
maximum so that navigation stays controllable. Reward is delivered only at the end
of the episode (terminal-goal task).

State space (35 flat states = 7 locations × 5 knob levels):
    x = location + N_LOCATIONS * knob
    location 0-4: navigation states (the goal is nav state θ)
    location 5:   safe sink (absorbing, guaranteed +0.33)
    location 6:   instructional cue (reveals θ)
    knob 0-4:     reactivity; action success probability = knob / 4

Static state θ (n_static = n_theta):
    θ ∈ {0, ..., n_theta-1} names which navigation state is the goal. It is hidden
    and only observable at the cue location. This maps onto the ch5 "config" axis
    exactly the way RockSample's hidden rock qualities do.

Actions (8):
    0-4: navigate to (location + action) mod 5   (stochastic: succeeds w.p. knob/4,
         otherwise falls to the safe sink)
    5:   decrease knob   6: increase knob         (same knob/4 success dynamics)
    7:   visit cue                                 (deterministic move to the cue)

Observation channels (shape (7 + 1, n_theta + 1, 35, n_static)):
    channels 0..6:  one per location, binary present/absent, θ-independent, always
                    informative (only outcomes {0, 1} are used)
    channel 7:      a SINGLE categorical θ modality over n_theta+1 outcomes (one
                    signal per θ value + a deterministic neutral outcome at index
                    n_theta), event-gated on the cue location — informative at the
                    cue, deterministic neutral elsewhere. This matches chapter-3's
                    theta_observation modality exactly.

Reward (terminal only, exactly the chapter-3 structure):
    +1.00  correct goal (location == θ) with max reactivity (knob == 4)
    +0.33  safe sink (location == 5), any knob
    -1.00  wrong navigation state with max reactivity (knob == 4)
    -0.33  any non-sink state with knob < 4
     0.00  otherwise (e.g. cue with knob == 4)
"""

import numpy as np
from dataclasses import dataclass

# Actions
NAV_0 = 0
NAV_1 = 1
NAV_2 = 2
NAV_3 = 3
NAV_4 = 4
DECREASE_KNOB = 5
INCREASE_KNOB = 6
VISIT_CUE = 7
N_ACTIONS = 8

# Locations
SAFE_SINK = 5
CUE = 6

# Environment constants
N_LOCATIONS = 7
N_KNOB_STATES = 5
N_NAV_STATES = 5
N_STATES = N_LOCATIONS * N_KNOB_STATES  # 35
MAX_KNOB = 4


def state_index(location: int, knob: int) -> int:
    """Flat state index: x = location + N_LOCATIONS * knob."""
    return location + N_LOCATIONS * knob


def unpack_state(x: int) -> tuple[int, int]:
    """Unpack flat state index into (location, knob)."""
    return x % N_LOCATIONS, x // N_LOCATIONS


def sample_configs(n_theta: int) -> np.ndarray:
    """Return the static configurations (goal identities).

    Each config θ names the navigation state that is the goal, so config θ maps to
    goal location θ. n_static == n_theta.

    Args:
        n_theta: number of possible goal identities (1..N_NAV_STATES)

    Returns:
        goals: (n_theta,) int array. goals[θ] == θ (the goal nav location for θ).
    """
    if not 1 <= n_theta <= N_NAV_STATES:
        raise ValueError(f"n_theta must be 1..{N_NAV_STATES}, got {n_theta}")
    return np.arange(n_theta, dtype=np.int32)


def generate_transition_tensor(n_theta: int, cue_dropoff: int | None = None) -> np.ndarray:
    """Generate transition tensor T(x_new, x_old, θ, action).

    Dynamics are θ-independent (θ only names the goal, it does not affect physics),
    so the tensor is broadcast across the static axis. Reproduces the chapter-3
    epistemic-maze dynamics exactly:
      - cue: any action -> nav-state drop-off (same knob), see ``cue_dropoff``
      - safe sink: absorbing
      - knob +/-: succeed w.p. knob/4 (stay, change knob); else fall to safe sink
      - visit cue: deterministic move to the cue
      - nav 0-4: succeed w.p. knob/4 to (loc+a) mod 5; else fall to safe sink

    Args:
        n_theta: number of goal identities (n_static)
        cue_dropoff: where the agent lands after the cue. ``None`` (default) is the
            canonical dynamics: uniform over the 5 nav states. An int in
            ``0..N_NAV_STATES-1`` makes the drop-off deterministic to that nav
            state (used by the fixed-drop-off ablation).

    Returns:
        T: (35, 35, n_theta, 8) float32
    """
    if cue_dropoff is not None and not 0 <= cue_dropoff < N_NAV_STATES:
        raise ValueError(f"cue_dropoff must be None or 0..{N_NAV_STATES - 1}, got {cue_dropoff}")

    T3 = np.zeros((N_STATES, N_STATES, N_ACTIONS), dtype=np.float32)

    for s in range(N_STATES):
        loc, knob = unpack_state(s)
        for a in range(N_ACTIONS):
            # Cue: any action -> nav-state drop-off (same knob). Canonically this
            # is uniform over the 5 nav states; with cue_dropoff set it is a
            # deterministic move to that single nav state.
            if loc == CUE:
                if cue_dropoff is None:
                    for nl in range(N_NAV_STATES):
                        T3[state_index(nl, knob), s, a] += 1.0 / N_NAV_STATES
                else:
                    T3[state_index(cue_dropoff, knob), s, a] += 1.0
                continue

            # Safe sink: absorbing
            if loc == SAFE_SINK:
                T3[s, s, a] = 1.0
                continue

            success = knob / MAX_KNOB
            fail = 1.0 - success

            if a == DECREASE_KNOB:
                new_knob = max(0, knob - 1)
                T3[state_index(loc, new_knob), s, a] += success
                T3[state_index(SAFE_SINK, new_knob), s, a] += fail
                continue

            if a == INCREASE_KNOB:
                new_knob = min(MAX_KNOB, knob + 1)
                T3[state_index(loc, new_knob), s, a] += success
                T3[state_index(SAFE_SINK, new_knob), s, a] += fail
                continue

            if a == VISIT_CUE:
                T3[state_index(CUE, knob), s, a] = 1.0
                continue

            # Navigation actions 0-4
            next_loc = (loc + a) % N_NAV_STATES
            T3[state_index(next_loc, knob), s, a] += success
            T3[state_index(SAFE_SINK, knob), s, a] += fail

    # Broadcast across the θ (static) axis: (x_new, x_old, θ, a)
    T = np.broadcast_to(T3[:, :, None, :], (N_STATES, N_STATES, n_theta, N_ACTIONS))
    return np.ascontiguousarray(T, dtype=np.float32)


def generate_observation_tensor(
    n_theta: int,
    cue_accuracy: float = 1.0,
    pos_noise: float = 0.05,
) -> np.ndarray:
    """Generate observation tensor B(channel, obs_type, x, θ).

    Chapter-3-EXACT θ observation: a SINGLE categorical θ channel with
    ``n_theta + 1`` outcomes — one signal per θ value plus a deterministic
    *neutral* outcome at index ``n_theta`` — event-gated on the cue location.
    This mirrors chapter-3's ``theta_observation`` modality (Epistemic Maze,
    ``create_epistemic_maze_tensors``) exactly, rather than the earlier
    n_theta independent binary cue channels.

    Channels (n_obs_types = n_theta + 1):
      - location channels 0..6: binary present/absent (outcomes {0, 1} only),
        θ- and knob-independent (location is always observable).
      - θ channel (index 7): categorical over n_theta+1 outcomes. At the cue,
        outcome j fires with p(j | cue, θ=j) = cue_accuracy and the wrong
        signals share (1 - cue_accuracy)/(n_theta - 1); everywhere else the
        deterministic neutral outcome (index n_theta) fires. The neutral outcome
        makes the off-cue reading uninformative about θ AND low-entropy, and —
        as in ch3 — a non-neutral θ reading also implies "at the cue" (its signal
        outcomes have probability 0 in every non-cue state), which sharpens
        location inference.

    Args:
        n_theta: number of goal identities (n_static)
        cue_accuracy: probability the correct θ signal fires at the cue
        pos_noise: location channel noise

    Returns:
        B: (7 + 1, n_theta + 1, 35, n_theta) float32
    """
    neutral = n_theta                    # index of the neutral θ outcome
    n_obs_types = max(2, n_theta + 1)    # location uses {0,1}; θ uses 0..n_theta
    n_channels = N_LOCATIONS + 1         # 7 location channels + 1 categorical θ channel
    theta_ch = N_LOCATIONS

    p_tp_loc = float(np.clip(1.0 - pos_noise, 0.0, 1.0))
    p_fp_loc = float(np.clip(pos_noise * 0.1, 0.0, 1.0))

    B = np.zeros((n_channels, n_obs_types, N_STATES, n_theta), dtype=np.float32)

    for x in range(N_STATES):
        loc, _ = unpack_state(x)

        # Location channels (θ-independent binary present/absent)
        for c in range(N_LOCATIONS):
            p = p_tp_loc if loc == c else p_fp_loc
            B[c, 1, x, :] = p
            B[c, 0, x, :] = 1.0 - p

        # Categorical θ channel (event-gated on the cue location)
        if loc == CUE:
            other = (1.0 - cue_accuracy) / (n_theta - 1) if n_theta > 1 else 0.0
            for theta in range(n_theta):
                for j in range(n_theta):
                    B[theta_ch, j, x, theta] = cue_accuracy if j == theta else other
        else:
            # Off-cue: deterministic neutral outcome, θ-independent (matches ch3)
            B[theta_ch, neutral, x, :] = 1.0

    return B


def generate_goal(
    n_theta: int,
    goal_temperature: float = 1.0,
    goal_reward: float = 1.0,
    wrong_penalty: float = 1.0,
    sink_reward: float = 0.33,
    lowknob_penalty: float = 1.0,
) -> np.ndarray:
    """Generate per-config preference factor C(x, θ) via softmax over states.

    Reproduces the chapter-3 reward structure as soft preferences. Because the
    goal navigation state is θ-dependent while the wrong-goal penalty is symmetric,
    under a uniform θ belief the expected preference of committing to any nav state
    is <= that of the safe sink — so the agent has no reason to commit before
    resolving θ at the cue. This is the same "guard against acting-for-nothing"
    that RockSample builds with bad_logit > good_logit.

    Args:
        n_theta: number of goal identities (n_static)
        goal_temperature: softmax temperature (lower = more peaked)
        goal_reward: logit at the correct goal (knob == 4)
        wrong_penalty: logit penalty at a wrong nav goal (knob == 4)
        sink_reward: logit at the safe sink
        lowknob_penalty: logit penalty for any non-sink state with knob < 4

    Returns:
        goal: (35, n_theta) per-config softmax preference (normalized over states)
    """
    logits = np.zeros((N_STATES, n_theta), dtype=np.float64)

    for x in range(N_STATES):
        loc, knob = unpack_state(x)
        for theta in range(n_theta):
            if loc == SAFE_SINK:
                v = sink_reward
            elif loc < N_NAV_STATES and knob == MAX_KNOB:
                v = goal_reward if loc == theta else -wrong_penalty
            elif loc != SAFE_SINK and knob < MAX_KNOB:
                v = -lowknob_penalty
            else:
                # cue with knob == 4: mildly avoid (not a terminal destination)
                v = -wrong_penalty
            logits[x, theta] = v

    scaled = logits / goal_temperature
    scaled -= scaled.max(axis=0, keepdims=True)  # per-config numerical stability
    goal = np.exp(scaled)
    goal /= goal.sum(axis=0, keepdims=True)
    return goal.astype(np.float32)


def get_outcome(location: int, knob: int, theta: int) -> str:
    """Classify a terminal state: 'goal', 'safe_sink', 'wrong_goal', 'penalty', 'other'."""
    if location == SAFE_SINK:
        return "safe_sink"
    if location < N_NAV_STATES and knob == MAX_KNOB:
        return "goal" if location == theta else "wrong_goal"
    if location != SAFE_SINK and knob < MAX_KNOB:
        return "penalty"
    return "other"


def get_final_reward(location: int, knob: int, theta: int) -> float:
    """Terminal reward for the chapter-3 reward structure."""
    outcome = get_outcome(location, knob, theta)
    return {
        "goal": 1.0,
        "safe_sink": 0.33,
        "wrong_goal": -1.0,
        "penalty": -0.33,
        "other": 0.0,
    }[outcome]


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class ReactivityStepResult:
    obs: np.ndarray  # (7 + 1,) per-channel outcome indices (location binary; θ categorical)
    reward: float
    terminated: bool
    truncated: bool


class ReactivityMazeEnv:
    """Reactivity Maze simulator with a fixed start and terminal-only reward.

    The agent always starts at navigation location 0 with knob = MAX_KNOB (so its
    initial state is known to the agent, matching the ch5 delta-belief reset). The
    hidden goal identity θ (the config) is randomized per episode. The episode runs
    for exactly ``max_steps`` steps and the reward is delivered on the final step
    based on the terminal state (there is no early termination and no death).

    Args:
        goals: (n_theta,) config array (goals[θ] == θ)
        obs_tensor: (7 + 1, n_theta + 1, 35, n_theta) observation tensor
        max_steps: number of steps per episode (terminal reward at the last step)
        start_location: fixed start navigation location (default 0)
        cue_dropoff: where the agent lands after the cue. ``None`` (default) is the
            canonical uniform-random drop-off over the 5 nav states; an int in
            ``0..N_NAV_STATES-1`` makes it deterministic to that nav state. Must
            match the ``cue_dropoff`` used to build the agent's transition tensor.
    """

    def __init__(
        self,
        goals: np.ndarray,
        obs_tensor: np.ndarray | None = None,
        max_steps: int = 5,
        start_location: int = 0,
        cue_dropoff: int | None = None,
    ):
        if cue_dropoff is not None and not 0 <= cue_dropoff < N_NAV_STATES:
            raise ValueError(f"cue_dropoff must be None or 0..{N_NAV_STATES - 1}, got {cue_dropoff}")
        self.goals = np.asarray(goals)
        self.n_theta = int(self.goals.shape[0])
        self.obs_tensor = obs_tensor
        self._max_steps = max_steps
        self.start_location = start_location
        self.cue_dropoff = cue_dropoff
        self.start_state_idx = state_index(start_location, MAX_KNOB)

        self._rng = np.random.default_rng(0)
        self._location = start_location
        self._knob = MAX_KNOB
        self._config_idx = 0
        self._steps = 0
        self._has_seen_cue = False

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def config_idx(self) -> int:
        return self._config_idx

    @property
    def theta(self) -> int:
        """The true goal navigation location for the current config."""
        return int(self.goals[self._config_idx])

    @property
    def _state_idx(self) -> int:
        return state_index(self._location, self._knob)

    def reset(self, seed: int | None = None, config_idx: int | None = None) -> ReactivityStepResult:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if config_idx is not None:
            self._config_idx = config_idx
        else:
            self._config_idx = int(self._rng.integers(0, self.n_theta))

        self._location = self.start_location
        self._knob = MAX_KNOB
        self._steps = 0
        self._has_seen_cue = self._location == CUE

        return ReactivityStepResult(
            obs=self._get_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def _execute_transition(self, action: int) -> None:
        loc, knob = self._location, self._knob

        # Cue: any action -> nav-state drop-off (same knob). Uniform random by
        # default; deterministic to cue_dropoff when set (fixed-drop-off ablation).
        if loc == CUE:
            if self.cue_dropoff is None:
                self._location = int(self._rng.integers(0, N_NAV_STATES))
            else:
                self._location = self.cue_dropoff
            return

        # Safe sink: absorbing
        if loc == SAFE_SINK:
            return

        success_prob = knob / MAX_KNOB

        if action == DECREASE_KNOB:
            self._knob = max(0, knob - 1)
            if self._rng.random() >= success_prob:
                self._location = SAFE_SINK
            return

        if action == INCREASE_KNOB:
            self._knob = min(MAX_KNOB, knob + 1)
            if self._rng.random() >= success_prob:
                self._location = SAFE_SINK
            return

        if action == VISIT_CUE:
            self._location = CUE
            return

        # Navigation actions 0-4
        if self._rng.random() < success_prob:
            self._location = (loc + action) % N_NAV_STATES
        else:
            self._location = SAFE_SINK

    def step(self, action: int) -> ReactivityStepResult:
        self._steps += 1
        self._execute_transition(action)

        if self._location == CUE:
            self._has_seen_cue = True

        truncated = self._steps >= self._max_steps
        # Terminal-only reward: evaluated on the final state, no early termination.
        reward = get_final_reward(self._location, self._knob, self.theta) if truncated else 0.0

        return ReactivityStepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=False,
            truncated=truncated,
        )

    def get_outcome(self) -> str:
        return get_outcome(self._location, self._knob, self.theta)

    @property
    def has_seen_cue(self) -> bool:
        return self._has_seen_cue

    def _get_obs(self) -> np.ndarray:
        """Sample per-channel categorical observations from the observation model.

        Returns one outcome index per channel: location channels are binary
        (0=absent, 1=present); the θ channel (last) is a categorical over
        n_theta+1 outcomes (θ signals 0..n_theta-1 plus the neutral index n_theta).
        """
        x = self._state_idx
        n_channels, n_obs_types = self.obs_tensor.shape[0], self.obs_tensor.shape[1]
        obs = np.zeros(n_channels, dtype=np.float32)
        for c in range(n_channels):
            p = np.asarray(self.obs_tensor[c, :, x, self._config_idx], dtype=np.float64)
            p = p / p.sum()
            obs[c] = self._rng.choice(n_obs_types, p=p)
        return obs

    def render_ascii(self) -> str:
        loc_names = ["Nav0", "Nav1", "Nav2", "Nav3", "Nav4", "Sink", "Cue "]
        marker = "*" if self._location < N_NAV_STATES and self._location == self.theta else " "
        return (
            f"loc={loc_names[self._location]}{marker} knob={self._knob} "
            f"theta={self.theta} seen_cue={self._has_seen_cue} step={self._steps}"
        )
