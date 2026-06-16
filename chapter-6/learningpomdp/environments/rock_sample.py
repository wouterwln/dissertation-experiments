"""Rock-sample BA-POMDP for the M4 factorized-θ benchmark.

A grid_size × grid_size grid with K rocks at fixed (layout-seeded) cells.
Each rock has a binary quality (good/bad) and a binary sensor reliability,
both modeled as θ components. Quality θ resets to a uniform prior at every
episode reset within a block; sensor θ persists across the block.

State (n_states = grid_size² · 2^K · (K+2)):
    encoded as a flat index of (pos, sampled_mask, last_event)
        pos          : agent cell ∈ [0, grid_size²)
        sampled_mask : bitmask of which rocks are sampled ∈ [0, 2^K)
        last_event   : OTHER=0, SENSE_0=1, ..., SENSE_{K-1}=K, SAMPLE=K+1

Action (n_actions = K + 5):
    0..3       : MOVE_{N, S, E, W}     (deterministic, clamped at edges)
    4..4+K-1   : SENSE_k               (no movement; emits noisy bit)
    4+K        : SAMPLE                (only triggers at a rock cell with
                                        mask[r]=0; flips mask[r]=1 and
                                        reveals quality_r)

Theta (component_sizes = (2,) * (2*K), episode_reset_components = 0..K-1):
    components 0..K-1   : θ_quality_r ∈ {bad=0, good=1}, episode-reset
    components K..2K-1  : θ_sensor_r ∈ {reliable=0, noisy=1}, persistent

Modalities:
    position       : θ-INDEPENDENT identity over grid cells, cardinality grid_size²
    rock_0..K-1    : θ-CONDITIONAL on (θ_quality_r, θ_sensor_r), cardinality 3
                     {bit_0=0, bit_1=1, no_info=2}

    rock_r emission rules at state (pos, mask, event):
      - event == SENSE_r              → noisy bit per α_eff(d_r, θ_sensor_r),
                                        flipped according to θ_quality_r
      - event == SAMPLE and pos == rock_r_cell and mask[r] == 1
                                      → deterministic θ_quality_r bit
      - otherwise                     → no_info

Tensor shapes:
    T          : (n_states, n_states, n_actions)         θ-independent
    B_position : (grid_size², n_states)                  θ-independent
    B_rock_r   : (3, n_states, 2, 2)                     for each rock r

Reward:
    Successful SAMPLE on rock r: +sample_reward if θ_quality_r=1 (good),
                                 -sample_reward if θ_quality_r=0 (bad).
    All other steps: 0.

Termination: bounded horizon (max_steps; runner-controlled, no env-side
              terminal absorbing state).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from learningpomdp.environments.base import StepResult
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.observations import (
    ModalityRegistry,
    ObservationModality,
    ThetaDependence,
)


# Rock-channel observation indices.
ROCK_BIT_0 = 0
ROCK_BIT_1 = 1
ROCK_NO_INFO = 2
N_ROCK_OBS = 3

# last_event values.
EVENT_OTHER = 0


# --- Layout -----------------------------------------------------------------


@dataclass(frozen=True)
class RockSampleLayout:
    """Static spatial layout: grid size, rock cells, agent start cell.

    Cells are indexed as `cell = row * grid_size + col`, i.e. row-major
    with (col=0, row=0) at the top-left.
    """

    grid_size: int
    K: int
    agent_start_cell: int
    rock_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.K != len(self.rock_cells):
            raise ValueError(f"K={self.K} != len(rock_cells)={len(self.rock_cells)}")
        if len(set(self.rock_cells)) != self.K:
            raise ValueError(f"rock_cells must be distinct, got {self.rock_cells}")
        if self.agent_start_cell in self.rock_cells:
            raise ValueError(
                f"agent_start_cell={self.agent_start_cell} cannot coincide with a rock"
            )

    @property
    def n_cells(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def n_events(self) -> int:
        # OTHER + K SENSE_k + SAMPLE
        return self.K + 2

    def cell_xy(self, cell: int) -> tuple[int, int]:
        return (cell % self.grid_size, cell // self.grid_size)

    def chebyshev_distance(self, cell: int, rock_idx: int) -> int:
        x1, y1 = self.cell_xy(cell)
        x2, y2 = self.cell_xy(self.rock_cells[rock_idx])
        return max(abs(x1 - x2), abs(y1 - y2))


def sample_layout(grid_size: int, K: int, seed: int) -> RockSampleLayout:
    """Sample K distinct rock cells avoiding (0,0)."""
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2, got {grid_size}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if K >= grid_size * grid_size:
        raise ValueError(
            f"K={K} too large for grid_size={grid_size} ({grid_size**2} cells)"
        )
    rng = np.random.default_rng(seed)
    agent_start_cell = 0
    available = [c for c in range(grid_size * grid_size) if c != agent_start_cell]
    rng.shuffle(available)
    rock_cells = tuple(sorted(available[:K]))
    return RockSampleLayout(
        grid_size=grid_size,
        K=K,
        agent_start_cell=agent_start_cell,
        rock_cells=rock_cells,
    )


# --- Encoding / decoding ----------------------------------------------------


def n_states_for(layout: RockSampleLayout) -> int:
    return layout.n_cells * (2 ** layout.K) * layout.n_events


def n_actions_for(K: int) -> int:
    return 5 + K  # 4 MOVE + K SENSE + 1 SAMPLE


def encode_state(pos: int, mask: int, event: int, K: int, n_events: int) -> int:
    return pos * ((2 ** K) * n_events) + mask * n_events + event


def decode_state(state: int, K: int, n_events: int) -> tuple[int, int, int]:
    block = (2 ** K) * n_events
    pos = state // block
    rest = state % block
    mask = rest // n_events
    event = rest % n_events
    return pos, mask, event


def move_cell(cell: int, direction: int, grid_size: int) -> int:
    """Apply MOVE_d, clamped at grid edges."""
    x = cell % grid_size
    y = cell // grid_size
    if direction == 0:    # N
        y = max(0, y - 1)
    elif direction == 1:  # S
        y = min(grid_size - 1, y + 1)
    elif direction == 2:  # E
        x = min(grid_size - 1, x + 1)
    elif direction == 3:  # W
        x = max(0, x - 1)
    else:
        raise ValueError(f"direction must be in [0,4), got {direction}")
    return y * grid_size + x


def action_role(action: int, K: int) -> tuple[str, int]:
    """Return ('move', d) | ('sense', k) | ('sample', 0)."""
    if action < 0 or action >= 5 + K:
        raise ValueError(f"action out of range: {action}")
    if action < 4:
        return ("move", action)
    if action < 4 + K:
        return ("sense", action - 4)
    return ("sample", 0)


# --- Tensor builders --------------------------------------------------------


def create_transition_tensor(layout: RockSampleLayout) -> np.ndarray:
    """T with shape (n_states, n_states, n_actions); θ-independent."""
    K = layout.K
    grid_size = layout.grid_size
    n_events = layout.n_events
    n_cells = layout.n_cells
    n_states = n_states_for(layout)
    n_actions = n_actions_for(K)

    rock_set = set(layout.rock_cells)
    cell_to_rock_idx = {c: i for i, c in enumerate(layout.rock_cells)}

    T = np.zeros((n_states, n_states, n_actions), dtype=np.float64)

    for state in range(n_states):
        pos, mask, _ = decode_state(state, K, n_events)
        for action in range(n_actions):
            role, payload = action_role(action, K)
            if role == "move":
                new_pos = move_cell(pos, payload, grid_size)
                new_mask = mask
                new_event = EVENT_OTHER
            elif role == "sense":
                k = payload
                new_pos = pos
                new_mask = mask
                new_event = 1 + k  # SENSE_k
            else:  # sample
                if pos in rock_set:
                    r = cell_to_rock_idx[pos]
                    if not ((mask >> r) & 1):
                        new_pos = pos
                        new_mask = mask | (1 << r)
                        new_event = K + 1  # SAMPLE
                    else:
                        new_pos = pos
                        new_mask = mask
                        new_event = EVENT_OTHER
                else:
                    new_pos = pos
                    new_mask = mask
                    new_event = EVENT_OTHER
            new_state = encode_state(new_pos, new_mask, new_event, K, n_events)
            T[new_state, state, action] = 1.0

    return T


def _alpha_d(d: np.ndarray, d0: float) -> np.ndarray:
    """Distance-attenuated base accuracy."""
    return 0.5 + 0.5 * (2.0 ** (-d / d0))


def _alpha_eff_noisy(alpha, sensor_noise_eta: float):
    """Noisy-regime effective accuracy: half-strength attenuation of the
    (alpha - 0.5) gap. At eta=1 the noisy sensor caps at (1+alpha)/2
    (= 0.75 at d=0), strictly below reliable's alpha — so θ_sensor stays
    identifiable at the rock cell across the full eta ∈ [0, 1] range.
    eta=0 is uninformative (0.5).
    """
    return 0.5 + 0.5 * (alpha - 0.5) * sensor_noise_eta


def create_observation_tensors(
    layout: RockSampleLayout,
    d0: float,
    sensor_noise_eta: float,
) -> list[np.ndarray]:
    """Return [B_position, B_rock_0, ..., B_rock_{K-1}].

    Shapes:
      B_position : (n_cells, n_states)        θ-independent
      B_rock_r   : (3, n_states, 2, 2)        θ-conditional on (θ_q_r, θ_s_r)
    """
    if not 0.0 < d0:
        raise ValueError(f"d0 must be > 0, got {d0}")
    if not 0.0 <= sensor_noise_eta <= 1.0:
        raise ValueError(
            f"sensor_noise_eta must be in [0, 1], got {sensor_noise_eta}"
        )

    K = layout.K
    grid_size = layout.grid_size
    n_events = layout.n_events
    n_cells = layout.n_cells
    n_states = n_states_for(layout)

    states = np.arange(n_states)
    block = (2 ** K) * n_events
    pos = states // block
    rest = states % block
    mask = rest // n_events
    event = rest % n_events

    # B_position: identity on cell.
    B_position = np.zeros((n_cells, n_states), dtype=np.float64)
    B_position[pos, states] = 1.0

    sx = pos % grid_size
    sy = pos // grid_size

    B_rocks: list[np.ndarray] = []
    sample_event = K + 1
    for r in range(K):
        B_r = np.zeros((N_ROCK_OBS, n_states, 2, 2), dtype=np.float64)
        rock_cell = layout.rock_cells[r]
        rx, ry = layout.cell_xy(rock_cell)

        d = np.maximum(np.abs(sx - rx), np.abs(sy - ry))
        alpha = _alpha_d(d, d0)  # (n_states,)

        mask_r = (mask >> r) & 1
        sense_event_r = 1 + r

        is_sense_r = (event == sense_event_r)
        is_sample_revealed = (
            (event == sample_event) & (pos == rock_cell) & (mask_r == 1)
        )
        is_other = ~is_sense_r & ~is_sample_revealed

        # No-info default for non-trigger states (constant across θ axes).
        B_r[ROCK_NO_INFO, is_other, :, :] = 1.0

        # Sense and sample emissions per (θ_quality_r, θ_sensor_r).
        for theta_q in (0, 1):
            for theta_s in (0, 1):
                if theta_s == 0:
                    alpha_eff = alpha
                else:
                    alpha_eff = _alpha_eff_noisy(alpha, sensor_noise_eta)

                # SENSE_r: emit θ_quality_r with prob alpha_eff, else flipped.
                B_r[theta_q, is_sense_r, theta_q, theta_s] = alpha_eff[is_sense_r]
                B_r[1 - theta_q, is_sense_r, theta_q, theta_s] = (
                    1.0 - alpha_eff[is_sense_r]
                )

                # SAMPLE-reveal: deterministic θ_quality_r.
                B_r[theta_q, is_sample_revealed, theta_q, theta_s] = 1.0

        B_rocks.append(B_r)

    return [B_position] + B_rocks


def create_goal(
    layout: RockSampleLayout,
    good_logit: float = 2.0,
    bad_logit: float = 4.0,
) -> np.ndarray:
    """(n_states, n_static) softmax preference over states for each θ.

    Higher preference for states whose sampled_mask covers more good rocks;
    lower preference for masks covering more bad rocks. Sensor θ does
    not affect the goal.

    The asymmetry (``bad_logit > good_logit``) makes the *expected* log C
    of "sample a rock under uniform quality belief" strictly negative,
    so AIF / VBP planners cannot pick SAMPLE purely for info gain when
    they don't know the rock's quality — they must reduce uncertainty
    via SENSE first. Symmetric (``bad_logit == good_logit``) ties the
    expected preference at zero and lets info gain dominate, which is
    the failure mode this default avoids.
    """
    K = layout.K
    n_events = layout.n_events
    n_states = n_states_for(layout)
    n_static = 4 ** K
    component_sizes = (2,) * (2 * K)

    states = np.arange(n_states)
    block = (2 ** K) * n_events
    rest = states % block
    mask = rest // n_events
    mask_bits = ((mask[:, None] >> np.arange(K)) & 1).astype(np.int32)  # (n_states, K)

    goal_logits = np.zeros((n_states, n_static), dtype=np.float64)
    for theta_flat in range(n_static):
        components = np.unravel_index(theta_flat, component_sizes)
        quality = np.array(components[:K], dtype=np.int32)
        n_good = (mask_bits * quality[None, :]).sum(axis=1)
        n_bad = (mask_bits * (1 - quality[None, :])).sum(axis=1)
        goal_logits[:, theta_flat] = good_logit * n_good - bad_logit * n_bad

    logits_centered = goal_logits - goal_logits.max(axis=0, keepdims=True)
    g = np.exp(logits_centered)
    return g / g.sum(axis=0, keepdims=True)


def informative_states_for_rock(
    layout: RockSampleLayout, r: int
) -> frozenset[int]:
    """States where rock_r's emission is θ-informative (not no_info)."""
    K = layout.K
    n_events = layout.n_events
    n_states = n_states_for(layout)
    rock_cell = layout.rock_cells[r]
    sense_event_r = 1 + r
    sample_event = K + 1

    out: list[int] = []
    for state in range(n_states):
        pos, mask, event = decode_state(state, K, n_events)
        mask_r = (mask >> r) & 1
        if event == sense_event_r:
            out.append(state)
        elif event == sample_event and pos == rock_cell and mask_r == 1:
            out.append(state)
    return frozenset(out)


def create_modality_registry(layout: RockSampleLayout) -> ModalityRegistry:
    modalities: list[ObservationModality] = [
        ObservationModality(
            name="position",
            cardinality=layout.n_cells,
            theta_dependence=ThetaDependence.INDEPENDENT,
        )
    ]
    for r in range(layout.K):
        modalities.append(
            ObservationModality(
                name=f"rock_{r}",
                cardinality=N_ROCK_OBS,
                theta_dependence=ThetaDependence.CONDITIONAL,
                informative_states=informative_states_for_rock(layout, r),
            )
        )
    return ModalityRegistry(modalities=tuple(modalities))


# --- Environment class ------------------------------------------------------


class RockSampleEnv:
    """Discrete RockSample BA-POMDP with factored θ (M4).

    Args:
        K: number of rocks (>= 1).
        grid_size: side length of the square grid (>= 2).
        d0: distance half-life for sensor accuracy α(d) = 0.5 + 0.5·2^(-d/d0).
            Defaults to grid_size / 2 if None.
        sensor_noise_eta: scales how far the noisy regime (θ_sensor_r=1)
            moves toward reliable. Noisy-sensor accuracy is
            alpha_eff = 0.5 + 0.5 * (α(d) - 0.5) * eta — a half-strength
            attenuation of the (α - 0.5) gap (see _alpha_eff_noisy).
            Range [0, 1]: eta=0 ⇒ uninformative (0.5); eta=1 ⇒
            half-strength reliable, capped at 0.75 at d=0. The half-
            strength factor keeps θ_sensor identifiable at the rock cell
            across the full eta range — reliable always emits bit_q with
            prob 1, noisy never above 0.75.
            Default 0.3 ("noisy is mostly uninformative").
        sample_reward: ±reward magnitude for a successful SAMPLE.
        layout_seed: seed for rock-cell placement.
        obs_seed: seed for the env's stochastic-emission RNG.
        max_steps: optional informational cap; the runner enforces actual
            termination via its `max_steps` argument.
    """

    def __init__(
        self,
        K: int = 3,
        grid_size: int = 4,
        d0: Optional[float] = None,
        sensor_noise_eta: float = 0.3,
        sample_reward: float = 1.0,
        good_logit: float = 2.0,
        bad_logit: float = 4.0,
        layout_seed: int = 0,
        obs_seed: int | None = 0,
        max_steps: int = 30,
    ) -> None:
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")
        if d0 is None:
            d0 = grid_size / 2.0

        self.K = K
        self.grid_size = grid_size
        self.d0 = float(d0)
        self.sensor_noise_eta = float(sensor_noise_eta)
        self.sample_reward = float(sample_reward)
        self.max_steps = int(max_steps)
        self._layout_seed = int(layout_seed)

        self._layout = sample_layout(grid_size, K, layout_seed)
        self.n_states = n_states_for(self._layout)
        self.n_actions = n_actions_for(K)
        # 2K binary components: K quality (per-episode) + K sensor (persistent).
        self.theta_spec = ThetaSpec(
            component_sizes=(2,) * (2 * K),
            episode_reset_components=tuple(range(K)),
        )
        self.n_static = self.theta_spec.n_static
        # T is θ-independent.
        self.t_factor_spec = FactorSpec(theta_components=())
        # B_position is θ-independent; each B_rock_r depends on its own
        # quality (component r) and sensor (component K + r) only.
        self.b_factor_specs: tuple[FactorSpec, ...] = (
            FactorSpec(theta_components=()),
        ) + tuple(
            FactorSpec(theta_components=(r, K + r)) for r in range(K)
        )

        self._T = create_transition_tensor(self._layout)
        self._B_list = create_observation_tensors(
            self._layout, d0=self.d0, sensor_noise_eta=self.sensor_noise_eta,
        )
        self._goal = create_goal(
            self._layout, good_logit=good_logit, bad_logit=bad_logit,
        )
        self._registry = create_modality_registry(self._layout)

        self._rock_set = set(self._layout.rock_cells)
        self._cell_to_rock_idx = {
            c: i for i, c in enumerate(self._layout.rock_cells)
        }

        self._rng = np.random.default_rng(obs_seed)
        self._theta: int = 0
        self._theta_components: tuple[int, ...] = (0,) * (2 * K)
        self._agent_state: int = 0
        self._step_count: int = 0

    # --- Environment protocol ----------------------------------------------

    @property
    def start_state_index(self) -> int:
        return encode_state(
            pos=self._layout.agent_start_cell,
            mask=0,
            event=EVENT_OTHER,
            K=self.K,
            n_events=self._layout.n_events,
        )

    @property
    def success_state_index(self) -> int:
        # No single canonical success state; expose the start state as a
        # placeholder for protocol conformance. Runners that depend on a
        # terminal success indicator (e.g. door_code's success-rate metric)
        # are not used directly with this env.
        return self.start_state_index

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
            int(c) for c in np.unravel_index(self._theta, self.theta_spec.component_sizes)
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

        # Compute reward before committing the new state.
        pos, mask, _ = decode_state(self._agent_state, self.K, self._layout.n_events)
        role, _ = action_role(action, self.K)
        reward = 0.0
        if role == "sample" and pos in self._rock_set:
            r = self._cell_to_rock_idx[pos]
            if not ((mask >> r) & 1):
                quality_r = self._theta_components[r]
                reward = self.sample_reward if quality_r == 1 else -self.sample_reward

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

    @property
    def layout(self) -> RockSampleLayout:
        return self._layout

    def state_to_cell(self, state: int) -> int:
        """Return the agent's grid cell for the given state index."""
        pos, _, _ = decode_state(state, self.K, self._layout.n_events)
        return pos

    # --- Observation generation -------------------------------------------

    def _make_obs(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        K = self.K
        n_events = self._layout.n_events
        pos, mask, event = decode_state(self._agent_state, K, n_events)
        n_cells = self._layout.n_cells

        position = np.zeros(n_cells)
        position[pos] = 1.0

        per_channel: dict[str, np.ndarray] = {"position": position}
        rock_obs_concat: list[np.ndarray] = []

        sample_event = K + 1
        for r in range(K):
            rock = np.zeros(N_ROCK_OBS)
            sense_event_r = 1 + r
            mask_r = (mask >> r) & 1
            theta_q = self._theta_components[r]
            theta_s = self._theta_components[K + r]

            if event == sense_event_r:
                d = self._layout.chebyshev_distance(pos, r)
                alpha = _alpha_d(np.array([d]), self.d0)[0]
                if theta_s == 0:
                    alpha_eff = alpha
                else:
                    alpha_eff = _alpha_eff_noisy(alpha, self.sensor_noise_eta)
                emit_truth = self._rng.random() < alpha_eff
                bit = theta_q if emit_truth else (1 - theta_q)
                rock[bit] = 1.0
            elif (
                event == sample_event
                and pos == self._layout.rock_cells[r]
                and mask_r == 1
            ):
                rock[theta_q] = 1.0
            else:
                rock[ROCK_NO_INFO] = 1.0

            per_channel[f"rock_{r}"] = rock
            rock_obs_concat.append(rock)

        obs = np.concatenate([position] + rock_obs_concat)
        return obs, per_channel
