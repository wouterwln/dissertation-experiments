"""Stochastic Maze environment for Active Inference experiments.

A flat grid-world (default 4x5 = 20 states, 4 NESW actions) with:
  - sink states        : absorbing cells the agent cannot leave
  - stochastic states  : "bridge" cells with randomized up/down movement
  - noisy observations  : cells where the location observation is corrupted
  - reward states       : fixed cells with +1 / -1 reward

This is a faithful Python/JAX port of the Julia `StochasticMaze`. The Julia code
is 1-indexed (x,y in 1..grid_size, linear state = x + (y-1)*grid_size_x); this
port is 0-indexed (x,y in 0..grid_size-1, linear state = x + y*grid_size_x) and
the default coordinates below are the 0-indexed translation of the Julia defaults
(the original 1-indexed values are noted in comments).

Unlike the (epistemic) T-Maze / reactivity maze, this environment has **no hidden
context theta**: it is a noisy, partially observed navigation task. In the
planner's generative model it is therefore used with ``n_theta = 1``.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional

import numpy as np
import jax.numpy as jnp
from jax import Array


class StochasticMazeAction(IntEnum):
    """Actions available in the stochastic maze (matches Julia 1..4 order)."""
    NORTH = 0  # +y
    EAST = 1   # +x
    SOUTH = 2  # -y
    WEST = 3   # -x


# ---------------------------------------------------------------------------
# Default 5x5 layout (0-indexed translation of the Julia experiment config:
# `create_stochastic_maze(5, 5, 4, start_state=11)`, biaslab/EFEasVFE).
#
# The grid is 5 wide x 5 tall (25 states). Columns x=0 and x=4 are entirely
# free of stochastic/sink cells: they are the SAFE corridors. The stochastic
# "bridge" row y=2 has clear ends (x=0 and x=4) where the start and goal sit.
# Nuijten-MP's ambiguity term is drawn to the low-observation-noise corridor,
# so the planner routes up x=0, across the (low-noise) top, and down x=4 to the
# goal, avoiding the stochastic interior entirely -> ~100% success.
# ---------------------------------------------------------------------------
DEFAULT_GRID_X = 5
DEFAULT_GRID_Y = 5
DEFAULT_N_ACTIONS = 4

# Julia sink_states = [(4,2),(4,4)]  (1-indexed)
DEFAULT_SINK_STATES: List[Tuple[int, int]] = [(3, 1), (3, 3)]
# Julia stochastic_states = [(2,3),(3,3),(4,3)]  (1-indexed)
DEFAULT_STOCHASTIC_STATES: List[Tuple[int, int]] = [(1, 2), (2, 2), (3, 2)]
# Julia noisy_observations (x,y,noise), 1-indexed -> here 0-indexed.
# (The Julia list contains a duplicate (2,1); kept here so the layout is identical.)
DEFAULT_NOISY_OBSERVATIONS: List[Tuple[int, int, float]] = [
    (0, 4, 0.1), (1, 4, 0.1), (2, 4, 0.4), (3, 4, 0.1),
    (1, 2, 0.4), (1, 3, 0.4), (2, 1, 0.4), (2, 2, 0.4),
    (3, 2, 0.2), (1, 1, 0.3), (2, 1, 0.4), (2, 3, 0.4),
]
# Julia reward_states = [(9,-1.0),(19,-1.0),(15,1.0)] (1-indexed linear) -> 0-indexed.
# On the 5x5 grid: state 8 = (3,1), state 18 = (3,3) (both coincide with the sinks),
# state 14 = (4,2) = the +1 goal.
DEFAULT_REWARD_STATES: List[Tuple[int, float]] = [(8, -1.0), (18, -1.0), (14, 1.0)]
DEFAULT_START_STATE = 10  # Julia start_state = 11 (1-indexed) = (0,2) 0-indexed


def coord_to_state(x: int, y: int, grid_size_x: int) -> int:
    """(x,y) -> linear state index (0-indexed, row-major in y)."""
    return x + y * grid_size_x


def state_to_xy(state: int, grid_size_x: int) -> Tuple[int, int]:
    """Linear state index -> (x,y) (0-indexed)."""
    return state % grid_size_x, state // grid_size_x


def generate_maze_tensors(
    grid_size_x: int = DEFAULT_GRID_X,
    grid_size_y: int = DEFAULT_GRID_Y,
    n_actions: int = DEFAULT_N_ACTIONS,
    sink_states: Optional[List[Tuple[int, int]]] = None,
    stochastic_states: Optional[List[Tuple[int, int]]] = None,
    noisy_observations: Optional[List[Tuple[int, int, float]]] = None,
    reward_states: Optional[List[Tuple[int, float]]] = None,
) -> Tuple[Array, Array, List[Tuple[int, float]]]:
    """Generate (observation_matrix A, transition_tensor B, reward_states).

    Returns:
        A: (n_states, n_states)        p(obs_state | state)         columns sum to 1
        B: (n_states, n_states, n_act) p(next_state | state, action) over axis 0
        reward_states: list of (state_idx, reward_value)
    """
    if sink_states is None:
        sink_states = DEFAULT_SINK_STATES
    if stochastic_states is None:
        stochastic_states = DEFAULT_STOCHASTIC_STATES
    if noisy_observations is None:
        noisy_observations = DEFAULT_NOISY_OBSERVATIONS
    if reward_states is None:
        reward_states = DEFAULT_REWARD_STATES

    gx, gy = grid_size_x, grid_size_y
    n_states = gx * gy
    sink_set = set(sink_states)
    stoch_set = set(stochastic_states)

    def cs(x, y):
        return coord_to_state(x, y, gx)

    # ---------------- Transition tensor B[next, current, action] ----------------
    B = np.zeros((n_states, n_states, n_actions), dtype=np.float64)
    for y in range(gy):
        for x in range(gx):
            s = cs(x, y)

            # Sink states: every action keeps the agent in place.
            if (x, y) in sink_set:
                B[s, s, :] = 1.0
                continue

            # Stochastic states: randomized up/down + drift, mixed with normal move.
            if (x, y) in stoch_set:
                for a in range(n_actions):
                    if y > 0:               # Julia y > 1
                        B[cs(x, y - 1), s, a] = 0.4
                    if y < gy - 1:          # Julia y < grid_size_y
                        B[cs(x, y + 1), s, a] = 0.4
                    B[cs(min(x + 1, gx - 1), y), s, a] = 0.2
                prob = 0.2
            else:
                prob = 1.0

            # Regular movement (also overwrites specific cells for stochastic states).
            # North (+y)
            if y < gy - 1:
                B[cs(x, y + 1), s, 0] = prob
            else:
                B[s, s, 0] = prob
            # East (+x)
            if x < gx - 1:
                B[cs(x + 1, y), s, 1] = prob
            else:
                B[s, s, 1] = prob
            # South (-y)
            if y > 0:
                B[cs(x, y - 1), s, 2] = prob
            else:
                B[s, s, 2] = prob
            # West (-x)
            if x > 0:
                B[cs(x - 1, y), s, 3] = prob
            else:
                B[s, s, 3] = prob

    # Normalize each (state, action) column.
    for s in range(n_states):
        for a in range(n_actions):
            total = B[:, s, a].sum()
            if total > 0:
                B[:, s, a] /= total

    # ---------------- Observation matrix A[obs, state] ----------------
    A = np.zeros((n_states, n_states), dtype=np.float64)
    for s in range(n_states):
        A[s, s] = 1.0

    for (x, y, noise_level) in noisy_observations:
        s = cs(x, y)
        A[s, s] = 1.0 - noise_level

        adjacent: List[int] = []
        if y > 0:
            adjacent.append(cs(x, y - 1))            # South
            if x > 0:
                adjacent.append(cs(x - 1, y - 1))    # Southwest
            if x < gx - 1:
                adjacent.append(cs(x + 1, y - 1))    # Southeast
        if y < gy - 1:
            adjacent.append(cs(x, y + 1))            # North
            if x > 0:
                adjacent.append(cs(x - 1, y + 1))    # Northwest
            if x < gx - 1:
                adjacent.append(cs(x + 1, y + 1))    # Northeast
        if x > 0:
            adjacent.append(cs(x - 1, y))            # West
        if x < gx - 1:
            adjacent.append(cs(x + 1, y))            # East

        if adjacent:
            prob_per = noise_level / len(adjacent)
            for adj in adjacent:
                A[adj, s] = prob_per

    # Normalize each column.
    for s in range(n_states):
        total = A[:, s].sum()
        if total > 0:
            A[:, s] /= total

    return jnp.asarray(A, dtype=jnp.float32), jnp.asarray(B, dtype=jnp.float32), list(reward_states)


@dataclass
class StochasticMaze:
    """Stochastic maze environment with probabilistic transitions/observations."""
    transition_tensor: Array            # (n_states, n_states, n_actions)
    observation_matrix: Array           # (n_states, n_states)
    reward_states: List[Tuple[int, float]]
    agent_state: int
    grid_size_x: int
    grid_size_y: int
    sink_states: List[Tuple[int, int]] = field(default_factory=list)
    stochastic_states: List[Tuple[int, int]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        grid_size_x: int = DEFAULT_GRID_X,
        grid_size_y: int = DEFAULT_GRID_Y,
        n_actions: int = DEFAULT_N_ACTIONS,
        sink_states: Optional[List[Tuple[int, int]]] = None,
        stochastic_states: Optional[List[Tuple[int, int]]] = None,
        noisy_observations: Optional[List[Tuple[int, int, float]]] = None,
        reward_states: Optional[List[Tuple[int, float]]] = None,
        start_state: int = DEFAULT_START_STATE,
    ) -> "StochasticMaze":
        sink_states = DEFAULT_SINK_STATES if sink_states is None else sink_states
        stochastic_states = DEFAULT_STOCHASTIC_STATES if stochastic_states is None else stochastic_states
        A, B, rewards = generate_maze_tensors(
            grid_size_x, grid_size_y, n_actions,
            sink_states=sink_states,
            stochastic_states=stochastic_states,
            noisy_observations=noisy_observations,
            reward_states=reward_states,
        )
        n_states = grid_size_x * grid_size_y
        if not 0 <= start_state < n_states:
            raise ValueError(f"start_state must be 0..{n_states - 1}, got {start_state}")
        return cls(
            transition_tensor=B,
            observation_matrix=A,
            reward_states=rewards,
            agent_state=start_state,
            grid_size_x=grid_size_x,
            grid_size_y=grid_size_y,
            sink_states=list(sink_states),
            stochastic_states=list(stochastic_states),
        )

    def reset(self, start_state: int = DEFAULT_START_STATE) -> "StochasticMaze":
        n_states = self.grid_size_x * self.grid_size_y
        if not 0 <= start_state < n_states:
            raise ValueError(f"start_state must be 0..{n_states - 1}, got {start_state}")
        self.agent_state = start_state
        return self

    @property
    def goal_state(self) -> int:
        """The (first) positive-reward state."""
        for s, r in self.reward_states:
            if r > 0:
                return s
        return -1

    def step(self, rng: np.random.Generator, action: int) -> Tuple[Array, int, float, bool]:
        """Take one step.

        Returns:
            location_obs: (n_states,) one-hot of the sampled (noisy) observation
            observation : int, the sampled observation state index
            reward      : float reward of the new state
            done        : True once the agent reaches any reward state
        """
        T = np.asarray(self.transition_tensor[:, self.agent_state, action])
        next_state = int(rng.choice(len(T), p=T / T.sum()))
        self.agent_state = next_state

        A = np.asarray(self.observation_matrix[:, next_state])
        observation = int(rng.choice(len(A), p=A / A.sum()))

        reward = 0.0
        for s, r in self.reward_states:
            if next_state == s:
                reward = float(r)
                break

        n_states = self.grid_size_x * self.grid_size_y
        location_obs = jnp.zeros(n_states).at[observation].set(1.0)
        done = reward != 0.0
        return location_obs, observation, reward, done
