"""Spatial door-code POMDP — Farama-MiniGrid-backed sibling of door_code.py.

This is door_code (M1) projected onto MiniGrid: the same persistent
permutation theta in S_K, the same partition-predicate probes, and the same
multi-step lock semantics, with two additions and nothing else:

- **Spatial navigation.** The agent occupies a (col, row) cell on a
  grid_size x grid_size MiniGrid layout, has a heading, and uses the
  4-action subset {turn-left, turn-right, forward, toggle}. Probes are
  colored Floor tiles (yellow) walked onto; lock doors are colored Door
  objects toggled when faced; success requires walking to a Goal() cell
  *behind* the partition wall *after* opening all lock doors in order.
- **Partial observability.** The agent sees a 3x3 partial FOV, hashed to
  a single integer view-index. Multiple cells map to the same view, so
  genuine localization is required. The FOV is theta-INDEPENDENT: doors
  are always rendered as locked at view-enumeration time, so
  lock-progress never leaks through the position channel — progress is
  inferred only via the agent's own action history and the outcome
  channel.

Everything else (theta in S_K, partition table, probe noise, lock_length
semantics) is reused from door_code.py.

Layout per outer-loop seed: door rows, probe rows, start row, goal cell,
door colors, and the probe->bit assignment are sampled from
`layout_seed`. Layout stays fixed within a block (n inner episodes); only
theta varies across resets, matching the m x n protocol.

Slip via T-as-truth: the precomputed transition tensor is the marginal of
the StochasticActionWrapper-style slip-then-det process,
T(x' | x, u, theta) = (1-p)*delta(T_det(x, m(u))) + (p/6)*sum_a delta(T_det(x, a)),
and step() samples directly from T[:, s, theta, u]. Distributionally
identical to slip-bool + slip-action + det-step, but T is the single
source of truth for env stochasticity.

State space (n_states = n_reachable_cells * 4 * (L+1) + 2):
    (reach_cell_idx * 4 + dir) * (L+1) + stage  : agent at cell, dir, stage k
    n_reachable_cells * 4 * (L+1)               : SUCCESS (absorbing)
    n_reachable_cells * 4 * (L+1) + 1           : FAILURE (absorbing)

reach_cell_idx is a stable index over interior cells that are not walls.
dir in {0=right, 1=down, 2=left, 3=up}. stage in [0, L+1): stage k means
"k lock doors opened correctly"; stage L means "all doors open, walk to
goal."

Action space (n_actions = 4):
    0 = LEFT     (turn left,  MiniGrid action 0)
    1 = RIGHT    (turn right, MiniGrid action 1)
    2 = FORWARD  (step forward, MiniGrid action 2)
    3 = TOGGLE   (toggle, MiniGrid action 5; opens an adjacent lock door)

TOGGLE on the lock door at stage k under theta:
  - if door_idx == lock[k] -> (same cell, same dir, stage + 1).
  - else -> FAILURE.
TOGGLE at stage = L: no-op.
TOGGLE adjacent to a non-door cell or facing a wall: no-op.
FORWARD onto goal_cell at stage = L -> SUCCESS.
FORWARD onto a closed lock door: blocked.

Modalities:
    position : theta-INDEPENDENT one-hot view-index built by enumerating
               all (cell, dir) -> 3x3 FOV bytes with all doors locked,
               then assigning a unique integer per distinct view.
               Cardinality n_views.
    probe    : theta-CONDITIONAL bit at probe-cell states (any stage,
               any heading); cardinality 3 (bit_0, bit_1, neutral).
    outcome  : theta-INDEPENDENT terminal indicator; cardinality 3.

Tensor shapes:
    T          : (n_states, n_states, K!, 4)
    B_position : (n_views, n_states)
    B_probe    : (3, n_states, K!)
    B_outcome  : (3, n_states)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Floor, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

from learningpomdp.environments.base import StepResult
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.environments.door_code import (
    N_OUTCOME_OBS,
    N_PROBE_OBS,
    OUTCOME_FAILURE,
    OUTCOME_NEUTRAL,
    OUTCOME_SUCCESS,
    PROBE_BIT_0,
    PROBE_BIT_1,
    PROBE_NEUTRAL,
    build_partition_table,
    default_lock_doors,
    default_partitions,
    enumerate_permutations,
    n_static_for_K,
    partitions_uniquely_identify,
)
from learningpomdp.observations import (
    ModalityRegistry,
    ObservationModality,
    ThetaDependence,
)


# --- Action constants ------------------------------------------------------

# Planner-facing 4-action subset. The integers are the indices our planners
# see (last axis of T); the values are the MiniGrid Actions enum integers.
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_TOGGLE = 3
N_ACTIONS = 4

# MiniGrid action enum — 7 underlying actions, 6 of which the slip
# distribution can produce (done is excluded because StochasticActionWrapper
# samples integers in [0, 6); we mirror that).
_MG_LEFT = 0
_MG_RIGHT = 1
_MG_FORWARD = 2
_MG_PICKUP = 3
_MG_DROP = 4
_MG_TOGGLE = 5
_N_SLIP_ACTIONS = 6  # uniform over [0, 6) for slipped actions

# Map planner action -> MiniGrid action.
_PLANNER_TO_MG: dict[int, int] = {
    ACTION_LEFT: _MG_LEFT,
    ACTION_RIGHT: _MG_RIGHT,
    ACTION_FORWARD: _MG_FORWARD,
    ACTION_TOGGLE: _MG_TOGGLE,
}

# 4 cardinal directions, MiniGrid convention.
N_DIRS = 4
_DIR_DELTAS: dict[int, tuple[int, int]] = {
    0: (1, 0),    # right (+x)
    1: (0, 1),    # down  (+y)
    2: (-1, 0),   # left  (-x)
    3: (0, -1),   # up    (-y)
}

# Probe Floor color (must be in COLOR_NAMES; chosen distinct from the
# K door colors).
_PROBE_COLOR = "yellow"


# --- Layout ---------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Cell-level structure of a MiniGrid door-code board.

    Coordinates use the MiniGrid (col, row) convention. Flat cell indices
    are `row * grid_size + col`. Wall cells are impassable; doors, probes
    and the goal must sit on cells that are not in `wall_cells`.

    `reachable_cells` lists the interior cells that are NOT walls — i.e.
    exactly the cells the agent can ever occupy. State indexing is over
    reach_cell_idx in [0, len(reachable_cells)), not flat cell indices,
    so wall cells don't inflate n_states.
    """
    grid_size: int
    start_cell: int
    start_dir: int
    door_cells: tuple[int, ...]   # length K
    door_colors: tuple[str, ...]  # length K, parallel to door_cells
    probe_cells: tuple[int, ...]  # length M
    probe_to_bit: tuple[int, ...] # length M; probe i emits bit_{probe_to_bit[i]}
    goal_cell: int
    wall_cells: frozenset[int]
    reachable_cells: tuple[int, ...]
    cell_to_state_idx: dict[int, int]   # flat_cell -> reach_cell_idx, walls excluded


def _flat(col: int, row: int, grid_size: int) -> int:
    return row * grid_size + col


def _coords(cell: int, grid_size: int) -> tuple[int, int]:
    """Return (col, row) for a flat cell index."""
    return cell % grid_size, cell // grid_size


def sample_layout(grid_size: int, K: int, M: int, seed: int) -> Layout:
    """Sample a layout from `seed`. See module docstring for what varies.

    Layout geometry:
    - Outer wall on all four edges; partition wall on `mid_col = grid_size // 2`
      with K colored locked doors at K randomly-chosen interior rows.
    - Probes (colored Floor tiles, color=yellow) at M randomly-chosen
      interior rows on column 1 (left half).
    - Start cell on column 1, distinct from probe rows (one row left over
      after picking M probe rows from interior rows — that's the start).
      Start direction fixed to `right` (facing the partition).
    - Goal at a uniformly chosen interior cell in the right half (cols
      mid_col+1 .. grid_size-2, rows 1 .. grid_size-2).
    - K door colors sampled without replacement from COLOR_NAMES \\ {yellow}.
    - Probe-to-bit assignment is a uniform permutation of range(M); probe i
      emits bit_{probe_to_bit[i]} of theta's lex-rank. For M=1 this is a
      no-op.

    Geometry constraint: grid_size >= max(5, K + 2, M + 3). The `>= 5` guard
    ensures the right half has at least one interior column for the goal.
    """
    if K < 1 or K > len(COLOR_NAMES) - 1:
        raise ValueError(
            f"K must be in [1, {len(COLOR_NAMES) - 1}], got {K}; need K "
            f"distinct door colors plus a separate probe color (yellow)."
        )
    min_grid_size = max(5, K + 2, M + 3)
    if grid_size < min_grid_size:
        raise ValueError(
            f"grid_size={grid_size} too small for K={K}, M={M}: need "
            f"grid_size >= max(5, K + 2, M + 3) = {min_grid_size} so that "
            f"K door rows, M probe rows + a start row, and a right-half "
            f"goal cell all fit."
        )
    rng = np.random.default_rng(seed)
    interior = list(range(1, grid_size - 1))
    mid_col = grid_size // 2

    # Door rows on mid_col.
    door_rows = rng.choice(interior, size=K, replace=False)
    door_cells = tuple(_flat(mid_col, int(r), grid_size) for r in door_rows)

    # Probe rows + start row on col=1, all distinct.
    probe_and_start_rows = rng.choice(interior, size=M + 1, replace=False)
    probe_rows = probe_and_start_rows[:M]
    start_row = int(probe_and_start_rows[M])
    probe_cells = tuple(_flat(1, int(r), grid_size) for r in probe_rows)
    start_cell = _flat(1, start_row, grid_size)

    # Goal cell anywhere in the right half interior.
    right_cols = list(range(mid_col + 1, grid_size - 1))
    goal_col = int(rng.choice(right_cols))
    goal_row = int(rng.choice(interior))
    goal_cell = _flat(goal_col, goal_row, grid_size)

    # Door colors.
    color_pool = [c for c in COLOR_NAMES if c != _PROBE_COLOR]
    door_color_idx = rng.choice(len(color_pool), size=K, replace=False)
    door_colors = tuple(color_pool[int(i)] for i in door_color_idx)

    # Probe -> bit assignment.
    probe_to_bit = tuple(int(b) for b in rng.permutation(M))

    # Wall cells: outer ring + partition column except where doors are.
    walls: set[int] = set()
    for c in range(grid_size):
        walls.add(_flat(c, 0, grid_size))
        walls.add(_flat(c, grid_size - 1, grid_size))
    for r in range(grid_size):
        walls.add(_flat(0, r, grid_size))
        walls.add(_flat(grid_size - 1, r, grid_size))
    for r in range(grid_size):
        walls.add(_flat(mid_col, r, grid_size))
    walls -= set(door_cells)

    # Reachable cells: every cell not in walls. Stable order = cell-index
    # order so that reach_cell_idx is deterministic given the layout.
    reachable_cells = tuple(
        c for c in range(grid_size * grid_size) if c not in walls
    )
    cell_to_state_idx = {c: i for i, c in enumerate(reachable_cells)}

    # Sanity: start, probes, goal, doors all on reachable cells.
    for label, cell in [("start", start_cell), ("goal", goal_cell)]:
        if cell not in cell_to_state_idx:
            raise ValueError(f"{label} cell {cell} fell on a wall (seed={seed}).")
    for cell in probe_cells:
        if cell not in cell_to_state_idx:
            raise ValueError(f"probe cell {cell} fell on a wall (seed={seed}).")
    for cell in door_cells:
        if cell not in cell_to_state_idx:
            raise ValueError(f"door cell {cell} fell on a wall (seed={seed}).")

    return Layout(
        grid_size=grid_size,
        start_cell=start_cell,
        start_dir=0,  # facing right (toward the partition)
        door_cells=door_cells,
        door_colors=door_colors,
        probe_cells=probe_cells,
        probe_to_bit=probe_to_bit,
        goal_cell=goal_cell,
        wall_cells=frozenset(walls),
        reachable_cells=reachable_cells,
        cell_to_state_idx=cell_to_state_idx,
    )


# --- State indexing -------------------------------------------------------


def _state_index(reach_cell_idx: int, dir: int, stage: int, L: int) -> int:
    """Encode (reach_cell_idx, dir, stage) -> non-terminal state index."""
    return (reach_cell_idx * N_DIRS + dir) * (L + 1) + stage


def _decode_state(s: int, L: int) -> tuple[int, int, int]:
    """Inverse of _state_index for non-terminal states."""
    stage = s % (L + 1)
    rest = s // (L + 1)
    dir = rest % N_DIRS
    reach_cell_idx = rest // N_DIRS
    return reach_cell_idx, dir, stage


def _success_index(n_reachable: int, L: int) -> int:
    return n_reachable * N_DIRS * (L + 1)


def _failure_index(n_reachable: int, L: int) -> int:
    return n_reachable * N_DIRS * (L + 1) + 1


# --- Inner Farama env -----------------------------------------------------


class _DoorCodeFaramaEnv(MiniGridEnv):
    """Inner Farama subclass that owns the grid and rendering machinery.

    `MiniGridDoorCodeEnv` (the protocol-satisfying outer class) composes
    one of these. We never call its `step()` — agent dynamics are entirely
    encoded in our T tensor — but we use it for (a) generating the 3x3 FOV
    encoding for view enumeration and (b) `render()` for debug videos.
    """

    def __init__(
        self,
        layout: Layout,
        agent_view_size: int = 3,
        max_steps: int = 10**9,
        render_mode: str | None = None,
    ) -> None:
        self._layout = layout
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=layout.grid_size,
            see_through_walls=False,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
        )

    @staticmethod
    def _gen_mission() -> str:
        return "open the door code, then reach the goal"

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        layout = self._layout
        # Partition wall (interior wall cells only — outer ring already set).
        for cell in layout.wall_cells:
            col, row = _coords(cell, layout.grid_size)
            if 0 < col < width - 1 and 0 < row < height - 1:
                self.grid.set(col, row, Wall())

        # Doors: placed locked. Theta-correctness check is in our T; we
        # never call Door.toggle (which would require a Key). is_locked=True
        # ensures the FOV state byte is 2 (locked).
        for door_idx, cell in enumerate(layout.door_cells):
            col, row = _coords(cell, layout.grid_size)
            self.grid.set(col, row, Door(layout.door_colors[door_idx], is_locked=True))

        # Probes: colored Floor tiles, walkable.
        for cell in layout.probe_cells:
            col, row = _coords(cell, layout.grid_size)
            self.grid.set(col, row, Floor(_PROBE_COLOR))

        # Goal cell: a Goal() object behind the partition. Termination is
        # handled by our state machine; Goal here is purely visual.
        goal_col, goal_row = _coords(layout.goal_cell, layout.grid_size)
        self.put_obj(Goal(), goal_col, goal_row)

        start_col, start_row = _coords(layout.start_cell, layout.grid_size)
        self.agent_pos = (start_col, start_row)
        self.agent_dir = layout.start_dir
        self.mission = self._gen_mission()

    # --- Helpers used by the outer env to mutate door state and render ---

    def set_door_open(self, door_idx: int, is_open: bool) -> None:
        layout = self._layout
        cell = layout.door_cells[door_idx]
        col, row = _coords(cell, layout.grid_size)
        obj = self.grid.get(col, row)
        assert isinstance(obj, Door), f"expected Door at {(col, row)}, got {obj}"
        obj.is_open = is_open
        obj.is_locked = not is_open

    def reset_doors(self) -> None:
        for i in range(len(self._layout.door_cells)):
            self.set_door_open(i, False)

    def set_pose(self, cell: int, dir: int) -> None:
        col, row = _coords(cell, self._layout.grid_size)
        self.agent_pos = (col, row)
        self.agent_dir = dir

    def encoded_view(self) -> np.ndarray:
        """Return the agent's current 3x3x3 partial-FOV encoding."""
        grid, vis_mask = self.gen_obs_grid(self.agent_view_size)
        return grid.encode(vis_mask)


# --- Tensor builders ------------------------------------------------------


def _step_det_under_action(
    layout: Layout,
    reach_cell_idx: int,
    dir: int,
    stage: int,
    L: int,
    mg_action: int,
    door_to_idx: dict[int, int],
    lock: tuple[int, ...],
) -> int:
    """Deterministic transition under MiniGrid action `mg_action` from
    state (reach_cell_idx, dir, stage) under permutation `lock = perm[:L]`.
    Returns the new state index (terminals are SUCCESS / FAILURE indices)."""
    n_reachable = len(layout.reachable_cells)
    SUCCESS = _success_index(n_reachable, L)
    FAILURE = _failure_index(n_reachable, L)
    cell = layout.reachable_cells[reach_cell_idx]

    if mg_action == _MG_LEFT:
        new_dir = (dir - 1) % N_DIRS
        return _state_index(reach_cell_idx, new_dir, stage, L)
    if mg_action == _MG_RIGHT:
        new_dir = (dir + 1) % N_DIRS
        return _state_index(reach_cell_idx, new_dir, stage, L)
    if mg_action == _MG_FORWARD:
        col, row = _coords(cell, layout.grid_size)
        dx, dy = _DIR_DELTAS[dir]
        new_col, new_row = col + dx, row + dy
        if not (0 <= new_col < layout.grid_size and 0 <= new_row < layout.grid_size):
            return _state_index(reach_cell_idx, dir, stage, L)
        new_cell = _flat(new_col, new_row, layout.grid_size)
        # Walls (outer + partition gaps): blocked. Walls are not in
        # cell_to_state_idx, so this is the canonical check.
        if new_cell not in layout.cell_to_state_idx:
            return _state_index(reach_cell_idx, dir, stage, L)
        # Closed lock door: blocked. A door at index j is open iff
        # j in lock[:stage].
        if new_cell in door_to_idx:
            j = door_to_idx[new_cell]
            if j not in lock[:stage]:
                return _state_index(reach_cell_idx, dir, stage, L)
        # Goal cell + stage = L: SUCCESS.
        if new_cell == layout.goal_cell and stage == L:
            return SUCCESS
        # Otherwise: walk in.
        new_reach = layout.cell_to_state_idx[new_cell]
        return _state_index(new_reach, dir, stage, L)
    if mg_action == _MG_TOGGLE:
        # No commit possible at stage = L (all doors already open).
        if stage == L:
            return _state_index(reach_cell_idx, dir, stage, L)
        col, row = _coords(cell, layout.grid_size)
        dx, dy = _DIR_DELTAS[dir]
        front_col, front_row = col + dx, row + dy
        if not (0 <= front_col < layout.grid_size and 0 <= front_row < layout.grid_size):
            return _state_index(reach_cell_idx, dir, stage, L)
        front_cell = _flat(front_col, front_row, layout.grid_size)
        if front_cell not in door_to_idx:
            return _state_index(reach_cell_idx, dir, stage, L)  # no-op
        toggled = door_to_idx[front_cell]
        if toggled == lock[stage]:
            return _state_index(reach_cell_idx, dir, stage + 1, L)
        return FAILURE
    if mg_action == _MG_PICKUP or mg_action == _MG_DROP:
        return _state_index(reach_cell_idx, dir, stage, L)  # no-op
    raise ValueError(f"unhandled minigrid action: {mg_action}")


def create_transition_tensor(
    layout: Layout,
    K: int,
    L: int,
    slip_prob: float,
    lock_doors_fn: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
) -> np.ndarray:
    """T with shape (n_states, n_states, K!, 4).

    For each non-terminal (reach_cell_idx, dir, stage) and planner action u:
      T(x'|x, u, theta) = (1 - p) * delta(x', T_det(x, m(u), theta))
                        + (p / 6)  * sum_{a in [0, 6)} delta(x', T_det(x, a, theta))
    where m(u) maps planner -> MiniGrid action. T_det depends on theta only
    through `lock = perm_theta[:L]` (which doors are open at each stage).
    """
    if not 0.0 <= slip_prob <= 1.0:
        raise ValueError(f"slip_prob must be in [0, 1], got {slip_prob}.")
    if lock_doors_fn is None:
        lock_doors_fn = lambda perm: default_lock_doors(perm, L)

    n_reachable = len(layout.reachable_cells)
    n_states = n_reachable * N_DIRS * (L + 1) + 2
    n_static = n_static_for_K(K)
    SUCCESS = _success_index(n_reachable, L)
    FAILURE = _failure_index(n_reachable, L)
    door_to_idx = {cell: i for i, cell in enumerate(layout.door_cells)}
    perms = enumerate_permutations(K)

    T = np.zeros((n_states, n_states, n_static, N_ACTIONS), dtype=np.float64)

    keep_w = 1.0 - slip_prob
    slip_w = slip_prob / _N_SLIP_ACTIONS

    for theta_idx, perm in enumerate(perms):
        lock = lock_doors_fn(perm)
        if len(lock) != L:
            raise ValueError(
                f"lock_doors_fn returned length {len(lock)} != L={L}."
            )

        for reach_cell_idx in range(n_reachable):
            for dir in range(N_DIRS):
                for stage in range(L + 1):
                    s_old = _state_index(reach_cell_idx, dir, stage, L)

                    # Deterministic next-state for each underlying MG action.
                    det_next = [
                        _step_det_under_action(
                            layout, reach_cell_idx, dir, stage, L, a,
                            door_to_idx, lock,
                        )
                        for a in range(_N_SLIP_ACTIONS)
                    ]

                    # Slip mixture row, common to all planner actions.
                    slip_row = np.zeros(n_states, dtype=np.float64)
                    for s_new in det_next:
                        slip_row[s_new] += slip_w

                    for u in range(N_ACTIONS):
                        intended = _PLANNER_TO_MG[u]
                        s_intended = det_next[intended]
                        T[:, s_old, theta_idx, u] = slip_row
                        T[s_intended, s_old, theta_idx, u] += keep_w

    # Terminal absorption.
    for theta_idx in range(n_static):
        for a in range(N_ACTIONS):
            T[SUCCESS, SUCCESS, theta_idx, a] = 1.0
            T[FAILURE, FAILURE, theta_idx, a] = 1.0

    return T


def _enumerate_locked_views(
    farama: _DoorCodeFaramaEnv,
    layout: Layout,
) -> tuple[np.ndarray, int]:
    """Enumerate (reach_cell_idx, dir) -> distinct 3x3 FOV bytes with all
    doors *locked*. Returns (view_idx_table, n_views) where view_idx_table
    has shape (n_reachable, N_DIRS). Terminal / non-existent (cell, dir)
    pairs are not included; the outer caller handles terminals separately
    via a sentinel view.
    """
    farama.reset_doors()
    n_reachable = len(layout.reachable_cells)
    view_to_idx: dict[bytes, int] = {}

    def get_view_idx(view_bytes: bytes) -> int:
        if view_bytes not in view_to_idx:
            view_to_idx[view_bytes] = len(view_to_idx)
        return view_to_idx[view_bytes]

    view_idx_table = np.zeros((n_reachable, N_DIRS), dtype=np.int64)
    for reach_cell_idx, cell in enumerate(layout.reachable_cells):
        for dir in range(N_DIRS):
            farama.set_pose(cell, dir)
            view_bytes = farama.encoded_view().tobytes()
            view_idx_table[reach_cell_idx, dir] = get_view_idx(view_bytes)

    return view_idx_table, len(view_to_idx)


def create_observation_tensors(
    farama: _DoorCodeFaramaEnv,
    layout: Layout,
    K: int,
    L: int,
    partition_table: np.ndarray,
    probe_noise: float,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Return ([B_position, B_probe, B_outcome], state_to_view, n_views).

    Shapes:
      B_position : (n_views, n_states)             theta-INDEPENDENT
      B_probe    : (3, n_states, K!)               theta-CONDITIONAL at probes
      B_outcome  : (3, n_states)                   theta-INDEPENDENT

    `state_to_view` is a length-`n_states` array mapping each non-terminal
    state to its view-index; terminals map to a sentinel view (the last
    one). The env emits position observations as one-hot at this index.
    """
    if not 0.0 <= probe_noise <= 0.5:
        raise ValueError(
            f"probe_noise must be in [0, 0.5]; got {probe_noise}."
        )
    M, n_static = partition_table.shape
    n_reachable = len(layout.reachable_cells)
    n_states = n_reachable * N_DIRS * (L + 1) + 2
    SUCCESS = _success_index(n_reachable, L)
    FAILURE = _failure_index(n_reachable, L)
    if len(layout.probe_cells) != M:
        raise ValueError(
            f"layout has {len(layout.probe_cells)} probe cells but "
            f"partition_table has M={M} predicates."
        )

    cell_dir_view, n_locked_views = _enumerate_locked_views(farama, layout)
    sentinel_idx = n_locked_views  # one beyond the locked-view range
    n_views = n_locked_views + 1

    # state_to_view: length n_states. Non-terminal states use cell_dir_view
    # (replicated across stages); terminals use the sentinel.
    state_to_view = np.full(n_states, sentinel_idx, dtype=np.int64)
    for reach_cell_idx in range(n_reachable):
        for dir in range(N_DIRS):
            v = int(cell_dir_view[reach_cell_idx, dir])
            for stage in range(L + 1):
                state_to_view[_state_index(reach_cell_idx, dir, stage, L)] = v

    # B_position: deterministic one-hot at state_to_view[s], theta-indep.
    B_position = np.zeros((n_views, n_states), dtype=np.float64)
    for s in range(n_states):
        B_position[state_to_view[s], s] = 1.0

    # Probe-cell state indices (any heading, any stage).
    probe_reach_indices = [
        layout.cell_to_state_idx[cell] for cell in layout.probe_cells
    ]
    probe_state_indices: list[list[int]] = [
        [
            _state_index(reach_idx, d, s, L)
            for d in range(N_DIRS)
            for s in range(L + 1)
        ]
        for reach_idx in probe_reach_indices
    ]
    probe_state_set: set[int] = {
        s for states in probe_state_indices for s in states
    }

    B_probe = np.zeros((N_PROBE_OBS, n_states, n_static), dtype=np.float64)
    p_correct = 1.0 - probe_noise
    p_flipped = probe_noise
    for theta_idx in range(n_static):
        for i in range(M):
            true_bit = int(partition_table[i, theta_idx])
            for s in probe_state_indices[i]:
                B_probe[true_bit, s, theta_idx] = p_correct
                B_probe[1 - true_bit, s, theta_idx] = p_flipped
        for s in range(n_states):
            if s not in probe_state_set:
                B_probe[PROBE_NEUTRAL, s, theta_idx] = 1.0

    B_outcome = np.zeros((N_OUTCOME_OBS, n_states), dtype=np.float64)
    for s in range(n_states):
        if s == SUCCESS:
            B_outcome[OUTCOME_SUCCESS, s] = 1.0
        elif s == FAILURE:
            B_outcome[OUTCOME_FAILURE, s] = 1.0
        else:
            B_outcome[OUTCOME_NEUTRAL, s] = 1.0

    return [B_position, B_probe, B_outcome], state_to_view, n_views


def create_goal(
    layout: Layout,
    K: int,
    L: int,
    success_logit: float = 4.0,
    failure_logit: float = -4.0,
    non_terminal_logit: float = 0.0,
) -> np.ndarray:
    """(n_states, n_static) softmax preference: SUCCESS preferred, FAILURE
    not. Plain logits — no auto-scaling needed since wall cells are no
    longer in the state index space."""
    n_reachable = len(layout.reachable_cells)
    n_states = n_reachable * N_DIRS * (L + 1) + 2
    n_static = n_static_for_K(K)
    SUCCESS = _success_index(n_reachable, L)
    FAILURE = _failure_index(n_reachable, L)

    logits = np.full((n_states, n_static), non_terminal_logit, dtype=np.float64)
    logits[SUCCESS, :] = success_logit
    logits[FAILURE, :] = failure_logit
    exp_logits = np.exp(logits - logits.max(axis=0, keepdims=True))
    return exp_logits / exp_logits.sum(axis=0, keepdims=True)


def create_modality_registry(
    layout: Layout,
    K: int,
    L: int,
    n_views: int,
) -> ModalityRegistry:
    probe_reach_indices = [
        layout.cell_to_state_idx[cell] for cell in layout.probe_cells
    ]
    probe_states = frozenset(
        _state_index(reach_idx, d, s, L)
        for reach_idx in probe_reach_indices
        for d in range(N_DIRS)
        for s in range(L + 1)
    )
    return ModalityRegistry(modalities=(
        ObservationModality(
            name="position",
            cardinality=n_views,
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


# --- Outer env class ------------------------------------------------------


class MiniGridDoorCodeEnv:
    """Spatial door-code POMDP backed by Farama MiniGrid. See module
    docstring for full semantics.

    Args:
        K: number of doors / symbols.
        lock_length: number of doors to open in sequence (1 <= L <= K).
        M: number of probe stations. Defaults to len(default_partitions(K)).
        probe_noise: per-probe bit-flip probability in [0, 0.5].
        slip_prob: per-step slip probability in [0, 1]; with this prob
            the chosen action is replaced by a uniform draw over the 6
            underlying MiniGrid actions {left, right, forward, pickup,
            drop, toggle}. Pickup/drop are no-ops in this env. T is the
            marginal of this slip-then-det process, and step() samples
            from T directly.
        grid_size: side length of the square grid.
            >= max(5, K + 2, M + 3).
        agent_view_size: side length of the partial FOV; must be odd
            and >= 3 (per Farama's `ViewSizeWrapper` constraint).
        layout_seed: seed for sampling the layout (door rows, probe rows,
            start, goal, door colors, probe->bit permutation). Defaults
            to obs_seed so callers that vary obs_seed across blocks get
            layout-per-seed for free.
        partitions: optional list of M predicates (defaults to door_code's
            default_partitions(K)).
        lock_doors_fn: optional callable (perm) -> tuple of length L.
        goal: optional override for the (n_states, n_static) preference.
        obs_seed: seed for the env's stochastic-emission RNG (probe
            flips, T-row sampling, and reset theta sampling). Also the
            default for layout_seed.
        render_mode: optional Farama render mode ("human" / "rgb_array")
            for ad-hoc debugging via the underlying MiniGridEnv.
    """

    def __init__(
        self,
        K: int = 2,
        lock_length: int = 1,
        M: int | None = None,
        probe_noise: float = 0.0,
        slip_prob: float = 0.0,
        grid_size: int = 5,
        agent_view_size: int = 3,
        layout_seed: int | None = None,
        partitions: list[Callable[[tuple[int, ...]], int]] | None = None,
        lock_doors_fn: Callable[[tuple[int, ...]], tuple[int, ...]] | None = None,
        goal: np.ndarray | None = None,
        obs_seed: int | None = 0,
        render_mode: str | None = None,
    ) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        if not 1 <= lock_length <= K:
            raise ValueError(f"lock_length must be in [1, {K}], got {lock_length}")
        if not 0.0 <= slip_prob <= 1.0:
            raise ValueError(f"slip_prob must be in [0, 1], got {slip_prob}")
        if agent_view_size < 3 or agent_view_size % 2 == 0:
            raise ValueError(
                f"agent_view_size must be odd and >= 3, got {agent_view_size}"
            )

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
        self.slip_prob = slip_prob
        self.grid_size = grid_size
        self.agent_view_size = agent_view_size

        # Sample the layout. Default layout_seed = obs_seed so the runner's
        # per-block obs_seed gives layout-per-block randomization for free.
        if layout_seed is None:
            layout_seed = 0 if obs_seed is None else int(obs_seed)
        self.layout_seed = layout_seed
        self._layout = sample_layout(grid_size, K, M, layout_seed)
        self._lock_doors_fn = lock_doors_fn

        # Reorder the user-supplied partitions according to the sampled
        # probe->bit permutation: probe i emits bit_{probe_to_bit[i]}.
        permuted_partitions = [partitions[b] for b in self._layout.probe_to_bit]

        n_reachable = len(self._layout.reachable_cells)
        self._n_reachable = n_reachable
        self.n_states = n_reachable * N_DIRS * (lock_length + 1) + 2
        self.n_actions = N_ACTIONS
        self.n_static = n_static_for_K(K)
        # M2 degenerate single-component factor structure (mirrors door-code).
        self.theta_spec = ThetaSpec(component_sizes=(self.n_static,))
        self.t_factor_spec = FactorSpec(theta_components=(0,))
        self.b_factor_specs: tuple[FactorSpec, ...] = (
            FactorSpec(theta_components=()),       # B_position (FOV)
            FactorSpec(theta_components=(0,)),     # B_probe
            FactorSpec(theta_components=()),       # B_outcome
        )

        self._SUCCESS = _success_index(n_reachable, lock_length)
        self._FAILURE = _failure_index(n_reachable, lock_length)
        self._START_REACH = self._layout.cell_to_state_idx[self._layout.start_cell]
        self._START_DIR = self._layout.start_dir
        self._START = _state_index(
            self._START_REACH, self._START_DIR, 0, lock_length,
        )
        self._terminals = {self._SUCCESS, self._FAILURE}
        self._door_cell_to_idx = {
            cell: i for i, cell in enumerate(self._layout.door_cells)
        }
        self._probe_cell_to_idx = {
            cell: i for i, cell in enumerate(self._layout.probe_cells)
        }

        self._perms = enumerate_permutations(K)
        self._partition_table = build_partition_table(K, permuted_partitions)
        self._unique_codewords = partitions_uniquely_identify(self._partition_table)

        # Inner Farama env for layout + rendering. Constructed once and
        # mirrored from the realized state index after every step().
        self._farama = _DoorCodeFaramaEnv(
            self._layout, agent_view_size=agent_view_size,
            render_mode=render_mode,
        )
        self._farama.reset(seed=0)

        # Tensors.
        self._T = create_transition_tensor(
            self._layout, K, lock_length, slip_prob,
            lock_doors_fn=lock_doors_fn,
        )
        self._B_list, self._state_to_view, self._n_views = (
            create_observation_tensors(
                self._farama, self._layout, K, lock_length,
                self._partition_table, probe_noise=probe_noise,
            )
        )
        if goal is None:
            self._goal = create_goal(self._layout, K, lock_length)
        else:
            expected_shape = (self.n_states, self.n_static)
            if goal.shape != expected_shape:
                raise ValueError(
                    f"goal shape {goal.shape} != expected {expected_shape}"
                )
            self._goal = np.asarray(goal)
        self._registry = create_modality_registry(
            self._layout, K, lock_length, n_views=self._n_views,
        )

        self._rng = np.random.default_rng(obs_seed)
        self._agent_state: int = self._START
        self._theta: int = 0

    # --- Environment protocol ------------------------------------------

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
        self._sync_farama()
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
        if self._agent_state in self._terminals:
            obs, obs_per_channel = self._make_obs()
            return StepResult(
                obs=obs, reward=0.0, terminated=True, truncated=False,
                obs_per_channel=obs_per_channel,
            )

        # T is the marginal of the slip-then-det process; sample from it
        # directly. Single source of truth for env stochasticity.
        row = np.asarray(
            self._T[:, self._agent_state, self._theta, action], dtype=np.float64,
        )
        next_state = int(self._rng.choice(self.n_states, p=row))
        self._agent_state = next_state
        self._sync_farama()

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

    def _sync_farama(self) -> None:
        """Mirror agent pose + door-open pattern from the current state
        index to the inner Farama env so renders reflect live state."""
        s = self._agent_state
        if s == self._SUCCESS:
            # Pose: place the agent on the goal cell (it just walked there)
            # so the rendered final frame visually matches "solved".
            # Agent's heading on the goal is unknown from state alone — use
            # the inner env's last set heading, which is the heading the
            # agent had when stepping forward onto the goal.
            self._farama.set_pose(self._layout.goal_cell, self._farama.agent_dir)
            lock = self._lock_doors_fn(self._perms[self._theta])
            for j in range(self.K):
                self._farama.set_door_open(j, j in lock)
            return
        if s == self._FAILURE:
            # Doors stay where they were; nothing to mirror.
            return
        reach_idx, dir, stage = _decode_state(s, self.L)
        cell = self._layout.reachable_cells[reach_idx]
        self._farama.set_pose(cell, dir)
        lock = self._lock_doors_fn(self._perms[self._theta])
        for j in range(self.K):
            self._farama.set_door_open(j, j in lock[:stage])

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

    def get_state_to_view(self) -> np.ndarray:
        """Length-n_states array mapping each non-terminal state to its
        FOV view-index (terminals map to the sentinel view)."""
        return self._state_to_view

    def render(self):
        return self._farama.render()

    # --- Properties ----------------------------------------------------

    @property
    def n_views(self) -> int:
        return self._n_views

    @property
    def theta(self) -> int:
        return self._theta

    @property
    def theta_permutation(self) -> tuple[int, ...]:
        return self._perms[self._theta]

    @property
    def lock_doors(self) -> tuple[int, ...]:
        return self._lock_doors_fn(self._perms[self._theta])

    @property
    def agent_state(self) -> int:
        return self._agent_state

    @property
    def agent_cell(self) -> int:
        """Flat-cell index of the agent, or -1 in terminal states."""
        if self._agent_state in self._terminals:
            return -1
        reach_idx, _dir, _stage = _decode_state(self._agent_state, self.L)
        return self._layout.reachable_cells[reach_idx]

    @property
    def agent_dir(self) -> int:
        if self._agent_state in self._terminals:
            return -1
        _reach, dir, _stage = _decode_state(self._agent_state, self.L)
        return dir

    @property
    def agent_stage(self) -> int:
        if self._agent_state in self._terminals:
            return -1
        _reach, _dir, stage = _decode_state(self._agent_state, self.L)
        return stage

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
    def layout(self) -> Layout:
        return self._layout

    @property
    def partitions_uniquely_identify_theta(self) -> bool:
        return self._unique_codewords

    def state_to_cell(self, state: int) -> int:
        """Public helper: flat-cell index for a non-terminal state, -1 for
        terminals. Useful for tests that walk the agent and check cells."""
        if state in self._terminals:
            return -1
        reach_idx, _dir, _stage = _decode_state(state, self.L)
        return self._layout.reachable_cells[reach_idx]

    # --- Internals -----------------------------------------------------

    def _make_obs(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Position modality: theta-independent one-hot over view-index.
        position = np.zeros(self._n_views)
        view_idx = int(self._state_to_view[self._agent_state])
        position[view_idx] = 1.0

        probe = np.zeros(N_PROBE_OBS)
        if self._agent_state not in self._terminals:
            reach_idx, _dir, _stage = _decode_state(self._agent_state, self.L)
            cell = self._layout.reachable_cells[reach_idx]
            if cell in self._probe_cell_to_idx:
                probe_idx = self._probe_cell_to_idx[cell]
                true_bit = int(self._partition_table[probe_idx, self._theta])
                if self.probe_noise > 0:
                    flipped = self._rng.random() < self.probe_noise
                else:
                    flipped = False
                emitted_bit = (1 - true_bit) if flipped else true_bit
                probe[emitted_bit] = 1.0
            else:
                probe[PROBE_NEUTRAL] = 1.0
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
