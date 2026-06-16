"""Tests for the Farama-MiniGrid-backed MiniGridDoorCodeEnv.

Coverage:
- Construction with default and varied (grid_size, K, L, M, slip_prob).
- Tensor shapes (T, theta-INDEPENDENT B_position, B_probe, B_outcome).
- Per-source normalization of T.
- Layout sanity: doors / probes / start / goal / walls pairwise disjoint.
- View-index hashing: consistent on resets; theta-INDEPENDENT.
- Deterministic dynamics under slip_prob=0: turn-left/right/forward/toggle.
- Slip semantics under slip_prob>0: T-row matches the StochasticActionWrapper
  marginal; step() samples from T directly (distributional check).
- Wall and closed-door blocking on FORWARD.
- Probe emission stand-on semantics.
- Toggle semantics under theta: correct toggle advances stage, wrong toggle
  -> FAILURE, toggle adjacent to non-door / facing wall -> no-op,
  toggle at stage=L is no-op.
- Goal-cell semantics: stepping onto goal_cell at stage=L -> SUCCESS.
- Layout randomization: layout_seed determinism + reset doesn't change
  layout.
- Modality registry: 3 channels, position is INDEPENDENT.
- Environment Protocol conformance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from learningpomdp.environments.base import Environment, StepResult
from learningpomdp.environments.door_code import (
    OUTCOME_FAILURE,
    OUTCOME_NEUTRAL,
    OUTCOME_SUCCESS,
    PROBE_BIT_0,
    PROBE_BIT_1,
    PROBE_NEUTRAL,
)
from learningpomdp.environments.minigrid_door_code import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_TOGGLE,
    MiniGridDoorCodeEnv,
    N_ACTIONS,
    N_DIRS,
    _coords,
    _decode_state,
    _flat,
    _state_index,
)
from learningpomdp.observations import ThetaDependence


# --- Helpers --------------------------------------------------------------


def _set_agent(env: MiniGridDoorCodeEnv, cell: int, dir: int, stage: int) -> None:
    """Force the agent into a specific (cell, dir, stage); used to pin
    test conditions independent of layout randomization."""
    reach_idx = env.layout.cell_to_state_idx[cell]
    env._agent_state = _state_index(reach_idx, dir, stage, env.L)
    env._sync_farama()


def _walk_facing(env: MiniGridDoorCodeEnv, target_dir: int) -> None:
    """Turn in place until facing `target_dir`."""
    while env.agent_dir != target_dir:
        env.step(ACTION_RIGHT)


def _walk_to(env: MiniGridDoorCodeEnv, target_cell: int, max_steps: int = 50) -> None:
    """Naive BFS walker: turns and forwards until at target_cell.
    Aborts after max_steps to prevent infinite loops on impossible goals.
    Used by tests that need to navigate from start to a specific cell
    without baking in layout-specific action sequences."""
    target_col, target_row = _coords(target_cell, env.grid_size)
    for _ in range(max_steps):
        if env.agent_cell == target_cell:
            return
        col, row = _coords(env.agent_cell, env.grid_size)
        # Greedy: face whichever cardinal moves us closer (col first).
        if col < target_col:
            want_dir = 0
        elif col > target_col:
            want_dir = 2
        elif row < target_row:
            want_dir = 1
        else:
            want_dir = 3
        if env.agent_dir != want_dir:
            # Rotate the short way.
            diff = (want_dir - env.agent_dir) % 4
            action = ACTION_RIGHT if diff <= 2 else ACTION_LEFT
            env.step(action)
            continue
        env.step(ACTION_FORWARD)
    raise RuntimeError(
        f"_walk_to gave up reaching {target_cell} from start "
        f"(currently at {env.agent_cell})"
    )


# --- Construction & shapes ------------------------------------------------


def test_construction_defaults():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    assert env.K == 3
    assert env.M == 3  # default partitions for K=3
    assert env.L == 1
    assert env.n_actions == N_ACTIONS == 4
    assert env.n_static == math.factorial(3)
    n_reachable = len(env.layout.reachable_cells)
    expected_states = n_reachable * N_DIRS * (env.L + 1) + 2
    assert env.n_states == expected_states


def test_zero_arg_construction_works():
    """Class defaults must construct without raising. Defaults match the
    canonical M2 ablation config (K=2, L=1, grid_size=5)."""
    env = MiniGridDoorCodeEnv()
    assert env.K == 2
    assert env.M == 1  # 1-bit codeword for 2 perms
    assert env.L == 1
    assert env.grid_size == 5
    res = env.reset(theta=0)
    assert res.terminated is False
    res = env.step(ACTION_FORWARD)
    assert isinstance(res, StepResult)


@pytest.mark.parametrize("K,L,grid_size", [
    (2, 1, 5),
    (3, 1, 6),
    (3, 2, 6),
    (4, 1, 8),
    (4, 2, 8),
])
def test_tensor_shapes(K, L, grid_size):
    env = MiniGridDoorCodeEnv(K=K, lock_length=L, grid_size=grid_size, layout_seed=0)
    n_reachable = len(env.layout.reachable_cells)
    n_states = n_reachable * N_DIRS * (L + 1) + 2
    n_static = math.factorial(K)
    T = env.get_transition_tensor()
    B = env.get_observation_tensors()
    assert T.shape == (n_states, n_states, n_static, N_ACTIONS)
    assert len(B) == 3
    B_position, B_probe, B_outcome = B
    assert B_position.shape == (env.n_views, n_states), (
        "position should be theta-independent: (n_views, n_states)"
    )
    assert B_probe.shape == (3, n_states, n_static)
    assert B_outcome.shape == (3, n_states)
    assert env.get_goal().shape == (n_states, n_static)


def test_grid_size_too_small_raises():
    with pytest.raises(ValueError, match="grid_size"):
        MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=4, layout_seed=0)
    with pytest.raises(ValueError, match="grid_size"):
        MiniGridDoorCodeEnv(K=4, lock_length=2, grid_size=6, layout_seed=0)


def test_K_color_pool_overflow_raises():
    with pytest.raises(ValueError, match="K must be in"):
        MiniGridDoorCodeEnv(K=6, lock_length=1, grid_size=10, layout_seed=0)


def test_invalid_lock_length_raises():
    with pytest.raises(ValueError, match="lock_length"):
        MiniGridDoorCodeEnv(K=3, lock_length=4, grid_size=6, layout_seed=0)


def test_invalid_slip_prob_raises():
    with pytest.raises(ValueError, match="slip_prob"):
        MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=-0.1, layout_seed=0)
    with pytest.raises(ValueError, match="slip_prob"):
        MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=1.5, layout_seed=0)


def test_invalid_agent_view_size_raises():
    with pytest.raises(ValueError, match="agent_view_size"):
        MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, agent_view_size=2, layout_seed=0)
    with pytest.raises(ValueError, match="agent_view_size"):
        MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, agent_view_size=4, layout_seed=0)


# --- Layout sanity --------------------------------------------------------


def test_layout_disjoint_in_bounds():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    layout = env.layout
    door_set = set(layout.door_cells)
    probe_set = set(layout.probe_cells)
    wall_set = set(layout.wall_cells)
    start = layout.start_cell
    goal = layout.goal_cell
    # Pairwise disjoint.
    assert door_set.isdisjoint(probe_set)
    assert door_set.isdisjoint(wall_set)
    assert probe_set.isdisjoint(wall_set)
    assert start not in door_set
    assert start not in probe_set
    assert start not in wall_set
    assert goal not in door_set
    assert goal not in probe_set
    assert goal not in wall_set
    assert goal != start
    n_cells = layout.grid_size * layout.grid_size
    for cell in door_set | probe_set | wall_set | {start, goal}:
        assert 0 <= cell < n_cells
    assert len(door_set) == env.K
    assert len(probe_set) == env.M
    # Reachable cells = interior minus walls.
    assert len(layout.reachable_cells) > 0
    for cell in layout.reachable_cells:
        assert cell not in wall_set


def test_layout_seed_determinism():
    """Same layout_seed -> identical layout; different seeds -> different
    layout on at least one of {door rows, start, goal, colors}."""
    env_a = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    env_b = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    env_c = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=42)
    a, b, c = env_a.layout, env_b.layout, env_c.layout
    assert a.door_cells == b.door_cells
    assert a.probe_cells == b.probe_cells
    assert a.start_cell == b.start_cell
    assert a.goal_cell == b.goal_cell
    assert a.door_colors == b.door_colors
    assert a.probe_to_bit == b.probe_to_bit
    diff = (
        a.door_cells != c.door_cells
        or a.probe_cells != c.probe_cells
        or a.start_cell != c.start_cell
        or a.goal_cell != c.goal_cell
        or a.door_colors != c.door_colors
        or a.probe_to_bit != c.probe_to_bit
    )
    assert diff, "layout_seed=0 and =42 produced identical layouts"


def test_layout_fixed_within_block():
    """Resetting with different theta does not change the layout."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    snapshot = env.layout
    for theta in range(env.n_static):
        env.reset(theta=theta)
        assert env.layout is snapshot


# --- T tensor row-stochastic ---------------------------------------------


@pytest.mark.parametrize("slip_prob", [0.0, 0.3, 1.0])
def test_T_row_stochastic_from_reachable(slip_prob):
    env = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=slip_prob, layout_seed=0,
    )
    T = env.get_transition_tensor()
    n_reachable = len(env.layout.reachable_cells)
    for reach_idx in range(n_reachable):
        for d in range(N_DIRS):
            for s in range(env.L + 1):
                s_old = _state_index(reach_idx, d, s, env.L)
                for theta_idx in range(env.n_static):
                    for u in range(N_ACTIONS):
                        col_sum = T[:, s_old, theta_idx, u].sum()
                        assert col_sum == pytest.approx(1.0, abs=1e-9), (
                            f"non-row-stochastic at s={s_old}, theta={theta_idx}, u={u}"
                        )
    SUCCESS = env.success_state_index
    FAILURE = env.failure_state_index
    for theta_idx in range(env.n_static):
        for u in range(N_ACTIONS):
            assert T[SUCCESS, SUCCESS, theta_idx, u] == 1.0
            assert T[FAILURE, FAILURE, theta_idx, u] == 1.0


# --- View hashing & FOV theta-independence -------------------------------


def test_state_to_view_shape_and_bijective_on_resets():
    env = MiniGridDoorCodeEnv(K=3, lock_length=2, grid_size=6, probe_noise=0.0, layout_seed=0)
    table = env.get_state_to_view()
    assert table.shape == (env.n_states,)
    env.reset(theta=0)
    s = env.agent_state
    obs1 = env._make_obs()[1]["position"].argmax()
    env.reset(theta=0)
    obs2 = env._make_obs()[1]["position"].argmax()
    assert obs1 == obs2
    assert table[s] == obs1


def test_position_channel_is_theta_independent():
    """The position observation at a fixed (cell, dir, stage) should not
    vary with theta — that's the whole point of G1."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=2, grid_size=6, probe_noise=0.0, layout_seed=0)
    # Pick a non-probe interior cell so the probe channel doesn't get in
    # the way (we're only inspecting position).
    layout = env.layout
    interior_non_probe = next(
        c for c in layout.reachable_cells
        if c not in layout.probe_cells and c != layout.start_cell
    )
    views_per_theta = []
    for theta in range(env.n_static):
        env.reset(theta=theta)
        _set_agent(env, interior_non_probe, dir=0, stage=0)
        obs = env._make_obs()[1]["position"]
        views_per_theta.append(int(obs.argmax()))
    assert len(set(views_per_theta)) == 1, (
        f"position view changed across theta: {views_per_theta}"
    )


def test_doors_render_as_locked_then_open():
    """Doors are placed locked (FOV state byte = 2). After
    `set_door_open`, FOV state byte flips to 0 (open). Render-fidelity
    check; the agent's POSITION channel is enumerated with all doors
    locked so this is purely about the renderer."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, probe_noise=0.0, layout_seed=0)
    farama = env._farama
    farama.reset_doors()
    door0_cell = env.layout.door_cells[0]
    col, row = _coords(door0_cell, env.grid_size)
    obj = farama.grid.get(col, row)
    enc = obj.encode()
    assert enc[2] == 2, f"expected locked state=2, got {enc}"
    farama.set_door_open(0, True)
    enc = farama.grid.get(col, row).encode()
    assert enc[2] == 0
    farama.set_door_open(0, False)
    enc = farama.grid.get(col, row).encode()
    assert enc[2] == 2


def test_view_idx_terminals_share_sentinel():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    table = env.get_state_to_view()
    SUCCESS = env.success_state_index
    FAILURE = env.failure_state_index
    assert table[SUCCESS] == table[FAILURE]


# --- Deterministic dynamics ----------------------------------------------


def test_left_right_rotate_in_place():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    cell0 = env.agent_cell
    dir0 = env.agent_dir
    env.step(ACTION_LEFT)
    assert env.agent_cell == cell0
    assert env.agent_dir == (dir0 - 1) % N_DIRS
    env.step(ACTION_RIGHT)
    assert env.agent_dir == dir0


def test_forward_walks_into_open_corridor():
    """Forward into a non-wall, non-door cell moves one step."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    # Pick a reachable cell with at least one walkable neighbor, place
    # the agent there facing that neighbor.
    layout = env.layout
    for cell in layout.reachable_cells:
        if cell in layout.door_cells:
            continue
        col, row = _coords(cell, env.grid_size)
        for dir, (dx, dy) in [(0, (1, 0)), (1, (0, 1)), (2, (-1, 0)), (3, (0, -1))]:
            n_col, n_row = col + dx, row + dy
            if not (0 <= n_col < env.grid_size and 0 <= n_row < env.grid_size):
                continue
            n_cell = _flat(n_col, n_row, env.grid_size)
            if n_cell in layout.cell_to_state_idx and n_cell not in layout.door_cells:
                _set_agent(env, cell, dir, stage=0)
                env.step(ACTION_FORWARD)
                assert env.agent_cell == n_cell, (
                    f"forward from {cell} dir={dir} should land at {n_cell}, "
                    f"got {env.agent_cell}"
                )
                return
    pytest.fail("no walkable forward-step found in layout (geometry too tight)")


def test_forward_blocked_by_wall():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    layout = env.layout
    # Find a reachable cell with a wall neighbor; place agent facing it.
    for cell in layout.reachable_cells:
        col, row = _coords(cell, env.grid_size)
        for dir, (dx, dy) in [(0, (1, 0)), (1, (0, 1)), (2, (-1, 0)), (3, (0, -1))]:
            n_col, n_row = col + dx, row + dy
            if not (0 <= n_col < env.grid_size and 0 <= n_row < env.grid_size):
                continue
            n_cell = _flat(n_col, n_row, env.grid_size)
            if n_cell in layout.wall_cells:
                _set_agent(env, cell, dir, stage=0)
                env.step(ACTION_FORWARD)
                assert env.agent_cell == cell, (
                    f"forward from {cell} dir={dir} into wall {n_cell} should "
                    f"be blocked, got {env.agent_cell}"
                )
                return
    pytest.fail("no cell-with-wall-neighbor found in layout")


def test_forward_blocked_by_closed_door():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    layout = env.layout
    # Find a reachable cell with a door neighbor; agent face the door at
    # stage 0 (no doors open) -> blocked.
    for cell in layout.reachable_cells:
        col, row = _coords(cell, env.grid_size)
        for dir, (dx, dy) in [(0, (1, 0)), (1, (0, 1)), (2, (-1, 0)), (3, (0, -1))]:
            n_col, n_row = col + dx, row + dy
            n_cell = _flat(n_col, n_row, env.grid_size)
            if n_cell in layout.door_cells:
                _set_agent(env, cell, dir, stage=0)
                env.step(ACTION_FORWARD)
                assert env.agent_cell == cell, (
                    f"forward into closed door {n_cell} should be blocked"
                )
                return
    pytest.fail("no cell-with-door-neighbor found in layout")


def test_toggle_correct_door_advances_stage():
    """Place agent in front of lock[0]; toggle correctly -> stage 1."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=2, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)  # perm (0,1,2), lock = (0, 1)
    lock = env.lock_doors
    door_cell = env.layout.door_cells[lock[0]]
    door_col, door_row = _coords(door_cell, env.grid_size)
    # Approach cell: one step left of the door (we face right -> door in front).
    approach_cell = _flat(door_col - 1, door_row, env.grid_size)
    _set_agent(env, approach_cell, dir=0, stage=0)
    res = env.step(ACTION_TOGGLE)
    assert env.agent_stage == 1, f"expected stage 1, got {env.agent_stage}"
    assert not res.terminated


def test_toggle_correct_final_door_advances_to_stage_L():
    """L=1: toggling lock[0] correctly advances to stage 1 (all doors
    open). SUCCESS only fires when the agent then walks onto goal_cell."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    lock = env.lock_doors
    door_cell = env.layout.door_cells[lock[0]]
    door_col, door_row = _coords(door_cell, env.grid_size)
    approach_cell = _flat(door_col - 1, door_row, env.grid_size)
    _set_agent(env, approach_cell, dir=0, stage=0)
    res = env.step(ACTION_TOGGLE)
    assert env.agent_stage == env.L
    assert not res.terminated
    assert env.agent_state != env.success_state_index


def test_toggle_wrong_door_kills():
    """Toggle a non-lock[stage] door at stage 0 -> FAILURE."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    lock = env.lock_doors
    wrong_door_idx = next(j for j in range(env.K) if j != lock[0])
    door_cell = env.layout.door_cells[wrong_door_idx]
    door_col, door_row = _coords(door_cell, env.grid_size)
    approach_cell = _flat(door_col - 1, door_row, env.grid_size)
    _set_agent(env, approach_cell, dir=0, stage=0)
    res = env.step(ACTION_TOGGLE)
    assert env.agent_state == env.failure_state_index
    assert res.reward == -1.0
    assert res.terminated


def test_toggle_facing_wall_or_floor_is_noop():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    s_before = env.agent_state
    # At start, the agent faces right; the cell in front is on col=2 of
    # the left half — possibly a Floor or wall (depends on layout). In
    # the canonical default_layout(seed=0) it's plain interior. This
    # toggle should be a no-op regardless.
    res = env.step(ACTION_TOGGLE)
    assert env.agent_state == s_before
    assert not res.terminated


def test_toggle_at_stage_L_is_noop():
    """At stage = L (all doors open), TOGGLE in front of any door is a
    no-op (the door is already open)."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    lock = env.lock_doors
    door_cell = env.layout.door_cells[lock[0]]
    door_col, door_row = _coords(door_cell, env.grid_size)
    approach_cell = _flat(door_col - 1, door_row, env.grid_size)
    _set_agent(env, approach_cell, dir=0, stage=env.L)
    s_before = env.agent_state
    res = env.step(ACTION_TOGGLE)
    assert env.agent_state == s_before, "toggle at stage=L should be no-op"
    assert not res.terminated


def test_toggle_in_terminal_is_absorbing():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    env._agent_state = env.failure_state_index
    res = env.step(ACTION_FORWARD)
    assert env.agent_state == env.failure_state_index
    assert res.terminated


# --- Goal-cell semantics --------------------------------------------------


def test_goal_cell_unreachable_pre_stage_L():
    """At stage < L, walking adjacent-to-goal-cell with stage 0 doesn't
    teleport; specifically, the only path to goal goes through closed
    lock doors which block."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    layout = env.layout
    goal_col, goal_row = _coords(layout.goal_cell, env.grid_size)
    # Find an in-bounds neighbor of goal that is NOT a wall (so it's a
    # reachable cell adjacent to the goal). Then force agent there at
    # stage=0 facing the goal.
    for dir, (dx, dy) in [(0, (1, 0)), (1, (0, 1)), (2, (-1, 0)), (3, (0, -1))]:
        n_col, n_row = goal_col - dx, goal_row - dy  # cell that faces goal
        if not (0 <= n_col < env.grid_size and 0 <= n_row < env.grid_size):
            continue
        approach = _flat(n_col, n_row, env.grid_size)
        if approach in layout.cell_to_state_idx and approach != layout.goal_cell:
            _set_agent(env, approach, dir=dir, stage=0)
            res = env.step(ACTION_FORWARD)
            # Either we walked into the goal (only if stage=L would have
            # been required... but stage=0 doesn't allow SUCCESS) — must
            # have stayed at approach OR landed somewhere reachable that
            # isn't SUCCESS.
            assert env.agent_state != env.success_state_index, (
                "stepping onto goal at stage<L should not produce SUCCESS"
            )
            assert not res.terminated
            return


def test_goal_cell_reach_succeeds():
    """Force the agent into stage=L adjacent to goal, walk forward,
    expect SUCCESS."""
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, slip_prob=0.0, layout_seed=0)
    env.reset(theta=0)
    layout = env.layout
    goal_col, goal_row = _coords(layout.goal_cell, env.grid_size)
    # Find a reachable approach cell + dir such that forward lands on goal.
    for dir, (dx, dy) in [(0, (1, 0)), (1, (0, 1)), (2, (-1, 0)), (3, (0, -1))]:
        n_col, n_row = goal_col - dx, goal_row - dy
        if not (0 <= n_col < env.grid_size and 0 <= n_row < env.grid_size):
            continue
        approach = _flat(n_col, n_row, env.grid_size)
        if approach in layout.cell_to_state_idx and approach != layout.goal_cell:
            _set_agent(env, approach, dir=dir, stage=env.L)
            res = env.step(ACTION_FORWARD)
            assert env.agent_state == env.success_state_index, (
                f"expected SUCCESS, got state {env.agent_state}"
            )
            assert res.reward == 1.0
            assert res.terminated
            return
    pytest.fail("no reachable cell adjacent to goal_cell with valid heading")


# --- Slip empirical behaviour --------------------------------------------


def test_slip_prob_zero_is_deterministic():
    env_a = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=0.0, obs_seed=1, layout_seed=0,
    )
    env_b = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=0.0, obs_seed=2, layout_seed=0,
    )
    env_a.reset(theta=0)
    env_b.reset(theta=0)
    actions = [ACTION_LEFT, ACTION_FORWARD, ACTION_FORWARD, ACTION_FORWARD]
    for a in actions:
        env_a.step(a)
        env_b.step(a)
        assert env_a.agent_state == env_b.agent_state


def test_slip_prob_one_uniform_over_six_actions():
    """With slip_prob=1, T row should not depend on the planner action."""
    env = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=1.0, obs_seed=7, layout_seed=0,
    )
    T = env.get_transition_tensor()
    s_old = env.start_state_index
    rows = [T[:, s_old, 0, u] for u in range(N_ACTIONS)]
    for u in range(1, N_ACTIONS):
        np.testing.assert_allclose(rows[u], rows[0], atol=1e-12)


def test_step_samples_from_T_distributionally():
    """Empirical step distribution at a fixed (s, theta, u) under
    slip_prob > 0 should match T[:, s, theta, u] within sampling noise."""
    env = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=0.3, obs_seed=0, layout_seed=0,
    )
    env.reset(theta=0)
    s_fixed = env.agent_state
    theta_fixed = env._theta
    u_fixed = ACTION_FORWARD
    T_row = env.get_transition_tensor()[:, s_fixed, theta_fixed, u_fixed]
    counts = np.zeros(env.n_states, dtype=np.int64)
    n_trials = 5000
    for trial in range(n_trials):
        env._agent_state = s_fixed  # pin (theta already set in reset)
        res = env.step(u_fixed)
        counts[env.agent_state] += 1
    empirical = counts / n_trials
    # L1 distance, allow ~0.05 for 5000 samples on small support.
    l1 = float(np.abs(empirical - T_row).sum())
    assert l1 < 0.06, f"empirical step distribution diverges from T (L1={l1})"


# --- Probe emission ------------------------------------------------------


def test_probe_emits_at_probe_cell_any_heading():
    env = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=0.0, probe_noise=0.0, layout_seed=0,
    )
    env.reset(theta=0)
    probe_cell = env.layout.probe_cells[0]
    # partition_table row 0 already accounts for probe_to_bit permutation.
    expected_bit = int(env.get_partition_table()[0, 0])
    for dir in range(N_DIRS):
        _set_agent(env, probe_cell, dir=dir, stage=0)
        # _set_agent updates the agent_state but doesn't emit a step
        # observation; emit one via _make_obs.
        obs = env._make_obs()[1]["probe"]
        assert obs[expected_bit] == 1.0, (
            f"probe emission wrong at dir={dir}: {obs}"
        )


def test_probe_neutral_off_probe_cells():
    env = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=0.0, probe_noise=0.0, layout_seed=0,
    )
    res = env.reset(theta=0)
    if env.layout.start_cell in env.layout.probe_cells:
        # If start happens to land on a probe (rare layouts), pick a
        # different cell that isn't a probe.
        non_probe = next(
            c for c in env.layout.reachable_cells
            if c not in env.layout.probe_cells and c != env.layout.start_cell
        )
        _set_agent(env, non_probe, dir=0, stage=0)
        res_obs = env._make_obs()[1]["probe"]
        assert res_obs[PROBE_NEUTRAL] == 1.0
    else:
        assert res.obs_per_channel["probe"][PROBE_NEUTRAL] == 1.0


def test_probe_noise_calibration():
    """With probe_noise=0.2, emitted bit equals partition bit ~0.8 of the time."""
    rng_seed = 0
    env = MiniGridDoorCodeEnv(
        K=3, lock_length=1, grid_size=6, slip_prob=0.0, probe_noise=0.2,
        obs_seed=rng_seed, layout_seed=0,
    )
    true_bit = int(env.get_partition_table()[0, 0])  # probe 0 at theta 0
    probe_cell = env.layout.probe_cells[0]
    n_trials = 1000
    correct = 0
    for trial in range(n_trials):
        env.reset(theta=0, seed=rng_seed + trial)
        _set_agent(env, probe_cell, dir=0, stage=0)
        probe_obs = env._make_obs()[1]["probe"]
        emitted_bit = int(probe_obs[:2].argmax())
        if emitted_bit == true_bit:
            correct += 1
    p_correct = correct / n_trials
    assert 0.75 < p_correct < 0.85, f"expected ~0.8, got {p_correct}"


# --- Theta persistence ---------------------------------------------------


def test_reset_with_explicit_theta():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    for theta in range(env.n_static):
        res = env.reset(theta=theta)
        assert env.theta == theta
        assert res.terminated is False
        assert env.agent_state == env.start_state_index


def test_reset_theta_out_of_range_raises():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    with pytest.raises(ValueError, match="theta"):
        env.reset(theta=env.n_static)
    with pytest.raises(ValueError, match="theta"):
        env.reset(theta=-1)


# --- State indices -------------------------------------------------------


def test_state_index_layout():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    reach_idx, dir, stage = _decode_state(env.start_state_index, env.L)
    assert env.layout.reachable_cells[reach_idx] == env.layout.start_cell
    assert dir == env.layout.start_dir
    assert stage == 0
    n_reachable = len(env.layout.reachable_cells)
    assert env.success_state_index == n_reachable * N_DIRS * (env.L + 1)
    assert env.failure_state_index == n_reachable * N_DIRS * (env.L + 1) + 1


# --- Modality registry ---------------------------------------------------


def test_modality_registry():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    reg = env.get_modality_registry()
    assert reg.n_channels == 3
    names = [m.name for m in reg.modalities]
    assert names == ["position", "probe", "outcome"]
    pos = reg.get_modality_by_name("position")
    pr = reg.get_modality_by_name("probe")
    out = reg.get_modality_by_name("outcome")
    assert pos.theta_dependence == ThetaDependence.INDEPENDENT
    assert pr.theta_dependence == ThetaDependence.CONDITIONAL
    assert out.theta_dependence == ThetaDependence.INDEPENDENT
    assert pos.cardinality == env.n_views
    assert pr.cardinality == 3
    assert out.cardinality == 3
    expected_probe_states = {
        _state_index(env.layout.cell_to_state_idx[cell], d, s, env.L)
        for cell in env.layout.probe_cells
        for d in range(N_DIRS)
        for s in range(env.L + 1)
    }
    assert pr.informative_states == expected_probe_states


def test_protocol_conformance():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, layout_seed=0)
    assert isinstance(env, Environment)


# --- B_position and view emissions match ----------------------------------


def test_B_position_one_hot_at_view_idx():
    env = MiniGridDoorCodeEnv(K=3, lock_length=2, grid_size=6, probe_noise=0.0, layout_seed=0)
    B_pos = env.get_observation_tensors()[0]
    table = env.get_state_to_view()
    for s in range(env.n_states):
        v = int(table[s])
        assert B_pos[v, s] == 1.0
        np.testing.assert_allclose(B_pos[:, s].sum(), 1.0, atol=1e-12)


def test_obs_emission_matches_state_to_view():
    env = MiniGridDoorCodeEnv(K=3, lock_length=1, grid_size=6, probe_noise=0.0, layout_seed=0)
    table = env.get_state_to_view()
    res = env.reset(theta=2)
    pos_obs = res.obs_per_channel["position"]
    expected_idx = int(table[env.agent_state])
    assert pos_obs.argmax() == expected_idx
    res = env.step(ACTION_FORWARD)
    pos_obs = res.obs_per_channel["position"]
    assert pos_obs.argmax() == int(table[env.agent_state])
