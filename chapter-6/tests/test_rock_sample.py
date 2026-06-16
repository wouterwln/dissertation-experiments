"""Tests for the RockSample BA-POMDP env (M4)."""

from __future__ import annotations

import numpy as np
import pytest

from learningpomdp.environments.base import Environment
from learningpomdp.environments.rock_sample import (
    EVENT_OTHER,
    N_ROCK_OBS,
    ROCK_NO_INFO,
    RockSampleEnv,
    RockSampleLayout,
    action_role,
    create_observation_tensors,
    create_transition_tensor,
    decode_state,
    encode_state,
    informative_states_for_rock,
    move_cell,
    n_actions_for,
    n_states_for,
    sample_layout,
)
from learningpomdp.factors import FactorSpec, ThetaSpec


# --- Layout -----------------------------------------------------------------


def test_sample_layout_is_deterministic_per_seed():
    a = sample_layout(grid_size=3, K=2, seed=0)
    b = sample_layout(grid_size=3, K=2, seed=0)
    assert a == b


def test_sample_layout_changes_with_seed():
    a = sample_layout(grid_size=3, K=2, seed=0)
    b = sample_layout(grid_size=3, K=2, seed=1)
    assert a.rock_cells != b.rock_cells


def test_sample_layout_avoids_agent_start():
    layout = sample_layout(grid_size=3, K=2, seed=0)
    assert layout.agent_start_cell == 0
    assert 0 not in layout.rock_cells


def test_sample_layout_rocks_are_distinct():
    layout = sample_layout(grid_size=4, K=3, seed=2)
    assert len(set(layout.rock_cells)) == 3


def test_sample_layout_rejects_K_too_large():
    # 2x2 grid has 4 cells; one is reserved for the agent start.
    with pytest.raises(ValueError):
        sample_layout(grid_size=2, K=4, seed=0)


def test_layout_validates_distinct_rocks():
    with pytest.raises(ValueError):
        RockSampleLayout(grid_size=3, K=2, agent_start_cell=0, rock_cells=(1, 1))


def test_layout_validates_agent_not_on_rock():
    with pytest.raises(ValueError):
        RockSampleLayout(grid_size=3, K=2, agent_start_cell=1, rock_cells=(1, 2))


def test_layout_chebyshev_distance():
    layout = RockSampleLayout(grid_size=4, K=1, agent_start_cell=0, rock_cells=(15,))
    # Cell 15 = (col=3, row=3); cell 0 = (0, 0); Chebyshev = max(3, 3) = 3.
    assert layout.chebyshev_distance(0, 0) == 3
    # Cell 5 = (col=1, row=1); distance to rock at (3,3) = max(2,2) = 2.
    assert layout.chebyshev_distance(5, 0) == 2


# --- Encoding ---------------------------------------------------------------


def test_encode_decode_round_trip():
    K = 2
    n_events = K + 2
    for state in range(9 * 4 * n_events):
        pos, mask, event = decode_state(state, K, n_events)
        assert encode_state(pos, mask, event, K, n_events) == state


# --- Movement ---------------------------------------------------------------


def test_move_cell_clamps_at_edges():
    # 3x3 grid; cell 0 = (0,0).
    assert move_cell(0, 0, grid_size=3) == 0  # N at top edge -> stay
    assert move_cell(0, 3, grid_size=3) == 0  # W at left edge -> stay
    assert move_cell(0, 1, grid_size=3) == 3  # S -> (0,1)
    assert move_cell(0, 2, grid_size=3) == 1  # E -> (1,0)
    # Bottom-right cell 8 = (2,2).
    assert move_cell(8, 1, grid_size=3) == 8  # S at bottom edge
    assert move_cell(8, 2, grid_size=3) == 8  # E at right edge


def test_action_role():
    assert action_role(0, K=2) == ("move", 0)
    assert action_role(3, K=2) == ("move", 3)
    assert action_role(4, K=2) == ("sense", 0)
    assert action_role(5, K=2) == ("sense", 1)
    assert action_role(6, K=2) == ("sample", 0)


def test_action_role_rejects_out_of_range():
    with pytest.raises(ValueError):
        action_role(7, K=2)


# --- Env protocol -----------------------------------------------------------


def test_env_satisfies_environment_protocol():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    assert isinstance(env, Environment)


def test_env_dimensions_at_K2_grid3():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    # n_states = 9 cells × 2^K=4 masks × (K+2)=4 events = 144.
    assert env.n_states == 144
    # n_actions = 5 + K = 7.
    assert env.n_actions == 7
    # n_static = 2^(2K) = 16.
    assert env.n_static == 16


def test_theta_spec_layout():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    assert env.theta_spec.component_sizes == (2, 2, 2, 2)
    # First K components = quality (per-episode); next K = sensor (persistent).
    assert env.theta_spec.episode_reset_components == (0, 1)
    assert env.theta_spec.persistent_components == (2, 3)


def test_factor_specs():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    # T is θ-independent.
    assert env.t_factor_spec == FactorSpec(theta_components=())
    # B_position is θ-independent; rock_r depends on (quality_r, sensor_r).
    assert env.b_factor_specs[0] == FactorSpec(theta_components=())
    assert env.b_factor_specs[1] == FactorSpec(theta_components=(0, 2))
    assert env.b_factor_specs[2] == FactorSpec(theta_components=(1, 3))


# --- Tensor shapes ----------------------------------------------------------


def test_T_shape():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    T = env.get_transition_tensor()
    assert T.shape == (env.n_states, env.n_states, env.n_actions)


def test_T_is_a_proper_distribution_over_next_state():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    T = env.get_transition_tensor()
    # For every (state, action), T sums to 1 over next_state.
    sums = T.sum(axis=0)
    assert np.allclose(sums, 1.0)


def test_B_shapes():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    B_list = env.get_observation_tensors()
    assert len(B_list) == 1 + env.K  # position + K rocks
    assert B_list[0].shape == (env.layout.n_cells, env.n_states)
    for r in range(env.K):
        assert B_list[1 + r].shape == (N_ROCK_OBS, env.n_states, 2, 2)


def test_B_position_is_identity_on_cell():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    B_position = env.get_observation_tensors()[0]
    # Each column (state) sums to 1 over the cell axis.
    assert np.allclose(B_position.sum(axis=0), 1.0)
    # The single hot row matches the state's pos.
    K = env.K
    n_events = env.layout.n_events
    for state in range(env.n_states):
        pos, _, _ = decode_state(state, K, n_events)
        assert B_position[pos, state] == 1.0


def test_B_rock_normalizes_over_emissions():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    B_list = env.get_observation_tensors()
    for r in range(env.K):
        # For every (state, θ_q_r, θ_s_r) the emission distribution sums to 1.
        s = B_list[1 + r].sum(axis=0)
        assert np.allclose(s, 1.0)


def test_B_rock_no_info_default():
    """At a state with event=OTHER (no SENSE_r, no SAMPLE-reveal), the rock
    channel emits no_info deterministically regardless of θ."""
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    B_list = env.get_observation_tensors()
    K = env.K
    n_events = env.layout.n_events
    state = encode_state(pos=0, mask=0, event=EVENT_OTHER, K=K, n_events=n_events)
    for r in range(K):
        for theta_q in (0, 1):
            for theta_s in (0, 1):
                p = B_list[1 + r][:, state, theta_q, theta_s]
                assert p[ROCK_NO_INFO] == 1.0


def test_B_rock_sense_at_zero_distance_reliable_sensor():
    """SENSE_r at rock r's cell with reliable sensor (θ_s=0) should be
    near-deterministic on θ_quality_r."""
    env = RockSampleEnv(K=1, grid_size=3, layout_seed=0)
    K = env.K
    n_events = env.layout.n_events
    rock_cell = env.layout.rock_cells[0]
    sense_event = 1  # SENSE_0
    B_rock = env.get_observation_tensors()[1]
    state = encode_state(
        pos=rock_cell, mask=0, event=sense_event, K=K, n_events=n_events,
    )
    # alpha(0) = 1.0; reliable sensor (θ_s=0) -> alpha_eff = 1.0.
    for theta_q in (0, 1):
        p = B_rock[:, state, theta_q, 0]
        assert p[theta_q] == pytest.approx(1.0, abs=1e-6)
        assert p[1 - theta_q] == pytest.approx(0.0, abs=1e-6)
        assert p[ROCK_NO_INFO] == 0.0


def test_B_rock_sense_noisy_sensor_attenuated():
    """Noisy sensor (θ_s=1) at d=0 with eta=0.3 should give half-strength-
    attenuated accuracy 0.5 + 0.5 * (1 - 0.5) * 0.3 = 0.575 (and 0.425
    on the flipped bit)."""
    env = RockSampleEnv(K=1, grid_size=3, layout_seed=0, sensor_noise_eta=0.3)
    K = env.K
    n_events = env.layout.n_events
    rock_cell = env.layout.rock_cells[0]
    B_rock = env.get_observation_tensors()[1]
    state = encode_state(
        pos=rock_cell, mask=0, event=1, K=K, n_events=n_events,
    )
    # alpha(0) = 1.0 -> alpha_eff = 0.5 + 0.5*(1-0.5)*0.3 = 0.575.
    p = B_rock[:, state, 0, 1]  # θ_q=0, θ_s=1
    assert p[0] == pytest.approx(0.575, abs=1e-6)
    assert p[1] == pytest.approx(0.425, abs=1e-6)


def test_B_rock_sample_reveal_is_deterministic():
    """At a state representing 'just sampled rock r' (event=SAMPLE,
    pos=rock_r, mask[r]=1), the rock channel emits θ_quality_r
    deterministically — sample reveals the truth, regardless of sensor."""
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    K = env.K
    n_events = env.layout.n_events
    sample_event = K + 1
    B_list = env.get_observation_tensors()
    for r in range(K):
        rock_cell = env.layout.rock_cells[r]
        state = encode_state(
            pos=rock_cell,
            mask=(1 << r),
            event=sample_event,
            K=K,
            n_events=n_events,
        )
        for theta_q in (0, 1):
            for theta_s in (0, 1):
                p = B_list[1 + r][:, state, theta_q, theta_s]
                assert p[theta_q] == pytest.approx(1.0)
                assert p[1 - theta_q] == pytest.approx(0.0)


# --- T transition correctness ----------------------------------------------


def test_T_move_advances_position():
    env = RockSampleEnv(K=1, grid_size=3, layout_seed=0)
    K = env.K
    n_events = env.layout.n_events
    T = env.get_transition_tensor()
    # From (pos=0, mask=0, event=OTHER), MOVE_S (action 1) should land at
    # (pos=3, mask=0, event=OTHER).
    state = encode_state(0, 0, EVENT_OTHER, K, n_events)
    expected = encode_state(3, 0, EVENT_OTHER, K, n_events)
    next_dist = T[:, state, 1]
    assert next_dist.argmax() == expected
    assert next_dist.sum() == pytest.approx(1.0)


def test_T_sense_does_not_move():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    K = env.K
    n_events = env.layout.n_events
    T = env.get_transition_tensor()
    state = encode_state(4, 0, EVENT_OTHER, K, n_events)
    # SENSE_1 = action 5; expected next state = (pos=4, mask=0, event=SENSE_1=2).
    expected = encode_state(4, 0, 2, K, n_events)
    assert T[:, state, 5].argmax() == expected


def test_T_sample_at_rock_flips_mask():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    K = env.K
    n_events = env.layout.n_events
    T = env.get_transition_tensor()
    sample_action = 4 + K  # 6
    sample_event = K + 1   # 3
    rock_cell = env.layout.rock_cells[0]  # rock 0
    state = encode_state(rock_cell, 0, EVENT_OTHER, K, n_events)
    expected = encode_state(rock_cell, 1, sample_event, K, n_events)  # mask[0]=1
    assert T[:, state, sample_action].argmax() == expected


def test_T_sample_at_non_rock_no_op():
    """SAMPLE off any rock cell leaves mask unchanged with event=OTHER."""
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    K = env.K
    n_events = env.layout.n_events
    T = env.get_transition_tensor()
    sample_action = 4 + K
    non_rock_cell = env.layout.agent_start_cell  # known to not be a rock
    state = encode_state(non_rock_cell, 0, EVENT_OTHER, K, n_events)
    expected = encode_state(non_rock_cell, 0, EVENT_OTHER, K, n_events)
    assert T[:, state, sample_action].argmax() == expected


def test_T_sample_already_sampled_no_op():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    K = env.K
    n_events = env.layout.n_events
    T = env.get_transition_tensor()
    sample_action = 4 + K
    rock_cell = env.layout.rock_cells[0]
    pre_state = encode_state(rock_cell, 1, EVENT_OTHER, K, n_events)  # mask[0]=1 already
    expected = encode_state(rock_cell, 1, EVENT_OTHER, K, n_events)
    assert T[:, pre_state, sample_action].argmax() == expected


# --- Reset / step semantics -------------------------------------------------


def test_reset_returns_to_start_state():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    env.reset(theta=0)
    assert env.agent_state == env.start_state_index
    K = env.K
    n_events = env.layout.n_events
    pos, mask, event = decode_state(env.agent_state, K, n_events)
    assert pos == env.layout.agent_start_cell
    assert mask == 0
    assert event == EVENT_OTHER


def test_reset_with_explicit_theta_records_components():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    # theta_flat = 5; component_sizes = (2,2,2,2); 5 = 0*8 + 1*4 + 0*2 + 1*1 -> (0,1,0,1)
    env.reset(theta=5)
    assert env.theta == 5
    assert env.theta_components == (0, 1, 0, 1)


def test_reset_random_theta_in_range():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0, obs_seed=42)
    env.reset()
    assert 0 <= env.theta < env.n_static


def test_step_reward_at_good_rock_sample():
    """Sample on a rock cell with quality=1 yields +sample_reward."""
    env = RockSampleEnv(K=1, grid_size=3, layout_seed=0, sample_reward=1.0)
    K = env.K
    # theta_flat must encode quality_0=1; component_sizes = (2, 2).
    # ravel((q0=1, s0=0)) = 2.
    env.reset(theta=2)
    # Walk to the rock cell. Use known layout from seed=0.
    rock_cell = env.layout.rock_cells[0]
    # Compute manhattan path (cheaty: just iterate moves).
    grid_size = env.grid_size
    rx, ry = rock_cell % grid_size, rock_cell // grid_size
    # Walk south then east.
    for _ in range(ry):
        env.step(1)  # MOVE_S
    for _ in range(rx):
        env.step(2)  # MOVE_E
    # Confirm position.
    pos, _, _ = decode_state(env.agent_state, K, env.layout.n_events)
    assert pos == rock_cell
    sample_action = 4 + K  # SAMPLE
    step_result = env.step(sample_action)
    assert step_result.reward == pytest.approx(1.0)


def test_step_reward_at_bad_rock_sample():
    env = RockSampleEnv(K=1, grid_size=3, layout_seed=0, sample_reward=1.0)
    K = env.K
    # quality_0 = 0; ravel((0, 0)) = 0.
    env.reset(theta=0)
    rock_cell = env.layout.rock_cells[0]
    grid_size = env.grid_size
    rx, ry = rock_cell % grid_size, rock_cell // grid_size
    for _ in range(ry):
        env.step(1)
    for _ in range(rx):
        env.step(2)
    sample_action = 4 + K
    step_result = env.step(sample_action)
    assert step_result.reward == pytest.approx(-1.0)


def test_step_reward_zero_off_rock():
    env = RockSampleEnv(K=1, grid_size=3, layout_seed=0)
    env.reset(theta=2)
    sample_action = 4 + env.K
    # Sample at start cell (not a rock) -> 0 reward.
    step_result = env.step(sample_action)
    assert step_result.reward == 0.0


def test_step_reward_zero_on_already_sampled_rock():
    env = RockSampleEnv(K=1, grid_size=3, layout_seed=0, sample_reward=1.0)
    K = env.K
    env.reset(theta=2)  # quality_0 = 1
    rock_cell = env.layout.rock_cells[0]
    grid_size = env.grid_size
    rx, ry = rock_cell % grid_size, rock_cell // grid_size
    for _ in range(ry):
        env.step(1)
    for _ in range(rx):
        env.step(2)
    sample_action = 4 + K
    first = env.step(sample_action)
    assert first.reward == pytest.approx(1.0)
    second = env.step(sample_action)
    assert second.reward == 0.0


# --- Modality registry ------------------------------------------------------


def test_modality_registry_lists_position_and_per_rock_channels():
    env = RockSampleEnv(K=2, grid_size=3, layout_seed=0)
    registry = env.get_modality_registry()
    names = [m.name for m in registry.modalities]
    assert names == ["position", "rock_0", "rock_1"]


def test_informative_states_for_rock_includes_sense_event_states():
    layout = sample_layout(grid_size=3, K=1, seed=0)
    K = layout.K
    n_events = layout.n_events
    info = informative_states_for_rock(layout, 0)
    # Every state with event = SENSE_0 = 1 should be in the informative set.
    sense_event_0 = 1
    for pos in range(layout.n_cells):
        for mask in range(2 ** K):
            state = encode_state(pos, mask, sense_event_0, K, n_events)
            assert state in info


def test_informative_states_for_rock_excludes_other_events():
    layout = sample_layout(grid_size=3, K=1, seed=0)
    K = layout.K
    n_events = layout.n_events
    info = informative_states_for_rock(layout, 0)
    # Plain OTHER-event states with mask=0 should NOT be informative.
    state = encode_state(0, 0, EVENT_OTHER, K, n_events)
    assert state not in info
