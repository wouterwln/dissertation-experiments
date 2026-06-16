"""Tests for the Factored Tiger BA-POMDP env."""

from __future__ import annotations

import numpy as np
import pytest

from learningpomdp.environments.base import Environment
from learningpomdp.environments.factored_tiger import (
    DOOR_NO_INFO,
    EVENT_OTHER,
    FactoredTigerEnv,
    N_DOOR_OBS,
    action_role,
    decode_state,
    encode_state,
    informative_states_for_door,
    n_actions_for,
    n_events_for,
    n_states_for,
)
from learningpomdp.factors import FactorSpec, ThetaSpec


# --- Encoding ---------------------------------------------------------------


def test_encode_decode_round_trip():
    K = 3
    n_events = n_events_for(K)
    for state in range(n_states_for(K)):
        mask, event = decode_state(state, K)
        assert encode_state(mask, event, K) == state


def test_n_actions_for_K():
    assert n_actions_for(2) == 4
    assert n_actions_for(4) == 8


def test_action_role():
    K = 3
    assert action_role(0, K) == ("listen", 0)
    assert action_role(2, K) == ("listen", 2)
    assert action_role(3, K) == ("open", 0)
    assert action_role(5, K) == ("open", 2)


def test_action_role_rejects_out_of_range():
    with pytest.raises(ValueError):
        action_role(6, K=3)
    with pytest.raises(ValueError):
        action_role(-1, K=3)


# --- Env protocol -----------------------------------------------------------


def test_success_state_index_is_sentinel():
    """Factored Tiger has no terminal SUCCESS state; the index returns
    -1 so any ``agent_state == success_state_index`` check is always
    False."""
    env = FactoredTigerEnv(K=3)
    assert env.success_state_index == -1
    # Must not collide with any reachable state.
    assert env.success_state_index != env.start_state_index
    assert env.agent_state != env.success_state_index


def test_env_satisfies_environment_protocol():
    env = FactoredTigerEnv(K=3)
    assert isinstance(env, Environment)


def test_env_dimensions_at_K3():
    env = FactoredTigerEnv(K=3)
    # n_states = 2^3 * (2*3+1) = 8 * 7 = 56.
    assert env.n_states == 56
    # n_actions = 2*K = 6.
    assert env.n_actions == 6
    # n_static = 2^K = 8.
    assert env.n_static == 8


def test_theta_spec_is_purely_persistent():
    env = FactoredTigerEnv(K=4)
    assert env.theta_spec.component_sizes == (2, 2, 2, 2)
    assert env.theta_spec.episode_reset_components == ()
    assert env.theta_spec.persistent_components == (0, 1, 2, 3)


def test_factor_specs():
    env = FactoredTigerEnv(K=3)
    assert env.t_factor_spec == FactorSpec(theta_components=())
    # B_position is θ-independent; each B_door_r depends only on θ_r.
    assert env.b_factor_specs[0] == FactorSpec(theta_components=())
    assert env.b_factor_specs[1] == FactorSpec(theta_components=(0,))
    assert env.b_factor_specs[2] == FactorSpec(theta_components=(1,))
    assert env.b_factor_specs[3] == FactorSpec(theta_components=(2,))


def test_env_rejects_K_below_2():
    with pytest.raises(ValueError):
        FactoredTigerEnv(K=1)


# --- Tensor shapes ----------------------------------------------------------


def test_T_shape():
    env = FactoredTigerEnv(K=3)
    T = env.get_transition_tensor()
    assert T.shape == (env.n_states, env.n_states, env.n_actions)


def test_T_is_proper_distribution():
    env = FactoredTigerEnv(K=3)
    T = env.get_transition_tensor()
    sums = T.sum(axis=0)
    assert np.allclose(sums, 1.0)


def test_B_shapes():
    env = FactoredTigerEnv(K=3)
    B_list = env.get_observation_tensors()
    # B_position + K door channels.
    assert len(B_list) == 1 + env.K
    n_events = n_events_for(env.K)
    assert B_list[0].shape == (n_events, env.n_states)
    for r in range(env.K):
        assert B_list[1 + r].shape == (N_DOOR_OBS, env.n_states, 2)


def test_B_position_is_identity_on_event():
    env = FactoredTigerEnv(K=3)
    B_position = env.get_observation_tensors()[0]
    assert np.allclose(B_position.sum(axis=0), 1.0)
    K = env.K
    for state in range(env.n_states):
        _, event = decode_state(state, K)
        assert B_position[event, state] == 1.0


def test_B_door_normalises_over_emissions():
    env = FactoredTigerEnv(K=3)
    B_list = env.get_observation_tensors()
    for r in range(env.K):
        s = B_list[1 + r].sum(axis=0)
        assert np.allclose(s, 1.0)


def test_B_door_no_info_default():
    """At state with event=OTHER, every door channel emits no_info
    deterministically regardless of θ."""
    env = FactoredTigerEnv(K=3)
    B_list = env.get_observation_tensors()
    K = env.K
    state = encode_state(mask=0, event=EVENT_OTHER, K=K)
    for r in range(K):
        for theta_k in (0, 1):
            p = B_list[1 + r][:, state, theta_k]
            assert p[DOOR_NO_INFO] == 1.0


def test_B_door_listen_emits_correct_bit_at_high_accuracy():
    """LISTEN_r at high listen_accuracy concentrates emission near θ_r."""
    env = FactoredTigerEnv(K=2, listen_accuracy=0.9)
    K = env.K
    listen_event_r = 1  # LISTEN_0
    state = encode_state(mask=0, event=listen_event_r, K=K)
    B_door_0 = env.get_observation_tensors()[1]
    for theta_k in (0, 1):
        p = B_door_0[:, state, theta_k]
        assert p[theta_k] == pytest.approx(0.9, abs=1e-6)
        assert p[1 - theta_k] == pytest.approx(0.1, abs=1e-6)
        assert p[DOOR_NO_INFO] == 0.0


def test_B_door_open_reveal_is_deterministic():
    """At a state representing 'just opened door r' (event=OPEN_r,
    mask[r]=1), the door channel emits θ_r deterministically."""
    env = FactoredTigerEnv(K=3)
    K = env.K
    B_list = env.get_observation_tensors()
    for r in range(K):
        open_event_r = 1 + K + r
        state = encode_state(mask=(1 << r), event=open_event_r, K=K)
        for theta_k in (0, 1):
            p = B_list[1 + r][:, state, theta_k]
            assert p[theta_k] == pytest.approx(1.0)
            assert p[1 - theta_k] == pytest.approx(0.0)


def test_B_door_silent_for_other_doors_listen():
    """LISTEN_0 should not inform door_1's channel — door_1 emits no_info."""
    env = FactoredTigerEnv(K=2)
    K = env.K
    listen_event_0 = 1  # LISTEN_0
    state = encode_state(mask=0, event=listen_event_0, K=K)
    B_door_1 = env.get_observation_tensors()[2]
    for theta_k in (0, 1):
        p = B_door_1[:, state, theta_k]
        assert p[DOOR_NO_INFO] == 1.0


# --- T transition correctness -----------------------------------------------


def test_T_listen_does_not_change_mask():
    env = FactoredTigerEnv(K=3)
    K = env.K
    T = env.get_transition_tensor()
    state = encode_state(mask=0b101, event=EVENT_OTHER, K=K)
    # LISTEN_1 = action 1 -> next event = LISTEN_1 = 2; mask unchanged.
    listen_action = 1
    listen_event = 1 + 1
    expected = encode_state(mask=0b101, event=listen_event, K=K)
    assert T[:, state, listen_action].argmax() == expected


def test_T_open_unopened_flips_mask():
    env = FactoredTigerEnv(K=3)
    K = env.K
    T = env.get_transition_tensor()
    state = encode_state(mask=0, event=EVENT_OTHER, K=K)
    # OPEN_2 = action K + 2 = 5; opens door 2 -> mask=0b100, event=OPEN_2.
    open_action = K + 2
    open_event = 1 + K + 2
    expected = encode_state(mask=0b100, event=open_event, K=K)
    assert T[:, state, open_action].argmax() == expected


def test_T_open_already_opened_is_no_op():
    env = FactoredTigerEnv(K=3)
    K = env.K
    T = env.get_transition_tensor()
    state = encode_state(mask=0b100, event=EVENT_OTHER, K=K)
    open_action = K + 2  # re-OPEN of door 2 (already mask[2]=1)
    expected = encode_state(mask=0b100, event=EVENT_OTHER, K=K)
    assert T[:, state, open_action].argmax() == expected


# --- Reset / step semantics -------------------------------------------------


def test_reset_returns_to_start_state():
    env = FactoredTigerEnv(K=3)
    env.reset(theta=0)
    assert env.agent_state == env.start_state_index
    mask, event = decode_state(env.agent_state, env.K)
    assert mask == 0
    assert event == EVENT_OTHER


def test_reset_with_explicit_theta_records_components():
    env = FactoredTigerEnv(K=3)
    # component_sizes=(2,2,2); theta=5 -> unravel -> (1, 0, 1)
    env.reset(theta=5)
    assert env.theta == 5
    assert env.theta_components == (1, 0, 1)


def test_step_reward_at_safe_door_open():
    """OPEN on door k with θ_k=1 yields +open_safe_reward."""
    env = FactoredTigerEnv(
        K=2, open_safe_reward=10.0, open_tiger_penalty=100.0,
    )
    # theta=3 = (1,1) -> both safe.
    env.reset(theta=3)
    open_door_0 = env.K + 0
    step_result = env.step(open_door_0)
    assert step_result.reward == pytest.approx(10.0)


def test_step_reward_at_tiger_door_open():
    """OPEN on door k with θ_k=0 yields -open_tiger_penalty."""
    env = FactoredTigerEnv(
        K=2, open_safe_reward=10.0, open_tiger_penalty=100.0,
    )
    # theta=0 = (0,0) -> both tiger.
    env.reset(theta=0)
    open_door_0 = env.K + 0
    step_result = env.step(open_door_0)
    assert step_result.reward == pytest.approx(-100.0)


def test_step_reward_listen_costs_listen_cost():
    env = FactoredTigerEnv(K=2, listen_cost=1.0)
    env.reset(theta=0)
    listen_action = 0  # LISTEN_0
    step_result = env.step(listen_action)
    assert step_result.reward == pytest.approx(-1.0)


def test_step_reward_listen_zero_by_default():
    """Default listen_cost=0 so AIF/VBP's state-only goal prior cleanly
    optimizes the reward structure (no per-action LISTEN cost)."""
    env = FactoredTigerEnv(K=2)  # default listen_cost=0
    env.reset(theta=0)
    step_result = env.step(0)
    assert step_result.reward == 0.0


def test_step_reward_zero_on_already_opened_door():
    env = FactoredTigerEnv(K=2, open_safe_reward=10.0)
    env.reset(theta=3)
    open_door_0 = env.K + 0
    first = env.step(open_door_0)
    assert first.reward == pytest.approx(10.0)
    # Re-OPEN of door 0: no-op, zero reward.
    second = env.step(open_door_0)
    assert second.reward == 0.0


# --- Modality registry ------------------------------------------------------


def test_modality_registry_lists_position_and_per_door_channels():
    env = FactoredTigerEnv(K=3)
    registry = env.get_modality_registry()
    names = [m.name for m in registry.modalities]
    # "position" (last_event channel, θ-independent) + per-door channels.
    assert names == ["position", "door_0", "door_1", "door_2"]


def test_informative_states_for_door_includes_listen_event_states():
    K = 3
    info = informative_states_for_door(K, 0)
    listen_event_0 = 1
    # Every state with event = LISTEN_0 is informative (any mask).
    for mask in range(2 ** K):
        state = encode_state(mask, listen_event_0, K)
        assert state in info


def test_informative_states_for_door_excludes_other_events():
    K = 3
    info = informative_states_for_door(K, 0)
    state = encode_state(mask=0, event=EVENT_OTHER, K=K)
    assert state not in info


def test_informative_states_for_door_includes_open_reveal_only_with_mask_set():
    K = 3
    info = informative_states_for_door(K, 1)
    open_event_1 = 1 + K + 1
    # Door 1 just opened (mask bit 1 set, event=OPEN_1) -> informative.
    state_opened = encode_state(mask=0b010, event=open_event_1, K=K)
    assert state_opened in info
    # OPEN_1 event without mask bit 1 set should never appear under T,
    # but the informative-states predicate filters by mask anyway.
    state_no_mask = encode_state(mask=0, event=open_event_1, K=K)
    assert state_no_mask not in info


# --- Goal prior -------------------------------------------------------------


def test_goal_is_normalized_distribution_per_theta():
    env = FactoredTigerEnv(K=3)
    goal = env.get_goal()
    assert goal.shape == (env.n_states, env.n_static)
    assert np.allclose(goal.sum(axis=0), 1.0)


def test_goal_prefers_safe_doors_opened():
    """Under θ with all-safe (e.g. all bits = 1), the state with all
    doors opened should have higher preference than the start state."""
    env = FactoredTigerEnv(K=3, good_logit=2.0, bad_logit=4.0)
    K = env.K
    # theta_all_safe = (1,1,1) -> ravel = 7.
    theta_all_safe = 7
    start_state = encode_state(mask=0, event=EVENT_OTHER, K=K)
    all_opened_state = encode_state(mask=0b111, event=EVENT_OTHER, K=K)
    goal = env.get_goal()
    assert goal[all_opened_state, theta_all_safe] > goal[start_state, theta_all_safe]
