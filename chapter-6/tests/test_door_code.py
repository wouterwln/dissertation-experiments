"""Tests for the door-code POMDP environment.

Coverage:
- Permutation enumeration and parity
- Default partition predicates uniquely identify S_K
- Tensor shapes and normalization (noisy and noiseless)
- State/action layout indices
- Theta persistence across an episode
- Probe channel emits bit consistent with partition_table at PROBE_i
- Noisy probe emits with empirical frequency matching probe_noise
- Outcome channel reports SUCCESS/FAILURE at terminal states
- Opening pi_theta(0) yields SUCCESS; any other door yields FAILURE
- Terminal states are absorbing
- Modality registry has the expected (position, probe, outcome) layout
- Environment protocol conformance
"""

import itertools
import math

import numpy as np
import pytest

from learningpomdp.environments.base import Environment
from learningpomdp.environments.door_code import (
    DoorCodeEnv,
    OUTCOME_FAILURE,
    OUTCOME_NEUTRAL,
    OUTCOME_SUCCESS,
    PROBE_BIT_0,
    PROBE_BIT_1,
    PROBE_NEUTRAL,
    action_indices,
    build_partition_table,
    create_goal,
    create_modality_registry,
    create_observation_tensors,
    create_transition_tensor,
    default_partitions,
    enumerate_permutations,
    n_actions_for,
    n_states_for,
    n_static_for_K,
    partitions_uniquely_identify,
    permutation_parity,
    state_indices,
)


# ---------- Permutations & parity ----------

@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_n_static_is_factorial(K: int) -> None:
    assert n_static_for_K(K) == math.factorial(K)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_enumerate_permutations(K: int) -> None:
    perms = enumerate_permutations(K)
    assert len(perms) == math.factorial(K)
    assert set(perms) == set(itertools.permutations(range(K)))
    assert len(set(perms)) == len(perms)


def test_permutation_parity_known_values() -> None:
    assert permutation_parity((0, 1, 2)) == 0    # identity is even
    assert permutation_parity((0, 2, 1)) == 1    # one transposition
    assert permutation_parity((1, 2, 0)) == 0    # 3-cycle = 2 transpositions
    assert permutation_parity((2, 1, 0)) == 1    # one transposition (swap 0,2)


@pytest.mark.parametrize("K", [3, 4])
def test_permutation_parity_balanced(K: int) -> None:
    """In S_K (K >= 2), exactly half the perms are even."""
    parities = [permutation_parity(p) for p in enumerate_permutations(K)]
    assert sum(parities) == math.factorial(K) // 2


# ---------- Default partitions ----------

@pytest.mark.parametrize("K", [3, 4, 5])
def test_default_partitions_uniquely_identify(K: int) -> None:
    table = build_partition_table(K, default_partitions(K))
    assert partitions_uniquely_identify(table)


@pytest.mark.parametrize("K", [3, 4, 5])
def test_default_partition_balanced(K: int) -> None:
    """Each row of the partition table is balanced or near-balanced."""
    table = build_partition_table(K, default_partitions(K))
    n_perms = math.factorial(K)
    for i, row in enumerate(table):
        ones = int(row.sum())
        assert n_perms // 4 <= ones <= 3 * n_perms // 4, (
            f"partition {i} for K={K} is too unbalanced: "
            f"{ones}/{n_perms} ones"
        )


# ---------- Layout indices ----------

@pytest.mark.parametrize("K,M_expected", [(3, 3), (4, 5), (5, 7)])
def test_default_M_for_K(K: int, M_expected: int) -> None:
    assert len(default_partitions(K)) == M_expected


@pytest.mark.parametrize("M", [3, 5, 7])
def test_state_layout(M: int) -> None:
    layout = state_indices(M)
    assert len(layout) == M + 3
    assert layout["START"] == 0
    assert layout["SUCCESS"] == M + 1
    assert layout["FAILURE"] == M + 2
    for i in range(M):
        assert layout[f"PROBE_{i}"] == 1 + i


@pytest.mark.parametrize("K,M", [(3, 3), (4, 5), (5, 7)])
def test_action_layout(K: int, M: int) -> None:
    actions = action_indices(M, K)
    assert len(actions) == M + K
    for i in range(M):
        assert actions[f"VISIT_PROBE_{i}"] == i
    for i in range(K):
        assert actions[f"OPEN_DOOR_{i}"] == M + i


# ---------- Tensor shapes & normalization ----------

@pytest.mark.parametrize("K", [3, 4])
def test_tensor_shapes(K: int) -> None:
    M = len(default_partitions(K))
    n_states = n_states_for(M)
    n_actions = n_actions_for(M, K)
    n_static = n_static_for_K(K)

    T = create_transition_tensor(K, M)
    assert T.shape == (n_states, n_states, n_static, n_actions)

    table = build_partition_table(K, default_partitions(K))
    B_list = create_observation_tensors(K, table, probe_noise=0.0)
    assert len(B_list) == 3
    B_pos, B_probe, B_outcome = B_list
    assert B_pos.shape == (n_states, n_states)
    assert B_probe.shape == (3, n_states, n_static)
    assert B_outcome.shape == (3, n_states)

    goal = create_goal(K, M)
    assert goal.shape == (n_states, n_static)


@pytest.mark.parametrize("K", [3, 4])
@pytest.mark.parametrize("noise", [0.0, 0.2, 0.5])
def test_T_and_B_normalized(K: int, noise: float) -> None:
    M = len(default_partitions(K))
    T = create_transition_tensor(K, M)
    np.testing.assert_allclose(T.sum(axis=0), 1.0, atol=1e-8)

    table = build_partition_table(K, default_partitions(K))
    B_pos, B_probe, B_outcome = create_observation_tensors(K, table, probe_noise=noise)
    np.testing.assert_allclose(B_pos.sum(axis=0), 1.0, atol=1e-8)
    np.testing.assert_allclose(B_probe.sum(axis=0), 1.0, atol=1e-8)
    np.testing.assert_allclose(B_outcome.sum(axis=0), 1.0, atol=1e-8)


def test_probe_noise_value_validated() -> None:
    K = 3
    table = build_partition_table(K, default_partitions(K))
    with pytest.raises(ValueError):
        create_observation_tensors(K, table, probe_noise=-0.1)
    with pytest.raises(ValueError):
        create_observation_tensors(K, table, probe_noise=0.6)


@pytest.mark.parametrize("K", [3, 4])
def test_goal_prefers_success(K: int) -> None:
    M = len(default_partitions(K))
    goal = create_goal(K, M)
    layout = state_indices(M)
    SUCCESS = layout["SUCCESS"]
    FAILURE = layout["FAILURE"]
    for theta_idx in range(n_static_for_K(K)):
        assert goal[SUCCESS, theta_idx] > goal[FAILURE, theta_idx]


# ---------- Env construction & rollout ----------

def test_default_K4_construction() -> None:
    env = DoorCodeEnv(K=4)
    assert env.K == 4
    assert env.M == 5
    assert env.n_states == 5 + 3
    assert env.n_actions == 5 + 4
    assert env.n_static == 24
    assert env.partitions_uniquely_identify_theta


def test_K_too_small_rejected() -> None:
    with pytest.raises(ValueError):
        DoorCodeEnv(K=1)


def test_invalid_theta_rejected() -> None:
    env = DoorCodeEnv(K=3)
    with pytest.raises(ValueError):
        env.reset(theta=-1)
    with pytest.raises(ValueError):
        env.reset(theta=env.n_static)


def test_invalid_action_rejected() -> None:
    env = DoorCodeEnv(K=3)
    env.reset(seed=0, theta=0)
    with pytest.raises(ValueError):
        env.step(-1)
    with pytest.raises(ValueError):
        env.step(env.n_actions)


def test_reset_starts_at_start_state() -> None:
    env = DoorCodeEnv(K=3)
    result = env.reset(seed=0, theta=0)
    assert env.agent_state == state_indices(env.M)["START"]
    assert not result.terminated
    assert result.reward == 0.0


def test_reset_pins_theta() -> None:
    env = DoorCodeEnv(K=3)
    for theta_idx in range(env.n_static):
        env.reset(seed=0, theta=theta_idx)
        assert env.theta == theta_idx
        assert env.theta_permutation == enumerate_permutations(3)[theta_idx]


def test_visit_probe_is_theta_independent() -> None:
    env = DoorCodeEnv(K=3)
    layout = state_indices(env.M)
    for theta_idx in range(env.n_static):
        for probe_idx in range(env.M):
            env.reset(seed=0, theta=theta_idx)
            env.step(probe_idx)
            assert env.agent_state == layout[f"PROBE_{probe_idx}"]


def test_theta_persists_across_steps() -> None:
    env = DoorCodeEnv(K=3)
    env.reset(seed=42, theta=3)
    initial_theta = env.theta
    initial_perm = env.theta_permutation
    for _ in range(5):
        env.step(0)
        assert env.theta == initial_theta
        assert env.theta_permutation == initial_perm


# ---------- Probe channel ----------

def test_deterministic_probe_emits_partition_bit() -> None:
    """With probe_noise=0, PROBE_i emits exactly partition_table[i, theta]."""
    K = 3
    env = DoorCodeEnv(K=K, probe_noise=0.0, obs_seed=0)
    table = env.get_partition_table()
    for theta_idx in range(env.n_static):
        for probe_idx in range(env.M):
            env.reset(seed=0, theta=theta_idx)
            result = env.step(probe_idx)
            probe_obs = result.obs_per_channel["probe"]
            expected_bit = int(table[probe_idx, theta_idx])
            assert int(probe_obs.argmax()) == expected_bit, (
                f"K={K}, theta={theta_idx}, probe={probe_idx}: expected "
                f"bit {expected_bit}, got {int(probe_obs.argmax())}"
            )


def test_noisy_probe_emission_frequency() -> None:
    """Empirical bit-flip rate matches probe_noise within tolerance."""
    K = 3
    noise = 0.3
    env = DoorCodeEnv(K=K, probe_noise=noise, obs_seed=42)
    table = env.get_partition_table()
    n_trials = 2000

    flips = 0
    for trial in range(n_trials):
        env.reset(theta=0)  # fixed theta; RNG continues across resets
        result = env.step(0)  # PROBE_0
        emitted = int(result.obs_per_channel["probe"].argmax())
        true_bit = int(table[0, 0])
        if emitted != true_bit:
            flips += 1
    empirical = flips / n_trials
    # 99% CI for binomial: noise * (1-noise) / n is the variance; 3 sigma
    sigma = math.sqrt(noise * (1 - noise) / n_trials)
    assert abs(empirical - noise) < 4 * sigma, (
        f"empirical noise rate {empirical} differs too much from "
        f"target {noise} (sigma={sigma:.4f})"
    )


def test_probe_neutral_at_start() -> None:
    env = DoorCodeEnv(K=3)
    result = env.reset(seed=0, theta=0)
    assert int(result.obs_per_channel["probe"].argmax()) == PROBE_NEUTRAL


def test_probe_neutral_after_terminal() -> None:
    env = DoorCodeEnv(K=3)
    env.reset(seed=0, theta=0)
    correct = env.correct_door
    result = env.step(env.M + correct)
    assert int(result.obs_per_channel["probe"].argmax()) == PROBE_NEUTRAL


# ---------- Outcome channel ----------

def test_outcome_neutral_at_non_terminal() -> None:
    env = DoorCodeEnv(K=3)
    result = env.reset(seed=0, theta=0)
    assert int(result.obs_per_channel["outcome"].argmax()) == OUTCOME_NEUTRAL
    result = env.step(0)
    assert int(result.obs_per_channel["outcome"].argmax()) == OUTCOME_NEUTRAL


def test_correct_door_yields_success_for_all_theta() -> None:
    K = 3
    env = DoorCodeEnv(K=K)
    layout = state_indices(env.M)
    for theta_idx in range(env.n_static):
        env.reset(seed=0, theta=theta_idx)
        correct = env.correct_door
        for door in range(K):
            env.reset(seed=0, theta=theta_idx)
            result = env.step(env.M + door)
            assert result.terminated
            if door == correct:
                assert result.reward == 1.0
                assert env.agent_state == layout["SUCCESS"]
                assert int(result.obs_per_channel["outcome"].argmax()) == OUTCOME_SUCCESS
            else:
                assert result.reward == -1.0
                assert env.agent_state == layout["FAILURE"]
                assert int(result.obs_per_channel["outcome"].argmax()) == OUTCOME_FAILURE


def test_door_action_works_from_probe_state_too() -> None:
    env = DoorCodeEnv(K=3)
    env.reset(seed=0, theta=0)
    env.step(0)  # VISIT_PROBE_0
    correct = env.correct_door
    result = env.step(env.M + correct)
    assert result.terminated
    assert result.reward == 1.0


# ---------- Terminal absorbing ----------

def test_terminal_states_absorbing() -> None:
    env = DoorCodeEnv(K=3)
    env.reset(seed=0, theta=0)
    correct = env.correct_door
    env.step(env.M + correct)
    success_state = env.agent_state
    for a in range(env.n_actions):
        env.step(a)
        assert env.agent_state == success_state


# ---------- Modality registry ----------

def test_modality_registry_structure() -> None:
    K = 3
    env = DoorCodeEnv(K=K)
    registry = env.get_modality_registry()
    assert registry.n_channels == 3
    assert [m.name for m in registry.modalities] == ["position", "probe", "outcome"]
    assert registry.theta_dependent_mask == [False, True, False]
    assert registry.channel_sizes == [env.n_states, 3, 3]
    probe_modality = registry.get_modality_by_name("probe")
    expected_probe_states = frozenset(
        state_indices(env.M)[f"PROBE_{i}"] for i in range(env.M)
    )
    assert probe_modality.informative_states == expected_probe_states


def test_observation_vector_layout_matches_registry() -> None:
    env = DoorCodeEnv(K=3)
    registry = env.get_modality_registry()
    result = env.reset(seed=0, theta=0)
    assert result.obs.shape == (registry.total_obs_dim,)
    expected = np.concatenate([
        result.obs_per_channel["position"],
        result.obs_per_channel["probe"],
        result.obs_per_channel["outcome"],
    ])
    np.testing.assert_array_equal(result.obs, expected)


# ---------- Protocol & attributes ----------

def test_implements_environment_protocol() -> None:
    env = DoorCodeEnv(K=3)
    assert isinstance(env, Environment)


def test_attributes_match_K_default_M() -> None:
    env = DoorCodeEnv(K=4)
    assert env.K == 4
    assert env.M == 5
    assert env.n_states == 8
    assert env.n_actions == 9
    assert env.n_static == 24


def test_constants_referenced() -> None:
    """Smoke test that the module constants resolve."""
    assert PROBE_BIT_0 == 0
    assert PROBE_BIT_1 == 1
    assert PROBE_NEUTRAL == 2


# ---------- Multi-step locks (lock_length > 1) ----------

def test_lock_length_validated() -> None:
    with pytest.raises(ValueError):
        DoorCodeEnv(K=4, lock_length=0)
    with pytest.raises(ValueError):
        DoorCodeEnv(K=4, lock_length=5)


def test_lock_length_2_state_layout() -> None:
    env = DoorCodeEnv(K=4, lock_length=2)
    # n_states = L + M + 2 = 2 + 5 + 2 = 9
    assert env.n_states == 9
    layout = state_indices(env.M, env.lock_length)
    assert layout["START"] == 0
    assert layout["STAGE_1"] == 1
    assert layout["PROBE_0"] == 2
    assert layout["SUCCESS"] == 7
    assert layout["FAILURE"] == 8


def test_lock_length_2_correct_sequence_yields_success() -> None:
    """Opening pi(0) then pi(1) reaches SUCCESS."""
    K = 4
    env = DoorCodeEnv(K=K, lock_length=2, probe_noise=0.0)
    layout = state_indices(env.M, env.L)
    for theta_idx in range(env.n_static):
        env.reset(seed=0, theta=theta_idx)
        perm = env.theta_permutation
        # Open pi(0) -> STAGE_1
        result1 = env.step(env.M + perm[0])
        assert env.agent_state == layout["STAGE_1"]
        assert not result1.terminated
        # Open pi(1) -> SUCCESS
        result2 = env.step(env.M + perm[1])
        assert env.agent_state == layout["SUCCESS"]
        assert result2.terminated
        assert result2.reward == 1.0


def test_lock_length_2_wrong_first_door_yields_failure() -> None:
    K = 4
    env = DoorCodeEnv(K=K, lock_length=2, probe_noise=0.0)
    layout = state_indices(env.M, env.L)
    for theta_idx in range(env.n_static):
        env.reset(seed=0, theta=theta_idx)
        perm = env.theta_permutation
        wrong_door = (perm[0] + 1) % K
        result = env.step(env.M + wrong_door)
        assert env.agent_state == layout["FAILURE"]
        assert result.terminated
        assert result.reward == -1.0


def test_lock_length_2_wrong_second_door_yields_failure() -> None:
    K = 4
    env = DoorCodeEnv(K=K, lock_length=2, probe_noise=0.0)
    layout = state_indices(env.M, env.L)
    for theta_idx in range(env.n_static):
        env.reset(seed=0, theta=theta_idx)
        perm = env.theta_permutation
        # First door correct -> STAGE_1
        env.step(env.M + perm[0])
        # Second door wrong -> FAILURE
        wrong_door = perm[2] if perm[2] != perm[1] else perm[3]
        result = env.step(env.M + wrong_door)
        assert env.agent_state == layout["FAILURE"]
        assert result.terminated
        assert result.reward == -1.0


def test_probing_during_commit_phase_is_failure() -> None:
    """Once the agent has opened the first door (STAGE_1), VISIT_PROBE
    leads to FAILURE — probing during commit phase = death."""
    K = 4
    env = DoorCodeEnv(K=K, lock_length=2, probe_noise=0.0)
    layout = state_indices(env.M, env.L)
    env.reset(seed=0, theta=0)
    perm = env.theta_permutation
    env.step(env.M + perm[0])  # advance to STAGE_1
    assert env.agent_state == layout["STAGE_1"]
    result = env.step(0)  # VISIT_PROBE_0 from STAGE_1
    assert env.agent_state == layout["FAILURE"]
    assert result.terminated
    assert result.reward == -1.0


def test_can_probe_then_commit_correct_sequence() -> None:
    """Probe before committing to the L=2 sequence works end-to-end."""
    K = 4
    env = DoorCodeEnv(K=K, lock_length=2, probe_noise=0.0)
    env.reset(seed=0, theta=0)
    perm = env.theta_permutation
    env.step(0)  # VISIT_PROBE_0
    env.step(env.M + perm[0])  # OPEN_DOOR_pi(0) -> STAGE_1
    result = env.step(env.M + perm[1])  # OPEN_DOOR_pi(1) -> SUCCESS
    assert result.reward == 1.0


def test_lock_length_K_full_permutation() -> None:
    """L=K requires opening the entire permutation in order."""
    K = 4
    env = DoorCodeEnv(K=K, lock_length=K, probe_noise=0.0)
    env.reset(seed=0, theta=0)
    perm = env.theta_permutation
    for k, door in enumerate(perm):
        result = env.step(env.M + door)
        if k < K - 1:
            assert not result.terminated, f"premature termination at stage {k}"
    assert result.terminated
    assert result.reward == 1.0


def test_custom_lock_doors_fn() -> None:
    """Pass a custom 2-door sequence (e.g., reverse order)."""
    K = 4
    rev_two = lambda perm: (perm[1], perm[0])
    env = DoorCodeEnv(K=K, lock_length=2, lock_doors_fn=rev_two)
    env.reset(seed=0, theta=0)
    perm = env.theta_permutation
    # Must open pi(1) FIRST
    env.step(env.M + perm[1])
    result = env.step(env.M + perm[0])
    assert result.reward == 1.0


def test_lock_length_2_T_normalized() -> None:
    K = 4
    L = 2
    M = 5
    T = create_transition_tensor(K, M, L=L)
    np.testing.assert_allclose(T.sum(axis=0), 1.0, atol=1e-8)
