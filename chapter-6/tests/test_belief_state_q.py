"""Tests for BeliefStateQLearner — the tabular Bayes-adaptive baseline.

Covers:
- Construction sets correct table dimensions for K=3, 4.
- Belief feature: MAP pi(0) and entropy bin computed correctly on
  hand-crafted q(theta) distributions.
- plan() returns a one-hot action distribution; epsilon=0 matches Q
  argmax, epsilon=1 produces uniform-random over many calls.
- post_step_update applies the correct TD target.
- Runner integration: run_block drives the learner through full
  episodes, Q-table grows non-zero entries.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_block, run_outer_loop
from learningpomdp.planning.belief_state_q import (
    N_ENTROPY_BINS,
    BeliefStateQLearner,
)


def test_construction_K3() -> None:
    learner = BeliefStateQLearner(K=3, M=3)
    assert learner.K == 3
    assert learner.M == 3
    assert learner.n_states == 6
    assert learner.n_actions == 6
    assert learner.n_static == 6
    # Q shape: (n_states, K, N_ENTROPY_BINS, n_actions)
    assert learner.Q.shape == (6, 3, N_ENTROPY_BINS, 6)


def test_construction_K4() -> None:
    learner = BeliefStateQLearner(K=4, M=5)
    assert learner.n_states == 8
    assert learner.n_actions == 9
    assert learner.n_static == 24
    assert learner.Q.shape == (8, 4, N_ENTROPY_BINS, 9)


# ---------- belief feature ----------

def test_belief_feature_sharp_q_static() -> None:
    learner = BeliefStateQLearner(K=3, M=3)
    # theta_idx=0 corresponds to perm (0,1,2) -> pi(0) = 0.
    q_static = np.zeros(6, dtype=np.float64)
    q_static[0] = 1.0
    feature = learner._belief_feature(x_idx=0, q_static=q_static)
    assert feature == (0, 0, 0)  # state 0, MAP pi(0)=0, entropy bin 0 (low)


def test_belief_feature_uniform_q_static() -> None:
    learner = BeliefStateQLearner(K=3, M=3)
    q_static = np.ones(6, dtype=np.float64) / 6
    feature = learner._belief_feature(x_idx=2, q_static=q_static)
    state, map_pi0, ent_bin = feature
    assert state == 2
    # All pi(0) values equally likely; argmax breaks ties at 0.
    assert map_pi0 == 0
    # Uniform entropy = log(6) — at the maximum.
    assert ent_bin == 2  # high


def test_belief_feature_two_perm_concentration() -> None:
    """q(theta) on two perms with the same pi(0)=2 should yield map_pi0=2."""
    learner = BeliefStateQLearner(K=3, M=3)
    # Perms with pi(0)=2 in S_3: indices 4 (2,0,1) and 5 (2,1,0).
    q_static = np.zeros(6, dtype=np.float64)
    q_static[4] = q_static[5] = 0.5
    feature = learner._belief_feature(x_idx=0, q_static=q_static)
    assert feature[1] == 2  # MAP pi(0) = 2
    # H = log(2) ~ 0.69; max H = log(6) ~ 1.79 -> frac ~ 0.39 -> bin 1
    assert feature[2] == 1


# ---------- plan() ----------

def test_plan_epsilon_zero_picks_argmax_q() -> None:
    learner = BeliefStateQLearner(K=3, M=3, epsilon=0.0)
    # Set Q so action 4 is best for the (0, 0, 2) feature (uniform belief).
    learner.Q[0, 0, 2, 4] = 1.0
    q_state = jnp.zeros(6, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(6, dtype=jnp.float32) / 6
    action_dist = np.asarray(learner.plan(
        q_state, q_static, None, [], None, horizon=3, theta_mask=[],
    ))
    assert int(action_dist.argmax()) == 4


def test_plan_epsilon_one_explores_uniformly() -> None:
    learner = BeliefStateQLearner(K=3, M=3, epsilon=1.0, rng_seed=0)
    q_state = jnp.zeros(6, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(6, dtype=jnp.float32) / 6
    actions = []
    for _ in range(600):
        actions.append(int(np.asarray(learner.plan(
            q_state, q_static, None, [], None, horizon=3, theta_mask=[],
        )).argmax()))
    # Every action should appear; counts roughly even.
    counts = np.bincount(actions, minlength=6)
    assert (counts > 0).all()
    np.testing.assert_allclose(counts / 600, 1.0 / 6, atol=0.06)


# ---------- post_step_update ----------

def test_td_update_applies_to_pending_step() -> None:
    learner = BeliefStateQLearner(K=3, M=3, lr=0.5, gamma=0.99, epsilon=0.0)
    q_state = jnp.zeros(6, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(6, dtype=jnp.float32) / 6
    learner.plan(q_state, q_static, None, [], None, horizon=3, theta_mask=[])
    # plan() set _pending = (0, 0, 2, 0) (argmax of zero Q is action 0).
    assert learner._pending == (0, 0, 2, 0)

    # Apply TD update: r=1, terminated=True.
    q_state_new = jnp.zeros(6, dtype=jnp.float32).at[4].set(1.0)
    q_static_new = jnp.zeros(6, dtype=jnp.float32).at[0].set(1.0)
    learner.post_step_update(q_state_new, q_static_new, reward=1.0, terminated=True)

    # Q[(0,0,2,0)] += 0.5 * (1.0 - 0.0) = 0.5
    assert learner.Q[0, 0, 2, 0] == pytest.approx(0.5)
    # Pending cleared.
    assert learner._pending is None


def test_post_step_update_no_pending_is_noop() -> None:
    learner = BeliefStateQLearner(K=3, M=3)
    learner.post_step_update(
        q_state_new=jnp.zeros(6, dtype=jnp.float32),
        q_static_new=jnp.zeros(6, dtype=jnp.float32),
        reward=1.0, terminated=False,
    )
    np.testing.assert_array_equal(learner.Q, np.zeros_like(learner.Q))


def test_prepare_episode_clears_pending() -> None:
    learner = BeliefStateQLearner(K=3, M=3, epsilon=0.0)
    q_state = jnp.zeros(6, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(6, dtype=jnp.float32) / 6
    learner.plan(q_state, q_static, None, [], None, horizon=3, theta_mask=[])
    assert learner._pending is not None
    learner.prepare_episode(q_static, np.random.default_rng(0))
    assert learner._pending is None


# ---------- runner integration ----------

def test_run_block_drives_q_learner() -> None:
    """Running a block populates the Q table (some entries become non-zero)."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    learner = BeliefStateQLearner(K=3, M=env.M, epsilon=0.5, rng_seed=0)
    initial_q = learner.Q.copy()
    block = run_block(
        env, learner, theta_true=0, n_episodes=20,
        horizon=3, max_steps=3, deterministic=True, seed=0,
    )
    # Q-table moved.
    assert not np.allclose(learner.Q, initial_q)
    # Block ran without crashing.
    assert len(block.episodes) == 20


def test_run_outer_loop_with_q_learner() -> None:
    """Q-learning slots into run_outer_loop just like AIF/VBP/PSRL."""
    learner = BeliefStateQLearner(K=3, M=3, epsilon=0.3, rng_seed=0)
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=learner,
        n_seeds=4, n_episodes=10,
        horizon=3, max_steps=3, deterministic=True, seed=0,
    )
    assert result.success.shape == (4, 10)


def test_q_table_shape_grows_with_lock_length() -> None:
    """Q-table dimensions track L: one MAP-pi(stage) axis per cell."""
    L1 = BeliefStateQLearner(K=4, M=5, L=1)
    L2 = BeliefStateQLearner(K=4, M=5, L=2)
    L3 = BeliefStateQLearner(K=4, M=5, L=3)
    # n_states = L + M + 2; Q shape = (n_states, K^L, N_ENTROPY_BINS, n_actions)
    assert L1.Q.shape == (8, 4, 3, 9)         # 864 entries
    assert L2.Q.shape == (9, 4, 4, 3, 9)      # 3888 entries
    assert L3.Q.shape == (10, 4, 4, 4, 3, 9)  # 17280 entries


def test_belief_feature_l2_tracks_both_cells() -> None:
    """At L=2, the belief feature contains MAP pi(0) AND MAP pi(1)."""
    learner = BeliefStateQLearner(K=3, M=3, L=2)
    # theta_idx=0 -> perm (0,1,2) so pi(0)=0, pi(1)=1.
    q_static = np.zeros(6, dtype=np.float64)
    q_static[0] = 1.0
    feature = learner._belief_feature(x_idx=0, q_static=q_static)
    assert len(feature) == 4  # (state, map_pi0, map_pi1, ent_bin)
    state, map_pi0, map_pi1, ent_bin = feature
    assert state == 0
    assert map_pi0 == 0
    assert map_pi1 == 1
    assert ent_bin == 0  # sharp


def test_plan_samples_x_idx_under_multimodal_q_state() -> None:
    """plan() must sample x_idx ~ q_state, not argmax. With a uniform
    q_state over two states, pending features across many calls must
    include both state indices — otherwise argmax silently picks one
    aliased mode (the bug under many-to-one B_position in minigrid).
    """
    K = 3
    M = 3
    learner = BeliefStateQLearner(K=K, M=M, epsilon=0.0, rng_seed=0)
    n_states = learner.n_states
    # Uniform belief over states 0 and 1.
    q_state = jnp.zeros(n_states, dtype=jnp.float32)
    q_state = q_state.at[0].set(0.5).at[1].set(0.5)
    q_static = jnp.ones(learner.n_static, dtype=jnp.float32) / learner.n_static

    seen_state_indices: set[int] = set()
    for _ in range(200):
        learner.plan(
            q_state, q_static,
            T=None, B_list=None, goal=None, horizon=1,
        )
        # _pending = (x_idx, map_pi0, ..., ent_bin, action).
        seen_state_indices.add(learner._pending[0])

    assert seen_state_indices == {0, 1}, (
        f"plan() didn't sample both modes of q_state; saw {seen_state_indices}"
    )


def test_l2_runner_drives_q_table_population() -> None:
    """At L=2, the Q-learner runs through run_block without crashing
    and updates Q-table entries."""
    from learningpomdp.environments.door_code import DoorCodeEnv
    from learningpomdp.experiment.door_code_runner import run_block
    env = DoorCodeEnv(K=4, lock_length=2, probe_noise=0.0)
    learner = BeliefStateQLearner(K=4, M=env.M, L=2, epsilon=0.5, rng_seed=0)
    initial_q = learner.Q.copy()
    block = run_block(
        env, learner, theta_true=0, n_episodes=20,
        horizon=4, max_steps=4, deterministic=True, seed=0,
    )
    assert not np.allclose(learner.Q, initial_q)
    assert len(block.episodes) == 20
