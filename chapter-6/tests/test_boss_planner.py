"""Tests for BOSSPlanner (merged-POMDP MCTS)."""

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_block, run_outer_loop
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.boss_planner import BOSSPlanner


def test_n_samples_validated() -> None:
    with pytest.raises(ValueError):
        BOSSPlanner(n_samples=0)


def test_prepare_episode_samples_correct_count() -> None:
    boss = BOSSPlanner(n_samples=4, n_simulations=8, depth=2)
    q_static = jnp.ones(6, dtype=jnp.float32) / 6
    rng = np.random.default_rng(0)
    samples = boss.prepare_episode(q_static, rng)
    assert len(samples) == 4
    for s in samples:
        assert 0 <= s < 6


def test_sharp_q_static_samples_only_supported() -> None:
    boss = BOSSPlanner(n_samples=10, n_simulations=8, depth=2)
    q_static = jnp.zeros(6, dtype=jnp.float32).at[3].set(1.0)
    rng = np.random.default_rng(0)
    samples = boss.prepare_episode(q_static, rng)
    assert all(s == 3 for s in samples)


def test_plan_raises_before_prepare() -> None:
    boss = BOSSPlanner(n_samples=2, n_simulations=8, depth=2)
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    with pytest.raises(RuntimeError):
        boss.plan(q_state, q_static, T, B_list, goal, 3,
                  theta_mask=env.get_modality_registry().theta_dependent_mask)


def test_n_samples_1_sharp_theta_picks_correct_door_K3() -> None:
    """With n_samples=1 and a sharp posterior, BOSS reduces to a fixed-
    theta MCTS; with enough simulations it should converge to OPEN_DOOR
    on pi(0). Mirrors test_plan_with_sharp_theta_picks_correct_door_K3
    from the BAMCP test suite."""
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask

    for theta_idx in [0, 4]:
        env.reset(theta=theta_idx)
        correct = env.correct_door
        q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
        q_static = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_idx].set(1.0)

        boss = BOSSPlanner(
            n_samples=1, n_simulations=512, depth=4, rollout_seed=0,
        )
        rng = np.random.default_rng(0)
        boss.prepare_episode(q_static, rng)
        dist = np.asarray(boss.plan(
            q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
        ))
        best_action = int(dist.argmax())
        assert best_action == env.M + correct, (
            f"theta={theta_idx}: expected OPEN_DOOR_{correct} "
            f"(action {env.M + correct}), got {best_action}; dist={dist}"
        )


def test_run_block_with_boss() -> None:
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    boss = BOSSPlanner(n_samples=4, n_simulations=64, depth=4, rollout_seed=0)
    block = run_block(
        env, boss, theta_true=0, n_episodes=4,
        horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    assert len(block.episodes) == 4


def test_tree_keys_alias_states_under_many_to_one_B_position() -> None:
    """BOSS shares BAMCP's _step, so its tree branches must also key on
    view-index, not latent state, under aliased B_position. Smoke-test:
    drive a few _step calls with a many-to-one B_position and confirm
    that successors sharing a view share an obs_idx.
    """
    n_states = 4
    n_views = 2
    T = np.zeros((n_states, n_states, 1, 1), dtype=np.float64)
    T[1, 0, 0, 0] = 0.5
    T[2, 0, 0, 0] = 0.5
    B_position = np.zeros((n_views, n_states), dtype=np.float64)
    B_position[0, [0, 1]] = 1.0
    B_position[1, [2, 3]] = 1.0
    B_probe = np.zeros((3, n_states, 1), dtype=np.float64)
    B_probe[2, :, :] = 1.0
    B_outcome = np.zeros((3, n_states), dtype=np.float64)
    B_outcome[0, :] = 1.0

    rng = np.random.default_rng(0)
    obs_idx_per_view: dict[int, set[int]] = {0: set(), 1: set()}
    for _ in range(200):
        nxt, obs_idx, _ = BAMCPPlanner._step(
            state=0, theta=0, action=0,
            T=T, B_position=B_position, B_probe=B_probe, B_outcome=B_outcome,
            probe_state_set=set(), rng=rng,
        )
        view = int(np.argmax(B_position[:, nxt]))
        obs_idx_per_view[view].add(obs_idx)

    assert len(obs_idx_per_view[0]) == 1
    assert len(obs_idx_per_view[1]) == 1
    assert obs_idx_per_view[0].isdisjoint(obs_idx_per_view[1])


def test_root_state_is_sampled_from_q_state() -> None:
    """BOSS must sample the root state from q_state per simulation, not
    collapse to argmax. Different RNG seeds on a uniform two-state belief
    should yield different action distributions.
    """
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32)
    q_state = q_state.at[1].set(0.5).at[2].set(0.5)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    boss_a = BOSSPlanner(n_samples=2, n_simulations=128, depth=4, rollout_seed=0)
    boss_b = BOSSPlanner(n_samples=2, n_simulations=128, depth=4, rollout_seed=42)
    boss_a.prepare_episode(q_static, np.random.default_rng(0))
    boss_b.prepare_episode(q_static, np.random.default_rng(0))
    dist_a = np.asarray(boss_a.plan(
        q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
    ))
    dist_b = np.asarray(boss_b.plan(
        q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
    ))
    assert not np.allclose(dist_a, dist_b, atol=1e-3), (
        f"different RNGs gave identical visit dists -> root_state likely "
        f"collapsed to argmax: a={dist_a}, b={dist_b}"
    )


def test_run_outer_loop_with_boss() -> None:
    boss = BOSSPlanner(n_samples=4, n_simulations=64, depth=4, rollout_seed=0)
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=boss, n_seeds=3, n_episodes=4,
        horizon=3, max_steps=4, deterministic=True, seed=0,
        show_progress=False,
    )
    assert result.success.shape == (3, 4)
