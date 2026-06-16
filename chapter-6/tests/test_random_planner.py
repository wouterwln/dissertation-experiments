"""Tests for RandomPlanner — the M1 lower-bound baseline."""

import jax.numpy as jnp
import numpy as np

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_outer_loop
from learningpomdp.planning.random_planner import RandomPlanner


def _plan_inputs(env: DoorCodeEnv):
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    return q_state, q_static, T, B_list, goal


def test_plan_returns_one_hot() -> None:
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    planner = RandomPlanner()
    q_state, q_static, T, B_list, goal = _plan_inputs(env)

    dist = np.asarray(planner.plan(
        q_state, q_static, T, B_list, goal, horizon=4,
        theta_mask=env.get_modality_registry().theta_dependent_mask,
    ))
    assert dist.shape == (env.n_actions,)
    np.testing.assert_allclose(dist.sum(), 1.0, atol=1e-6)
    assert np.count_nonzero(dist) == 1


def test_plan_samples_varied_actions() -> None:
    """Repeated plan() calls must not collapse onto a single action —
    argmax over the returned one-hot is the runner's action, so this is
    exactly the all-action-0 regression."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    planner = RandomPlanner()
    planner.prepare_episode(None, np.random.default_rng(0))
    q_state, q_static, T, B_list, goal = _plan_inputs(env)

    actions = [
        int(np.argmax(planner.plan(q_state, q_static, T, B_list, goal, horizon=4)))
        for _ in range(50)
    ]
    assert len(set(actions)) > 1
    assert set(actions) <= set(range(env.n_actions))


def test_plan_reproducible_given_rng_seed() -> None:
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    q_state, q_static, T, B_list, goal = _plan_inputs(env)

    sequences = []
    for _ in range(2):
        planner = RandomPlanner()
        planner.prepare_episode(None, np.random.default_rng(42))
        sequences.append([
            int(np.argmax(planner.plan(q_state, q_static, T, B_list, goal, horizon=4)))
            for _ in range(20)
        ])
    assert sequences[0] == sequences[1]


def test_run_outer_loop_with_random() -> None:
    planner = RandomPlanner()
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=planner, n_seeds=2, n_episodes=3,
        horizon=4, max_steps=4, deterministic=False, seed=0,
        show_progress=False,
    )
    assert result.success.shape == (2, 3)
