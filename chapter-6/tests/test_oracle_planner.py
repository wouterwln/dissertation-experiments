"""Tests for OraclePlanner — the M1 upper-bound baseline."""

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_outer_loop
from learningpomdp.planning.oracle_planner import OraclePlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


def test_prepare_episode_required() -> None:
    """plan() before prepare_episode must raise."""
    planner = OraclePlanner(VBPPlanner(n_iterations=10))
    with pytest.raises(RuntimeError):
        env = DoorCodeEnv(K=3, probe_noise=0.0)
        T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
        B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
        goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
        q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[0].set(1.0)
        q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
        planner.plan(q_state, q_static, T, B_list, goal, horizon=4)


def test_oracle_picks_correct_door_despite_misleading_belief() -> None:
    """Oracle ignores q_static; given the true theta it should pick the
    correct OPEN_DOOR action even if the agent's belief is sharp on the
    wrong theta. This isolates the q_static collapse that defines the
    oracle.
    """
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask

    rng = np.random.default_rng(0)
    for theta_true in [0, 4]:
        env.reset(theta=theta_true)
        correct = env.correct_door
        # Agent's belief is sharp on the WRONG theta; oracle should
        # ignore it and use theta_true.
        wrong = (theta_true + 1) % env.n_static
        q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
        q_static = jnp.zeros(env.n_static, dtype=jnp.float32).at[wrong].set(1.0)

        planner = OraclePlanner(VBPPlanner(n_iterations=80))
        planner.prepare_episode(q_static, rng, theta_true=theta_true)
        dist = np.asarray(planner.plan(
            q_state, q_static, T, B_list, goal, horizon=4,
            theta_mask=theta_mask,
            theta_spec=env.theta_spec,
            t_factor_spec=env.t_factor_spec,
            b_factor_specs=env.b_factor_specs,
        ))
        best_action = int(dist.argmax())
        assert best_action == env.M + correct, (
            f"theta_true={theta_true}: expected OPEN_DOOR_{correct} "
            f"(action {env.M + correct}), got {best_action}; dist={dist}"
        )


def test_run_outer_loop_with_oracle() -> None:
    planner = OraclePlanner(VBPPlanner(n_iterations=80))
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=planner, n_seeds=2, n_episodes=3,
        horizon=4, max_steps=4, deterministic=True, seed=0,
        show_progress=False,
    )
    assert result.success.shape == (2, 3)
    # Oracle with zero probe noise + lock_length default should solve
    # every episode.
    assert (result.success == 1).all(), (
        f"Oracle failed an episode under noiseless K=3: success={result.success}"
    )
