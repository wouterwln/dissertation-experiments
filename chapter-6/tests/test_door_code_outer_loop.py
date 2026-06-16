"""Tests for run_outer_loop and the OuterLoopResult JSON summary.

Covers:
- Aggregate metric arrays have shape (n_seeds, n_episodes).
- Determinism: same seed yields identical results.
- to_summary_dict() round-trips through json.dumps.
- theta_seeds override pins thetas explicitly.
- Custom env knobs (lock_doors_fn, goal) flow through.
"""

import json

import numpy as np
import pytest

from learningpomdp.environments.door_code import (
    DoorCodeEnv,
    create_goal,
    enumerate_permutations,
)
from learningpomdp.experiment.door_code_runner import (
    OuterLoopResult,
    run_outer_loop,
)
from learningpomdp.planning.aif_planner import AIFPlanner


@pytest.fixture(scope="module")
def planner() -> AIFPlanner:
    return AIFPlanner()


def test_outer_loop_aggregate_shapes(planner: AIFPlanner) -> None:
    n_seeds, n_episodes = 4, 6
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=planner,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    assert isinstance(result, OuterLoopResult)
    assert result.n_seeds == n_seeds
    assert result.n_episodes == n_episodes
    assert len(result.theta_trues) == n_seeds
    assert len(result.blocks) == n_seeds

    expected_shape = (n_seeds, n_episodes)
    assert result.success.shape == expected_shape
    assert result.n_steps.shape == expected_shape
    assert result.probe_visits.shape == expected_shape
    assert result.posterior_mass_true.shape == expected_shape
    assert result.posterior_entropy.shape == expected_shape


def test_outer_loop_determinism(planner: AIFPlanner) -> None:
    """Same seed -> same results (theta_trues, success arrays)."""
    kwargs = dict(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=planner, n_seeds=3, n_episodes=4,
        horizon=3, max_steps=4, deterministic=True, seed=42,
    )
    a = run_outer_loop(**kwargs)
    b = run_outer_loop(**kwargs)

    assert a.theta_trues == b.theta_trues
    np.testing.assert_array_equal(a.success, b.success)
    np.testing.assert_allclose(a.posterior_mass_true, b.posterior_mass_true)


def test_outer_loop_different_seeds_differ(planner: AIFPlanner) -> None:
    """Different seeds yield different theta_trues (with high probability)."""
    a = run_outer_loop(
        env_kwargs=dict(K=4, probe_noise=0.0),
        planner=planner, n_seeds=8, n_episodes=2,
        horizon=3, max_steps=2, deterministic=True, seed=0,
    )
    b = run_outer_loop(
        env_kwargs=dict(K=4, probe_noise=0.0),
        planner=planner, n_seeds=8, n_episodes=2,
        horizon=3, max_steps=2, deterministic=True, seed=1,
    )
    assert a.theta_trues != b.theta_trues


def test_theta_seeds_override(planner: AIFPlanner) -> None:
    pinned = [0, 1, 2, 3]
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=planner, n_seeds=4, n_episodes=2,
        horizon=3, max_steps=4, deterministic=True, seed=0,
        theta_seeds=pinned,
    )
    assert list(result.theta_trues) == pinned


def test_theta_seeds_length_validated(planner: AIFPlanner) -> None:
    with pytest.raises(ValueError):
        run_outer_loop(
            env_kwargs=dict(K=3, probe_noise=0.0),
            planner=planner, n_seeds=4, n_episodes=2,
            horizon=3, max_steps=4, deterministic=True, seed=0,
            theta_seeds=[0, 1, 2],  # wrong length
        )


def test_summary_dict_json_serializable(planner: AIFPlanner) -> None:
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.1),
        planner=planner, n_seeds=2, n_episodes=3,
        horizon=3, max_steps=3, deterministic=True, seed=0,
    )
    summary = result.to_summary_dict()
    blob = json.dumps(summary)  # must not raise
    parsed = json.loads(blob)

    assert parsed["n_seeds"] == 2
    assert parsed["n_episodes"] == 3
    assert "config" in parsed
    assert "per_episode" in parsed
    assert "curves" in parsed

    for name in ("success", "n_steps", "probe_visits",
                 "posterior_mass_on_true_theta", "posterior_entropy"):
        per_ep = parsed["per_episode"][name]
        assert len(per_ep) == 2
        assert all(len(row) == 3 for row in per_ep)
        curve = parsed["curves"][name]
        for stat in ("mean", "p05", "p95"):
            assert len(curve[stat]) == 3


def test_custom_lock_doors_fn_flows_through(planner: AIFPlanner) -> None:
    """Override lock_doors_fn so a different cell of pi is action-relevant
    (single-step lock pointing at the LAST cell). Sharp prior on theta
    still solves correctly with the new rule."""
    import jax.numpy as jnp

    last_cell = lambda perm: (perm[-1],)  # length-1 tuple at lock_length=1
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1, lock_doors_fn=last_cell)

    perms = enumerate_permutations(3)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32) for B_c in env.get_observation_tensors()
    ]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask

    for theta_idx in range(env.n_static):
        expected_door = perms[theta_idx][-1]
        q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[0].set(1.0)
        q_static = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_idx].set(1.0)
        action_dist = planner.plan(
            q_state, q_static, T, B_list, goal, 3, theta_mask=theta_mask,
            theta_spec=env.theta_spec,
            t_factor_spec=env.t_factor_spec,
            b_factor_specs=env.b_factor_specs,
        )
        best = int(np.asarray(action_dist).argmax())
        assert best == env.M + expected_door, (
            f"theta={theta_idx}: expected OPEN_DOOR_{expected_door} "
            f"(action {env.M + expected_door}), got {best}"
        )

    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0, lock_length=1, lock_doors_fn=last_cell),
        planner=planner, n_seeds=2, n_episodes=2,
        horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    assert result.success.shape == (2, 2)


def test_custom_goal_flows_through(planner: AIFPlanner) -> None:
    """Override goal with a stronger SUCCESS preference; outer loop runs."""
    custom_goal = create_goal(K=3, M=3, success_logit=8.0, failure_logit=-8.0)
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0, goal=custom_goal),
        planner=planner, n_seeds=2, n_episodes=2,
        horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    assert result.success.shape == (2, 2)
