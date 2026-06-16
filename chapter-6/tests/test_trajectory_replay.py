"""Replay-sufficiency: the per-episode trajectory blob stored in the
JSON contains everything needed to reconstruct the final q(θ) by
re-running the Bayesian filter offline. Pinning this contract guarantees
that the full `output/` tree can be used to recompute arbitrary metrics
without re-running planners.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_episode, run_outer_loop
from learningpomdp.inference.state_inference import bayesian_state_update
from learningpomdp.planning.aif_planner import AIFPlanner


def test_door_code_trajectory_replays_final_posterior() -> None:
    """Run one short door_code episode, then replay (actions, observations)
    through the Bayesian filter starting from `initial_q_static` and
    confirm we recover the saved `final_q_static`.
    """
    K = 3
    env = DoorCodeEnv(K=K, probe_noise=0.0)
    planner = AIFPlanner()
    q_static_init = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    result = run_episode(
        env=env, planner=planner, q_static_init=q_static_init,
        theta_true=0, horizon=3, max_steps=3,
        deterministic=True, rng=np.random.default_rng(0),
    )
    assert len(result.actions) == len(result.observations)
    assert result.actions, "episode produced no steps; can't exercise replay"

    # Replay against a fresh env to get the same T/B used during the run.
    # The matrices are deterministic functions of the env's static config,
    # so a fresh env gives identical tensors (no obs_seed dependence here).
    env_replay = DoorCodeEnv(K=K, probe_noise=0.0)
    env_replay.reset(theta=0)
    T = jnp.asarray(env_replay.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32)
        for B_c in env_replay.get_observation_tensors()
    ]
    log_B_list = tuple(jnp.log(B_c + 1e-12) for B_c in B_list)
    theta_spec = env_replay.theta_spec
    t_factor_spec = env_replay.t_factor_spec
    b_factor_specs = env_replay.b_factor_specs
    n_actions = env_replay.n_actions

    q_state = jnp.zeros(env_replay.n_states, dtype=jnp.float32).at[
        env_replay.start_state_index
    ].set(1.0)
    q_static = jnp.asarray(result.initial_q_static, dtype=jnp.float32)

    for action, obs in zip(result.actions, result.observations):
        obs_jnp = jnp.asarray(obs, dtype=jnp.float32)
        action_onehot = jnp.zeros(n_actions, dtype=jnp.float32).at[action].set(1.0)
        q_state, q_static = bayesian_state_update(
            q_state, q_static, T, log_B_list, obs_jnp, action_onehot,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
        )

    replayed = np.asarray(q_static).reshape(-1)
    saved = np.asarray(result.final_q_static).reshape(-1)
    np.testing.assert_allclose(replayed, saved, atol=1e-5)


def test_outer_loop_summary_contains_trajectories() -> None:
    """The aggregated JSON-friendly summary must include a `trajectories`
    block shaped (n_seeds, n_episodes), with one dict per episode that
    carries the trajectory fields used by the replay path.
    """
    env_kwargs = {"K": 3, "probe_noise": 0.0}
    result = run_outer_loop(
        env_kwargs=env_kwargs,
        planner=AIFPlanner(),
        n_seeds=2,
        n_episodes=2,
        horizon=3,
        max_steps=3,
        deterministic=True,
        seed=0,
        show_progress=False,
    )
    summary = result.to_summary_dict()

    assert "trajectories" in summary
    trajs = summary["trajectories"]
    assert len(trajs) == 2  # n_seeds
    for per_seed in trajs:
        assert len(per_seed) == 2  # n_episodes
        for ep in per_seed:
            for key in (
                "actions",
                "observations",
                "states_visited",
                "initial_q_static",
                "final_q_static",
                "terminated_early",
            ):
                assert key in ep, f"missing trajectory key {key!r}"
            assert len(ep["actions"]) == len(ep["observations"])
            # K=3 -> n_static = 3! = 6 permutations.
            assert len(ep["initial_q_static"]) == 6
            assert len(ep["final_q_static"]) == 6
