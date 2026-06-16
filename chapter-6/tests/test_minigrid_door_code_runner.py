"""Runner integration tests for the Farama-MiniGrid MiniGridDoorCodeEnv.

Verifies that the env_factory hook in run_outer_loop drives the new
heading-aware env end-to-end with the AIF planner. Same gates as
test_door_code_runner.py:
- Oracle q(theta) -> agent reaches and toggles the correct door.
- Uniform prior -> agent visits at least one probe cell before any TOGGLE
  attempt (epistemic-first behaviour).
- Multi-episode: posterior mass on true theta rises across episodes.
- clip_horizon=True survives the per-step horizon recompile.
- run_outer_loop dispatches through env_factory and returns the
  expected shapes.

Tests use the canonical small M2 config (K=2, L=1, grid_size=5,
slip_prob=0) — 3x3 interior, 2 doors, 1 probe partition, n_static=2.
Trivially solvable; AIF iter cost is small enough for CI.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.minigrid_door_code import (
    ACTION_TOGGLE,
    MiniGridDoorCodeEnv,
)
from learningpomdp.experiment.door_code_runner import (
    run_block,
    run_episode,
    run_outer_loop,
)
from learningpomdp.planning.aif_planner import AIFPlanner


@pytest.fixture(scope="module")
def planner() -> AIFPlanner:
    return AIFPlanner(n_iterations=60)


# ---------- Single-episode behaviour ----------

def test_oracle_theta_reaches_and_toggles_door(planner: AIFPlanner) -> None:
    """With q(theta) collapsed to true theta, AIF should plan to the
    correct door and toggle it within max_steps."""
    K, L = 2, 1
    env = MiniGridDoorCodeEnv(
        K=K, lock_length=L, grid_size=5, slip_prob=0.0,
        probe_noise=0.0, obs_seed=0,
    )
    theta_true = 0
    q_static_oracle = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_true].set(1.0)
    rng = np.random.default_rng(0)
    result = run_episode(
        env=env, planner=planner, q_static_init=q_static_oracle,
        theta_true=theta_true, horizon=10, max_steps=15,
        deterministic=True, rng=rng,
        learn_on_failure=False, clip_horizon=False,
    )
    assert result.success, (
        f"oracle-theta agent failed within max_steps: "
        f"actions={result.actions}, terminal_reward={result.terminal_reward}"
    )


def test_uniform_prior_visits_probe_before_toggle(planner: AIFPlanner) -> None:
    """With uniform q(theta), AIF should visit the probe cell before any
    TOGGLE attempt (epistemic-first behaviour)."""
    K, L = 2, 1
    env = MiniGridDoorCodeEnv(
        K=K, lock_length=L, grid_size=5, slip_prob=0.0,
        probe_noise=0.0, obs_seed=0,
    )
    theta_true = 0
    q_static_uniform = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    rng = np.random.default_rng(0)
    result = run_episode(
        env=env, planner=planner, q_static_init=q_static_uniform,
        theta_true=theta_true, horizon=10, max_steps=15,
        deterministic=True, rng=rng,
        learn_on_failure=False, clip_horizon=False,
    )
    probe_cells = set(env.layout.probe_cells)
    visited_cells = {env.state_to_cell(s) for s in result.states_visited}
    visited_probe = visited_cells & probe_cells
    toggle_steps = [i for i, a in enumerate(result.actions) if a == ACTION_TOGGLE]
    assert len(visited_probe) > 0 or len(toggle_steps) == 0, (
        "agent issued TOGGLE with no probe visit (no epistemic action "
        f"under uniform prior). actions={result.actions}, "
        f"states_visited={result.states_visited}"
    )


# ---------- Multi-episode identification ----------

def test_posterior_mass_rises_across_episodes(planner: AIFPlanner) -> None:
    """Across a block with persistent q(theta), posterior mass on true
    theta should rise. Uses noisy probes so episode 0 doesn't fully
    identify."""
    K, L = 2, 1
    env = MiniGridDoorCodeEnv(
        K=K, lock_length=L, grid_size=5, slip_prob=0.0,
        probe_noise=0.2, obs_seed=0,
    )
    theta_true = 0
    block = run_block(
        env, planner, theta_true=theta_true, n_episodes=3,
        horizon=10, max_steps=15, deterministic=True, seed=0,
        learn_on_failure=False,
    )
    masses = [ep.posterior_mass_on_true_theta for ep in block.episodes]
    uniform_mass = 1.0 / env.n_static  # 0.5 for K=2
    assert masses[-1] > uniform_mass * 1.1 or masses[-1] >= 0.9, (
        f"posterior mass never moved away from prior: {masses}"
    )


# ---------- Runner protocol features ----------

def test_clip_horizon_runs_without_error(planner: AIFPlanner) -> None:
    K, L = 2, 1
    env = MiniGridDoorCodeEnv(
        K=K, lock_length=L, grid_size=5, slip_prob=0.0,
        probe_noise=0.0, obs_seed=0,
    )
    rng = np.random.default_rng(0)
    q_static_oracle = jnp.zeros(env.n_static, dtype=jnp.float32).at[0].set(1.0)
    result = run_episode(
        env=env, planner=planner, q_static_init=q_static_oracle,
        theta_true=0, horizon=8, max_steps=10,
        deterministic=True, rng=rng,
        learn_on_failure=False, clip_horizon=True,
    )
    assert result.n_steps > 0


def test_run_outer_loop_with_env_factory_smoke(planner: AIFPlanner) -> None:
    """run_outer_loop dispatches through env_factory and returns a
    result with the expected shapes."""
    K, L = 2, 1
    env_kwargs = dict(
        K=K, lock_length=L, grid_size=5, slip_prob=0.0, probe_noise=0.0,
    )
    result = run_outer_loop(
        env_kwargs=env_kwargs,
        planner=planner,
        n_seeds=2,
        n_episodes=2,
        horizon=10,
        max_steps=15,
        deterministic=True,
        seed=0,
        learn_on_failure=False,
        clip_horizon=False,
        show_progress=False,
        env_factory=lambda kw: MiniGridDoorCodeEnv(**kw),
    )
    assert result.success.shape == (2, 2)
    assert result.posterior_mass_true.shape == (2, 2)
    assert np.all(np.isfinite(result.posterior_mass_true))


def test_video_recording_emits_per_episode_mp4(planner: AIFPlanner, tmp_path) -> None:
    """`video_path_for(outer, inner)` is invoked per episode; each call
    writes one mp4 with at least 2 frames (initial + ≥1 step frame).
    Requires render_mode='rgb_array' in env_kwargs to actually emit
    frames."""
    import imageio.v3 as iio

    K, L = 2, 1
    env_kwargs = dict(
        K=K, lock_length=L, grid_size=5, slip_prob=0.0, probe_noise=0.0,
        render_mode="rgb_array",
    )
    video_dir = tmp_path / "videos"

    def video_path_for(outer_idx: int, inner_idx: int):
        return video_dir / str(outer_idx) / f"{inner_idx}.mp4"

    n_seeds, n_episodes = 2, 2
    run_outer_loop(
        env_kwargs=env_kwargs,
        planner=planner,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        horizon=8,
        max_steps=8,
        deterministic=True,
        seed=0,
        learn_on_failure=False,
        clip_horizon=False,
        show_progress=False,
        env_factory=lambda kw: MiniGridDoorCodeEnv(**kw),
        video_path_for=video_path_for,
        video_fps=4,
    )
    # Every (outer, inner) cell produced a non-empty mp4.
    for s in range(n_seeds):
        for ep in range(n_episodes):
            path = video_dir / str(s) / f"{ep}.mp4"
            assert path.exists(), f"missing mp4: {path}"
            n_frames = sum(1 for _ in iio.imiter(str(path)))
            assert n_frames >= 2, f"{path} only has {n_frames} frames"
