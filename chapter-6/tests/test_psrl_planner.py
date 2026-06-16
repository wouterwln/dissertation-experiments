"""Tests for PSRLPlanner — the posterior-sampling RL baseline.

Covers:
- Episode preparation samples theta correctly from q_static.
- plan() raises if prepare_episode was never called.
- plan() with sample s is equivalent to VBPPlanner with sharp prior on s.
- run_block / run_outer_loop drive PSRL through the duck-typed
  prepare_episode hook.
- PSRL on the door-code env does not visit probes (it commits to a
  sampled theta and acts greedily); identification proceeds via door
  outcomes alone.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv, state_indices
from learningpomdp.experiment.door_code_runner import (
    run_block,
    run_episode,
    run_outer_loop,
)
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.psrl_planner import PSRLPlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


@pytest.fixture(scope="module")
def vbp() -> VBPPlanner:
    return VBPPlanner()


# ---------- Episode preparation ----------

def test_prepare_episode_samples_from_sharp_q_static(vbp: VBPPlanner) -> None:
    """A sharp q(theta) on theta=2 must sample 2 every time."""
    psrl = PSRLPlanner(vbp)
    q_static = jnp.zeros(6, dtype=jnp.float32).at[2].set(1.0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        sample = psrl.prepare_episode(q_static, rng)
        assert sample == 2
    assert psrl.current_sample == 2


def test_prepare_episode_uniform_samples_distributed(vbp: VBPPlanner) -> None:
    """Uniform q(theta) over 6 should spread samples across the support."""
    psrl = PSRLPlanner(vbp)
    q_static = jnp.ones(6, dtype=jnp.float32) / 6
    rng = np.random.default_rng(0)
    samples = [psrl.prepare_episode(q_static, rng) for _ in range(200)]
    assert set(samples) == set(range(6)), (
        f"expected all 6 thetas to appear; got {set(samples)}"
    )


def test_plan_raises_before_prepare(vbp: VBPPlanner) -> None:
    psrl = PSRLPlanner(vbp)
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B_c, dtype=jnp.float32) for B_c in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    with pytest.raises(RuntimeError):
        psrl.plan(q_state, q_static, T, B_list, goal, 3,
                  theta_mask=env.get_modality_registry().theta_dependent_mask)


def test_invalid_q_static_rejected(vbp: VBPPlanner) -> None:
    psrl = PSRLPlanner(vbp)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        psrl.prepare_episode(jnp.zeros(6, dtype=jnp.float32), rng)


# ---------- plan() equivalence ----------

def test_plan_matches_vbp_with_sharp_prior(vbp: VBPPlanner) -> None:
    """PSRL with sample=s plans identically to VBP given a one-hot q(theta)
    on s — same env tensors, same goal, same horizon."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B_c, dtype=jnp.float32) for B_c in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[0].set(1.0)

    psrl = PSRLPlanner(vbp)
    rng = np.random.default_rng(0)
    sample = psrl.prepare_episode(
        jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static, rng,
    )

    psrl_action = np.asarray(psrl.plan(
        q_state, jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static,
        T, B_list, goal, 3, theta_mask=theta_mask,
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
    ))

    sharp = jnp.zeros(env.n_static, dtype=jnp.float32).at[sample].set(1.0)
    vbp_action = np.asarray(vbp.plan(
        q_state, sharp, T, B_list, goal, 3, theta_mask=theta_mask,
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
    ))

    np.testing.assert_allclose(psrl_action, vbp_action, atol=1e-6)


# ---------- Runner integration ----------

def test_run_block_with_psrl_plays_through(vbp: VBPPlanner) -> None:
    """PSRL runs through run_block; prepare_episode is invoked at each
    episode boundary."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    psrl = PSRLPlanner(vbp)
    block = run_block(
        env, psrl, theta_true=0, n_episodes=4,
        horizon=3, max_steps=2, deterministic=True, seed=0,
    )
    assert len(block.episodes) == 4
    # PSRL acts greedily on its sample, so trajectories are short.
    for ep in block.episodes:
        assert ep.n_steps >= 1


def test_psrl_does_not_visit_probes(vbp: VBPPlanner) -> None:
    """PSRL commits to a sampled theta and acts greedily; with default
    parameters and a deterministic env it never visits probe stations.

    This is a key behavioral distinction from AIF: PSRL learns from
    door outcomes only, AIF learns from probes and door outcomes."""
    env = DoorCodeEnv(K=4, probe_noise=0.2)
    psrl = PSRLPlanner(vbp)
    block = run_block(
        env, psrl, theta_true=0, n_episodes=15,
        horizon=3, max_steps=2, deterministic=True, seed=0,
    )
    total_probes = sum(ep.probe_visits for ep in block.episodes)
    assert total_probes == 0, (
        f"PSRL visited a probe — expected greedy door commits only, "
        f"got {total_probes} probe visits"
    )


def test_run_outer_loop_with_psrl(vbp: VBPPlanner) -> None:
    """run_outer_loop creates a fresh planner per call but PSRL needs to
    persist state. Pass a single PSRL instance — prepare_episode is
    called at each episode start so cross-episode state doesn't leak."""
    psrl = PSRLPlanner(vbp)
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=psrl, n_seeds=3, n_episodes=4,
        horizon=3, max_steps=2, deterministic=True, seed=0,
    )
    assert result.success.shape == (3, 4)
    # PSRL identification: by episode 3 most seeds should have solved.
    assert result.success[:, -1].mean() >= 0.5


# ---------- PSRL identifies via door outcomes ----------

def test_psrl_identifies_after_few_door_commits(vbp: VBPPlanner) -> None:
    """PSRL on K=3 deterministic should identify pi(0) within a few
    episodes via failed-door evidence."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    psrl = PSRLPlanner(vbp)
    block = run_block(
        env, psrl, theta_true=2, n_episodes=10,
        horizon=3, max_steps=2, deterministic=True, seed=0,
    )
    # Cumulative success rate across all 10 episodes should be well
    # above the 1/3 random-door baseline.
    successes = [int(ep.success) for ep in block.episodes]
    assert sum(successes) >= 6, (
        f"expected >=6 successes in 10 episodes for K=3 PSRL; "
        f"got {successes}"
    )


# ---------- AIF also implements the Planner protocol ----------

def test_psrl_wraps_aif_too() -> None:
    """PSRL should be base-planner-agnostic; wrap AIF and run."""
    psrl = PSRLPlanner(AIFPlanner())
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    block = run_block(
        env, psrl, theta_true=0, n_episodes=4,
        horizon=3, max_steps=2, deterministic=True, seed=0,
    )
    assert len(block.episodes) == 4
