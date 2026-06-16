"""Runner-level smoke tests for the Factored Tiger BA-POMDP.

These exist as a CI guard on the wiring between
``FactoredTigerEnv`` and ``factored_tiger_runner``: result shape,
posterior normalisation, persistent-entropy monotonicity under AIF,
and NaN handling for the ``reward_normalised`` edge case where a block
draws an all-tiger θ.

Kept deliberately small (K=2, n_episodes=2, max_steps=4, n_iterations=20)
so the suite stays cheap enough to run locally.
"""

from __future__ import annotations

import numpy as np
import pytest

from learningpomdp.environments.factored_tiger import FactoredTigerEnv
from learningpomdp.experiment.factored_tiger_runner import (
    run_block,
    run_outer_loop,
)
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.random_planner import RandomPlanner


@pytest.fixture(scope="module")
def aif_planner() -> AIFPlanner:
    return AIFPlanner(n_iterations=20, damping=0.9)


def test_run_block_with_random_planner():
    env = FactoredTigerEnv(K=2, listen_accuracy=1.0, max_steps=4)
    block = run_block(
        env, RandomPlanner(), block_theta_true=3,  # (1, 1) — both safe
        n_episodes=2, horizon=3, max_steps=4, deterministic=False, seed=0,
    )
    assert block.K == 2
    assert block.theta_true == 3
    assert len(block.episodes) == 2
    for ep in block.episodes:
        assert ep.n_steps == 4
        q = np.asarray(ep.final_q_static)
        # q_static is stored in factorized shape (component_sizes), not flat.
        assert q.size == env.n_static
        assert np.isclose(q.sum(), 1.0)
        assert np.isfinite(ep.cumulative_reward)


def test_run_block_with_aif_planner_tightens_persistent_posterior(aif_planner):
    """AIF on a noiseless K=2 instance: across-block persistent entropy
    must not increase. Listen accuracy=1.0 makes each LISTEN strictly
    informative, so any selected listen tightens q(θ)."""
    env = FactoredTigerEnv(K=2, listen_accuracy=1.0, max_steps=4)
    block = run_block(
        env, aif_planner, block_theta_true=2,  # (0, 1)
        n_episodes=2, horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    assert len(block.episodes) == 2
    for ep in block.episodes:
        q = np.asarray(ep.final_q_static)
        assert np.isclose(q.sum(), 1.0)
        assert np.isfinite(ep.cumulative_reward)
    h0 = block.episodes[0].posterior_entropy_persistent
    h1 = block.episodes[-1].posterior_entropy_persistent
    assert h1 <= h0 + 1e-6


def test_run_outer_loop_smoke_handles_all_tiger_normalisation():
    """All-tiger block (θ=0) has max_possible_reward=0 → reward_normalised
    column must be NaN, not infinite, and downstream aggregates must
    still produce finite values via NaN-safe means."""
    env_kwargs = dict(K=2, listen_accuracy=1.0, max_steps=4)
    result = run_outer_loop(
        env_kwargs=env_kwargs,
        planner_factory=RandomPlanner,
        n_seeds=2,
        n_episodes=2,
        horizon=3,
        max_steps=4,
        deterministic=False,
        seed=0,
        theta_seeds=[0, 3],  # all-tigers, all-safes
        clip_horizon=False,
        show_progress=False,
        workers=1,
    )
    assert result.cumulative_reward.shape == (2, 2)
    assert result.reward_normalised.shape == (2, 2)
    # All-tiger seed: max_possible_reward=0 -> NaN column.
    assert np.all(np.isnan(result.reward_normalised[0]))
    # All-safes seed: finite normalisation.
    assert np.all(np.isfinite(result.reward_normalised[1]))

    summary = result.to_summary_dict()
    # NaN-safe across-seed mean: the all-safes seed alone keeps the curve finite.
    norm_curve = summary["curves"]["reward_normalised"]["mean"]
    assert all(np.isfinite(v) for v in norm_curve)
