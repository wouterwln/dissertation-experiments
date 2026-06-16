"""Multiprocessing correctness: 4-worker run produces the same per-seed
outputs as 1-worker run.

Each seed is deterministically assigned to a worker by seed index, so
the per-seed RNG path is identical to sequential execution. This holds
because:

- Each worker constructs its own ``RockSampleEnv`` from the same
  ``env_kwargs + obs_seed = seed*10_000 + s`` recipe.
- Each worker constructs its own planner via ``planner_factory()``,
  which produces a freshly-seeded ``np.random.Generator``.
- ``run_block(seed=seed*10_000+s)`` uses the same per-seed offset.

So the parallel path is a re-ordering of the sequential one — same
outputs, different wall-clock execution.
"""

from __future__ import annotations

import numpy as np

from learningpomdp.experiment.rock_sample_runner import run_outer_loop
from learningpomdp.planning.bamcp_planner import BAMCPPlanner


def _factory():
    # Module-level closure so it's pickle-friendly under spawn.
    return BAMCPPlanner(
        n_simulations=32, depth=3, ucb_c=1.0, gamma=0.99,
        rollout_seed=0,
    )


def test_workers_4_matches_workers_1_for_bamcp():
    cfg = dict(
        env_kwargs=dict(K=2, grid_size=3, layout_seed=0, max_steps=6),
        n_seeds=4,
        n_episodes=2,
        horizon=2,
        max_steps=6,
        seed=0,
        show_progress=False,
    )
    seq = run_outer_loop(planner_factory=_factory, workers=1, **cfg)
    par = run_outer_loop(planner_factory=_factory, workers=4, **cfg)
    assert np.allclose(seq.cumulative_reward, par.cumulative_reward), (
        f"sequential={seq.cumulative_reward}\nparallel={par.cumulative_reward}"
    )
    assert np.allclose(
        seq.posterior_entropy_persistent, par.posterior_entropy_persistent,
    )
    assert np.allclose(
        seq.posterior_entropy_episodic, par.posterior_entropy_episodic,
    )
