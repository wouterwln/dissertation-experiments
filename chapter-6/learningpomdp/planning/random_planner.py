"""Random baseline — uniform action selection.

Lower-bound planner for the M1 ablation: makes no use of the model,
the belief, or the goal. The mean episode return under RandomPlanner
is the floor any informed method must clear. Stays inside the
inference-only-planner constraint by virtue of being a degenerate
inference (one-hot posterior at a uniformly sampled action).

`plan()` returns a one-hot rather than the uniform vector so the
runner's deterministic argmax selects the sampled action instead of
collapsing onto action 0 every step. `prepare_episode` adopts the
runner's per-seed RNG (the PSRL pattern), keeping action sequences
reproducible per seed. `n_actions` is read from `T.shape[-1]` so the
planner is env-agnostic and reusable on the spatial (M2) and
factorized-theta (M3) envs without changes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class RandomPlanner:
    """Returns a one-hot at a uniformly sampled action at every step."""

    def __init__(self) -> None:
        # Fallback RNG for bare plan() calls; the runner's per-seed RNG
        # replaces it via prepare_episode.
        self._rng = np.random.default_rng(0)

    def prepare_episode(
        self,
        q_static: jnp.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self._rng = rng

    def plan(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B_list: list,
        goal,
        horizon: int,
        theta_mask=None,
        theta_spec=None,
        t_factor_spec=None,
        b_factor_specs=None,
    ) -> jnp.ndarray:
        n_actions = T.shape[-1]
        action = int(self._rng.integers(n_actions))
        return jnp.zeros(n_actions, dtype=jnp.float32).at[action].set(1.0)
