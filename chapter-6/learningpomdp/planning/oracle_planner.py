"""Oracle baseline — VBP planner with q(theta) collapsed to truth.

Upper-bound planner for the M1 ablation: knows theta exactly. The
generative model is the same; only q_static differs (a one-hot at
theta_true instead of the agent's posterior). Conceptually a regular
POMDP solver applied to the BA-POMDP under perfect theta knowledge.

Wraps a VBPPlanner (per the user's choice for the inference-based
oracle), so the same machinery generalises to the spatial (M2) and
factorized-theta (M3) envs without a second hand-rolled policy.

Runner contract: at episode start, the runner calls
`prepare_episode(q_static, rng, theta_true=theta_true)` so the oracle
can stash the truth. `plan()` then forwards to the VBP planner with
`q_static` replaced by a one-hot at `theta_true`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from learningpomdp.planning.base import Planner


class OraclePlanner:
    """Plans against a one-hot q(theta) anchored at theta_true.

    Args:
        base_planner: model-based planner used for the inner solve given
            known theta. VBP is the canonical choice.
    """

    def __init__(self, base_planner: Planner) -> None:
        self._base = base_planner
        self._theta_true: int | None = None

    def prepare_episode(
        self,
        q_static: jnp.ndarray,
        rng: np.random.Generator,
        theta_true: int | None = None,
    ) -> None:
        if theta_true is None:
            raise RuntimeError(
                "OraclePlanner.prepare_episode requires theta_true; "
                "the runner must pass it as a kwarg."
            )
        self._theta_true = int(theta_true)

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
        if self._theta_true is None:
            raise RuntimeError(
                "OraclePlanner.prepare_episode() must be called before plan()."
            )
        n_static = int(np.prod(q_static.shape))
        if not 0 <= self._theta_true < n_static:
            raise ValueError(
                f"theta_true {self._theta_true} out of range [0, {n_static})"
            )
        # Build a flat one-hot; the base planner reshapes to component_sizes
        # internally.
        sharp = jnp.zeros(n_static, dtype=jnp.float32).at[self._theta_true].set(1.0)
        return self._base.plan(
            q_state, sharp, T, B_list, goal, horizon,
            theta_mask=theta_mask,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
        )
