"""Posterior-Sampling for Reinforcement Learning (PSRL).

PSRL (Strens 2000; Osband, Russo, Van Roy 2013) is the principled Bayesian
model-based RL competitor to AIF on the door-code task: it shares the
generative model (T, B, theta prior, goal) but replaces AIF's expected
free energy with a single sample of theta from q(theta) per episode and
a deterministic plan against that sample.

The same machinery as AIF when q(theta) is sharp on the sampled value.
The runner is responsible for episode boundaries: it calls
`prepare_episode(q_static, rng)` before each episode so PSRL knows when
to draw a fresh theta_sample.

State inference and the actual q(theta) update live in the runner, not
here — PSRL only changes WHAT q_static the planner *plans against*; the
agent's belief over theta evolves the same way it does under AIF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
import numpy as np

from learningpomdp.inference.state_inference import (
    _condition_on_persistent,
    sample_persistent_theta,
)
from learningpomdp.planning.base import Planner

if TYPE_CHECKING:
    from learningpomdp.factors import ThetaSpec


class PSRLPlanner:
    """Planner wrapper that samples theta once per episode and plans
    deterministically against that sample.

    Wraps any model-based base planner (VBPPlanner, AIFPlanner). Calls
    to plan() forward to the base planner with q_static replaced by a
    one-hot on the sampled theta_idx; q_state is unchanged.

    Usage:
        psrl = PSRLPlanner(VBPPlanner())
        # at episode start (runner does this if prepare_episode exists):
        psrl.prepare_episode(q_static, rng)
        # during the episode, plan() picks up the sample:
        action_dist = psrl.plan(q_state, q_static, T, B_list, goal, horizon)
    """

    def __init__(self, base_planner: Planner) -> None:
        self._base = base_planner
        self._sample_idx: int | None = None
        # Persistent-axis values sampled at episode start. plan() uses
        # these to recondition the *current* q_static every step so the
        # episodic axes track within-episode evidence. None when the
        # legacy flat-sample path is in effect (door-code).
        self._persistent_idx_tuple: Optional[tuple[int, ...]] = None

    @property
    def current_sample(self) -> int | None:
        return self._sample_idx

    def prepare_episode(
        self,
        q_static: jnp.ndarray,
        rng: np.random.Generator,
        theta_spec: "Optional[ThetaSpec]" = None,
    ) -> int:
        """Sample θ̃ ~ q(θ) for the upcoming episode.

        With ``theta_spec.episode_reset_components`` non-empty, we sample
        θ̃_persistent ~ q(θ_persistent) and store the persistent-axis
        values. ``plan()`` then reconditions the *current* q_static on
        θ̃_persistent at each step, so the inner planner sees a fresh
        ``q(θ_episodic | θ̃_persistent)`` derived from the live
        within-episode posterior — only θ̃_persistent is frozen, which
        is the correct PSRL adaptation when only persistent θ is the
        BA parameter. Door-code (no episodic components) hits the
        legacy flat-sample branch byte-identically.

        Returns the sampled flat θ̃ index for logging / inspection.
        """
        if theta_spec is None or not theta_spec.episode_reset_components:
            p = np.asarray(q_static, dtype=np.float64).reshape(-1)
            s = p.sum()
            if s <= 0:
                raise ValueError("q_static is not a valid distribution (sum <= 0).")
            p = p / s
            self._sample_idx = int(rng.choice(len(p), p=p))
            self._persistent_idx_tuple = None
            return self._sample_idx

        q_cond, theta_persistent_full = sample_persistent_theta(
            q_static, theta_spec, rng,
        )
        # theta_persistent_full has ints at persistent positions and
        # None at episodic positions. Filter to persistent values in
        # axis order (matches what _condition_on_persistent expects).
        self._persistent_idx_tuple = tuple(
            int(i) for i in theta_persistent_full if i is not None
        )
        flat = np.asarray(q_cond, dtype=np.float64).reshape(-1)
        self._sample_idx = int(np.argmax(flat))
        return self._sample_idx

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
        **kwargs,
    ) -> jnp.ndarray:
        if self._sample_idx is None:
            raise RuntimeError(
                "PSRLPlanner.prepare_episode() must be called before plan()."
            )
        if self._persistent_idx_tuple is not None:
            if theta_spec is None:
                raise RuntimeError(
                    "PSRLPlanner persistent-only path requires theta_spec "
                    "to be passed to plan()."
                )
            q = jnp.asarray(q_static).reshape(theta_spec.component_sizes)
            q_cond, _ = _condition_on_persistent(
                q, self._persistent_idx_tuple, theta_spec,
            )
            sharp = q_cond.astype(jnp.float32)
        else:
            n_static = int(np.prod(q_static.shape))
            if not 0 <= self._sample_idx < n_static:
                raise ValueError(
                    f"sampled theta_idx {self._sample_idx} out of range "
                    f"[0, {n_static})"
                )
            sharp = jnp.zeros(n_static, dtype=jnp.float32).at[self._sample_idx].set(1.0)
        return self._base.plan(
            q_state, sharp, T, B_list, goal, horizon,
            theta_mask=theta_mask,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
            **kwargs,
        )
