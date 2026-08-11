"""Persistent-oracle reference arm — knows θ_persistent, infers θ_episodic.

`OraclePlanner` collapses q(θ) onto the *full* truth, episodic components
included, so on RockSample it is handed the rock qualities for free and
SENSE actions become worthless to it. This planner is the intermediate
reference point between that and the agent: it pins only the persistent
components (RockSample's sensor-reliability pattern) to the truth and
leaves the episodic ones (rock qualities) to be inferred from
observations exactly as the agent does, so SENSE still pays off.

Mechanically this is `PSRLPlanner`'s persistent-only path with the
*sampling* step replaced by the truth: instead of drawing
θ̃_persistent ~ q(θ_persistent) once per episode, we take
θ_persistent = θ_true_persistent, and `plan()` reconditions the *live*
within-episode posterior on it at every step, yielding
q(θ_episodic | θ_true_persistent) for the inner solve.

**A reference arm, not an upper bound.** Handing an *approximate* planner
more information does not monotonically improve it, so nothing here bounds
anything: at `K=2, grid_size=3, horizon=5` the VBP-inner arm scores 0.38
against plain VBP's 0.50 (2 seeds x 2 episodes, mean normalized return) —
*below* a planner that knows strictly less — and it never gets above plain
VBP on mean cumulative reward at 2-12 seeds either. With the persistent
axes pinned it over-senses (4-5 SENSE per episode vs the full Oracle's
0-1) and burns the step budget, so the score says as much about how well
the base planner explores as about the information it was handed.

**Reference points, not a two-term decomposition.** `aif <
persistent_oracle < oracle` holds at every seed count measured (mean
cumulative reward 0.63 < 1.13 < 1.31 at 8 seeds x 2 episodes on that
config), but only the lower step is an interpretable quantity:

- `persistent_oracle - aif` is the **value of knowing θ_persistent**.
  Both arms run an AIF inner solve, so exactly one thing differs: the
  information handed to the planner. It is *not* "planner suboptimality"
  — that would be optimal(θ known) - planner(θ known), which no arm here
  measures.
- `oracle - persistent_oracle` is **not** the value of knowing
  θ_episodic, and should not be read as one: the inner solver changes
  along with the information, since `oracle` wraps VBP while
  `persistent_oracle` wraps AIF. At matched θ knowledge the solver effect
  alone is 1.125 (AIF inner) vs 0.563 (VBP inner) — a 0.56 swing against
  a 1.31 - 1.13 = 0.19 step, so the confound is ~3x the quantity it would
  reveal. Its direction is known: AIF beats VBP at matched knowledge, so
  the VBP-based `oracle` is a *depressed* endpoint and the contamination
  biases the step **downward** — the true value of knowing θ_episodic is
  likely larger than 0.19, not smaller. Reading it cleanly would need an
  `OraclePlanner(AIFPlanner)` arm, which is not in this slate.

`persistent_oracle` is the AIF-inner arm; `persistent_oracle_vbp` mirrors
`OraclePlanner`'s VBP base and is kept only as a diagnostic of that base.
`tests/test_persistent_oracle_planner.py -m slow` re-measures both.

**Only meaningful on rock_sample.** It is the one benchmark with both
persistent and episodic θ components. factored_tiger's θ is purely
persistent, and door_code / minigrid have a single persistent component,
so on all three "persistent-only" is the same as the full Oracle and
this planner is redundant — do not add it to their slates.

Runner contract: `rock_sample_runner._call_prepare_episode` inspects the
signature and passes `theta_true=` and `theta_spec=` independently.
`door_code_runner` (which also serves minigrid) passes only
`theta_true`, so this planner raises there — loudly, and it is
deliberately not wired on the benchmarks that runner serves.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
import numpy as np

from learningpomdp.inference.state_inference import _condition_on_persistent
from learningpomdp.planning.base import Planner

if TYPE_CHECKING:
    from learningpomdp.factors import ThetaSpec


class PersistentOraclePlanner:
    """Plans against q(θ_episodic | θ_true_persistent).

    Args:
        base_planner: model-based planner used for the inner solve. AIF
            is the reported reference arm; VBP mirrors `OraclePlanner`'s
            base. See the module docstring — the choice changes the
            number substantially, which is exactly why the AIF and VBP
            arms are not comparable endpoints.
    """

    def __init__(self, base_planner: Planner) -> None:
        self._base = base_planner
        # Persistent-axis values of θ_true, in axis order — what
        # `_condition_on_persistent` expects. None until prepare_episode.
        self._persistent_idx_tuple: Optional[tuple[int, ...]] = None

    def prepare_episode(
        self,
        q_static: jnp.ndarray,
        rng: np.random.Generator,
        theta_true: int | None = None,
        theta_spec: "Optional[ThetaSpec]" = None,
    ) -> None:
        if theta_true is None or theta_spec is None:
            raise RuntimeError(
                "PersistentOraclePlanner.prepare_episode requires both "
                "theta_true and theta_spec; the runner must pass them as "
                "kwargs (got theta_true="
                f"{theta_true!r}, theta_spec={theta_spec!r})."
            )
        components = theta_spec.unravel(int(theta_true))
        self._persistent_idx_tuple = tuple(
            components[k] for k in theta_spec.persistent_components
        )

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
        if self._persistent_idx_tuple is None:
            raise RuntimeError(
                "PersistentOraclePlanner.prepare_episode() must be called "
                "before plan()."
            )
        if theta_spec is None:
            raise RuntimeError(
                "PersistentOraclePlanner.plan() requires theta_spec."
            )
        q = jnp.asarray(q_static).reshape(theta_spec.component_sizes)
        # `_condition_on_persistent`'s zero-mass fallback is unreachable for
        # PSRL (which samples the persistent value from q, so it has mass)
        # but reachable here: we pin to the truth, and the filter's float32
        # softmax can underflow it to exactly 0. The fallback then hands the
        # inner solve a *uniform* episodic conditional, throwing away this
        # episode's SENSE evidence, so it must not pass silently. With no
        # episodic axes there is nothing to lose (q_cond is one-hot on the
        # truth either way), so only check when there are.
        episodic_axes = tuple(theta_spec.episode_reset_components)
        if episodic_axes and float(
            q.sum(axis=episodic_axes)[self._persistent_idx_tuple]
        ) <= 0.0:
            warnings.warn(
                "PersistentOraclePlanner: q(θ) has zero mass on the true "
                f"persistent value {self._persistent_idx_tuple} (filter "
                "underflow); falling back to a uniform q(θ_episodic), which "
                "discards the within-episode SENSE evidence for this step.",
                RuntimeWarning,
                stacklevel=2,
            )
        q_cond, _ = _condition_on_persistent(
            q, self._persistent_idx_tuple, theta_spec,
        )
        return self._base.plan(
            q_state, q_cond.astype(jnp.float32), T, B_list, goal, horizon,
            theta_mask=theta_mask,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
        )
