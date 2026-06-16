"""Goal/preference abstractions for Active Inference planning.

Supports two preference modes:
1. State preferences: C(x) or C(x, theta) - "prefer being in state x"
2. Observation preferences: C_obs(y) - "prefer observing y in modality k"

Both modes can be combined, and both integrate with BP planning via
terminal messages in the backward pass.
"""

from dataclasses import dataclass
from enum import Enum, auto

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from learningpomdp.factors import FactorSpec, ThetaSpec

from .messages import safe_log
from .bp_core import compute_obs_pref_to_x_msgs


class GoalType(Enum):
    """Type of goal specification."""
    STATE = auto()       # C_state only: prefer being in state(s)
    OBSERVATION = auto() # C_obs present: prefer observing certain outcomes


@dataclass(frozen=True)
class Goal:
    """Unified goal supporting state and/or observation preferences.

    Usage patterns:
        1. State-only (1D): Goal(C_state=jnp.array([...]))
           Shape (n_states,) - theta-independent state preferences.

        2. State-only (2D): Goal(C_state=jnp.array([[...]]))
           Shape (n_states, n_static) - state preferences per theta.

        3. Observation-only: Goal(C_obs={channel: jnp.array([...])})
           Dict mapping channel index to (n_obs,) log-preferences.

        4. Combined: Goal(C_state=..., C_obs=...)
           Both state and observation preferences active.

    All preference tensors should be in log-space (unnormalized log-probs).
    The planner will normalize as needed.

    Attributes:
        C_state: State preferences. Shape (n_states,) or (n_states, n_static).
                 None if using observation preferences only.
        C_obs: Observation preferences. Dict mapping channel index to
               (n_obs_types,) array of log-preferences.
               None if using state preferences only.
    """
    C_state: jnp.ndarray | None = None
    C_obs: dict[int, jnp.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.C_state is None and self.C_obs is None:
            raise ValueError("Goal must have at least C_state or C_obs")

        if self.C_state is not None:
            if self.C_state.ndim not in (1, 2):
                raise ValueError(
                    f"C_state must be 1D or 2D, got shape {self.C_state.shape}"
                )

        if self.C_obs is not None:
            for ch, pref in self.C_obs.items():
                if not isinstance(ch, int):
                    raise ValueError(f"C_obs keys must be int, got {type(ch)}")
                if pref.ndim != 1:
                    raise ValueError(
                        f"C_obs[{ch}] must be 1D, got shape {pref.shape}"
                    )

    @property
    def goal_type(self) -> GoalType:
        """Determine the goal type based on which preferences are set."""
        if self.C_state is not None and self.C_obs is None:
            return GoalType.STATE
        return GoalType.OBSERVATION  # C_obs present (possibly with C_state)

    @property
    def has_theta_dependency(self) -> bool:
        """Whether state preferences vary with theta."""
        return self.C_state is not None and self.C_state.ndim == 2

    @property
    def has_observation_preferences(self) -> bool:
        """Whether observation preferences are specified."""
        return self.C_obs is not None and len(self.C_obs) > 0

    @property
    def n_states(self) -> int | None:
        """Number of states (from C_state shape), or None if obs-only."""
        if self.C_state is None:
            return None
        return self.C_state.shape[0]

    @property
    def n_static(self) -> int | None:
        """Number of theta values (from 2D C_state), or None if 1D/obs-only."""
        if self.C_state is None or self.C_state.ndim == 1:
            return None
        return self.C_state.shape[1]

    @classmethod
    def from_state_preferences(
        cls,
        C_state: jnp.ndarray,
    ) -> "Goal":
        """Create a state-only goal from a preference array."""
        return cls(C_state=C_state)

    @classmethod
    def from_observation_preferences(
        cls,
        C_obs: dict[int, jnp.ndarray],
    ) -> "Goal":
        """Create an observation-only goal from channel preferences."""
        return cls(C_obs=C_obs)


def is_goal_object(goal: jnp.ndarray | Goal) -> bool:
    """Check if goal is a Goal object vs legacy array."""
    return isinstance(goal, Goal)


def compute_terminal_goal(log_C: jnp.ndarray, log_theta: jnp.ndarray) -> jnp.ndarray:
    """Compute terminal backward init by marginalizing C(x, theta) over theta.

    Args:
        log_C: (n_states, n_static) log preference.
        log_theta: (n_static,) log theta belief.

    Returns:
        (n_states,) log-normalized terminal goal.
    """
    terms = log_C + log_theta[None, :]
    log_goal = logsumexp(terms, axis=1)
    return log_goal - logsumexp(log_goal)


def resolve_goal(
    goal: Goal,
    B_list: list[jnp.ndarray],
    q_static: jnp.ndarray,
    theta_spec: ThetaSpec,
    b_factor_specs: tuple[FactorSpec, ...],
) -> jnp.ndarray:
    """Convert Goal object to raw goal array.

    Combines state preferences and observation preferences into a single
    state-space goal tensor. Spec-aware: requires `theta_spec` and
    `b_factor_specs` to describe the factor structure of θ.
    """
    n_states = B_list[0].shape[1]
    log_B_list = [safe_log(B_c) for B_c in B_list]

    # Build per-component log marginals. q_static may arrive flat
    # (n_static,) or already factored (*component_sizes,); reshape works
    # either way. Under the single-component default this is just the
    # original log q_static.
    q_static_factored = q_static.reshape(theta_spec.component_sizes)
    log_theta_components = []
    for k in range(theta_spec.n_components):
        marginal_axes = tuple(j for j in range(theta_spec.n_components) if j != k)
        if marginal_axes:
            marginal = q_static_factored.sum(axis=marginal_axes)
        else:
            marginal = q_static_factored
        log_theta_components.append(safe_log(marginal))
    log_theta_components = tuple(log_theta_components)

    # Start with zeros or C_state
    if goal.C_state is not None:
        goal_array = goal.C_state
    else:
        # Obs-only: start with uniform state preference
        if goal.C_obs is not None:
            goal_array = jnp.zeros(n_states)
        else:
            raise ValueError("Goal must have C_state or C_obs")

    # Add observation preference contribution
    if goal.C_obs is not None:
        log_obs_pref = compute_obs_pref_to_x_msgs(
            goal.C_obs, log_B_list, log_theta_components, b_factor_specs
        )
        if goal_array.ndim == 1:
            # 1D state goal: add obs pref directly
            goal_array = safe_log(goal_array) + log_obs_pref
            goal_array = jnp.exp(goal_array - logsumexp(goal_array))
        else:
            # 2D state goal: broadcast obs pref across theta
            log_C = safe_log(goal_array)
            combined = log_C + log_obs_pref[:, None]
            goal_array = jnp.exp(combined - logsumexp(combined, axis=0, keepdims=True))

    return goal_array
