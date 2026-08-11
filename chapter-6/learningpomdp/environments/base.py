"""Base environment abstractions for POMDP experiments."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from learningpomdp.factors import FactorSpec, ThetaSpec

if TYPE_CHECKING:
    from learningpomdp.observations import ModalityRegistry


# Smallest goal-prior entry the planners can still distinguish: they read
# log C through planning.messages.safe_log, which clamps at
# log(max(C, 1e-30)), so every entry below this floor collapses onto one
# value. The factored_tiger / rock_sample create_goal functions, whose
# logits scale with K, refuse to return a goal that crosses it.
SAFE_LOG_FLOOR = 1e-30


@dataclass
class StepResult:
    """Result of a single environment step.

    Attributes:
        obs: Observation vector. Concatenation of all observation channels.
        reward: Scalar reward (kept for backward compatibility and logging).
        terminated: True if the episode ended (goal reached, failure, etc.).
        truncated: True if the episode was cut short by a step limit.
        obs_per_channel: Optional dict mapping channel name -> observation.
                         Provided when environment has a modality registry.
    """
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    obs_per_channel: dict[str, np.ndarray] | None = field(default=None)


@runtime_checkable
class Environment(Protocol):
    """Protocol that all POMDP environments must satisfy.

    Tensor conventions:
        T: (n_states_new, n_states_old, *t_factor_spec.shape_for(theta_spec), n_actions)
        B_list: List of per-modality tensors with natural shapes:
          - Theta-independent: (n_obs, n_states)
          - Theta-dependent: (n_obs, n_states, *b_factor_spec.shape_for(theta_spec))

    For the M1/M2 single-component case (theta_spec.component_sizes = (K!,))
    and all theta-dependent factors having theta_components=(0,), these reduce
    to the legacy `(..., n_static, ...)` shapes.
    """

    n_states: int
    n_actions: int
    n_static: int
    theta_spec: ThetaSpec
    t_factor_spec: FactorSpec
    b_factor_specs: tuple[FactorSpec, ...]

    @property
    def start_state_index(self) -> int:
        """Index of the canonical START state (where reset() places the agent)."""
        ...

    @property
    def success_state_index(self) -> int:
        """Index of the canonical SUCCESS terminal state."""
        ...

    def reset(self, seed: int | None = None) -> StepResult: ...

    def step(self, action: int) -> StepResult: ...

    def get_transition_tensor(self) -> np.ndarray:
        """Return T with shape (n_states, n_states, n_static, n_actions)."""
        ...

    def get_observation_tensors(self) -> list[np.ndarray]:
        """Return list of B tensors, one per observation modality.

        Each tensor has its natural shape:
          - Theta-independent channels: (n_obs, n_states)
          - Theta-dependent channels: (n_obs, n_states, n_static)
        """
        ...

    def get_goal(self) -> np.ndarray:
        """Return goal preferences with shape (n_states, n_static)."""
        ...

    def get_modality_registry(self) -> "ModalityRegistry | None":
        """Return modality metadata, or None for legacy environments.

        When provided, enables theta-aware inference and learning.
        """
        ...
