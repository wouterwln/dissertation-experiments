"""Base planner abstraction."""

from typing import Protocol

import jax.numpy as jnp


class Planner(Protocol):
    """Protocol for action-selection planners.

    A planner takes current beliefs and model tensors and returns
    an action distribution.
    """

    def plan(
        self,
        q_state: jnp.ndarray,
        q_static: jnp.ndarray,
        T: jnp.ndarray,
        B: jnp.ndarray,
        goal: jnp.ndarray,
        horizon: int,
    ) -> jnp.ndarray:
        """Return (n_actions,) action distribution."""
        ...
