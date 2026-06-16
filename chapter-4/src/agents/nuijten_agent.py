"""Nuijten-MP agent: plans in a chapter-3 environment via message passing.

Mirrors chapter-5's `_RockSampleAgentBase` / `RockSampleNuijtenMPAgent`: dense
Bayesian state inference between steps, then `nuijten_mp_planning` for action
selection. Generic over both the T-Maze and the reactivity maze via the
`GenerativeModel` adapter.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp
from jax import nn
from jax.scipy.special import logsumexp

from src.inference.messages import safe_log
from src.inference.nuijten_mp import nuijten_mp_planning
from src.adapters.generative_model import GenerativeModel
from src.adapters.state_inference import infer_state


@dataclass(frozen=True)
class NuijtenAgent:
    """Active-inference agent using the Nuijten-MP planner."""
    model: GenerativeModel
    q_state: jnp.ndarray       # (n_states,)
    q_static: jnp.ndarray      # (n_theta,)
    planning_horizon: int
    planning_iterations: int
    last_action: int
    terminal_goal_only: bool

    @staticmethod
    def create(
        model: GenerativeModel,
        planning_horizon: int,
        planning_iterations: int,
        terminal_goal_only: bool = False,
    ) -> "NuijtenAgent":
        q_state = jnp.zeros(model.n_states).at[model.start_state_idx].set(1.0)
        return NuijtenAgent(
            model=model,
            q_state=q_state,
            q_static=jnp.asarray(model.theta_prior),
            planning_horizon=planning_horizon,
            planning_iterations=planning_iterations,
            last_action=-1,
            terminal_goal_only=terminal_goal_only,
        )

    def _planning_goal(self, q_static: jnp.ndarray) -> jnp.ndarray:
        """2D preference C(x,theta), or its 1D marginal (terminal-only) under q_static."""
        if self.terminal_goal_only and self.model.goal.ndim == 2:
            log_C = safe_log(self.model.goal)
            log_pref = logsumexp(log_C + safe_log(q_static)[None, :], axis=1)
            return nn.softmax(log_pref)
        return self.model.goal

    def step(self, obs: jnp.ndarray, time_remaining: int):
        """One agent step.

        Args:
            obs: (n_channels,) per-channel categorical outcome indices
            time_remaining: steps left in the episode (for receding horizon)

        Returns:
            (action: int, updated_agent: NuijtenAgent)
        """
        n_actions = self.model.n_actions
        if self.last_action >= 0:
            action_onehot = jnp.zeros(n_actions).at[self.last_action].set(1.0)
        else:
            action_onehot = jnp.ones(n_actions) / n_actions

        q_state, q_static = infer_state(
            self.q_state, self.q_static,
            self.model.transition_tensor, self.model.observation_tensor,
            obs, action_onehot,
        )

        horizon = min(time_remaining, self.planning_horizon)
        action_dist, _, _ = nuijten_mp_planning(
            q_state, q_static,
            self.model.transition_tensor,
            self.model.planning_observation_tensor,
            self._planning_goal(q_static),
            horizon=horizon,
            n_iterations=self.planning_iterations,
            action_prior=self.model.action_prior,
        )
        action = int(jnp.argmax(action_dist))

        updated = replace(self, q_state=q_state, q_static=q_static, last_action=action)
        return action, updated
