"""BP-marginal agent: the within-MP ablation for chapter 4.

Identical to `NuijtenAgent` (same dense Bayesian state inference, same generative-
model adapter, argmax action selection) except it plans with chapter-5's
`region_extended_loopy_bp_planning` instead of `nuijten_mp_planning`.

This is Nuijten-MP with the epistemics switched off: the region-graph machinery is
the same, but action selection comes from goal-conditioned *marginal* inference
against the preference factor C(x, θ) rather than from the EFE-based action-prior
fixed point. It is the KL-control / pragmatic-only leg of the comparison — the
direct analogue of chapter 3's marginal objective, realized in message passing.

Because it carries no EFE ambiguity term, on the stochastic maze it has nothing
biasing it away from high-observation-noise states: it pursues the goal greedily
and gambles through the stochastic interior, whereas Nuijten-MP routes along the
zero-noise safe corridor. That contrast is the ablation result.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp
from jax import nn
from jax.scipy.special import logsumexp

from src.inference.messages import safe_log
from src.inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from src.adapters.generative_model import GenerativeModel
from src.adapters.state_inference import infer_state


@dataclass(frozen=True)
class BPMarginalAgent:
    """Active-inference agent using marginal loopy-BP planning (epistemics off)."""
    model: GenerativeModel
    q_state: jnp.ndarray       # (n_states,)
    q_static: jnp.ndarray      # (n_theta,)
    planning_horizon: int
    planning_iterations: int
    last_action: int
    terminal_goal_only: bool
    damping: float

    @staticmethod
    def create(
        model: GenerativeModel,
        planning_horizon: int,
        planning_iterations: int,
        terminal_goal_only: bool = False,
        damping: float = 1.0,
    ) -> "BPMarginalAgent":
        q_state = jnp.zeros(model.n_states).at[model.start_state_idx].set(1.0)
        return BPMarginalAgent(
            model=model,
            q_state=q_state,
            q_static=jnp.asarray(model.theta_prior),
            planning_horizon=planning_horizon,
            planning_iterations=planning_iterations,
            last_action=-1,
            terminal_goal_only=terminal_goal_only,
            damping=damping,
        )

    def _planning_goal(self, q_static: jnp.ndarray) -> jnp.ndarray:
        """2D preference C(x,theta), or its 1D marginal (terminal-only) under q_static.

        With ``terminal_goal_only=False`` the 2D goal is passed straight through and
        the loopy-BP planner applies it as a per-step preference factor at every
        timestep (textbook KL-control). With ``terminal_goal_only=True`` it is
        marginalised under the theta belief and bound to the final node, mirroring
        the Nuijten agent so the only thing that differs between them is the planner.
        """
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
            (action: int, updated_agent: BPMarginalAgent)
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
        action_dist, _, _ = region_extended_loopy_bp_planning(
            q_state, q_static,
            self.model.transition_tensor,
            self.model.planning_observation_tensor,
            self._planning_goal(q_static),
            horizon=horizon,
            n_iterations=self.planning_iterations,
            damping=self.damping,
            action_prior=self.model.action_prior,
        )
        action = int(jnp.argmax(action_dist))

        updated = replace(self, q_state=q_state, q_static=q_static, last_action=action)
        return action, updated
