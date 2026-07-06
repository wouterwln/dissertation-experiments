"""Reactivity Maze agents using Active Inference (message-passing) planning.

State inference uses location + cue observations to update beliefs over the
dynamic state (location, knob) and the hidden goal identity θ. Planning uses only
the θ-dependent cue channels (the location channels are θ-independent and carry no
information about the goal), mirroring RockSample's rock-vs-position channel split.

The reward is terminal (chapter-3 reactivity maze), so ``terminal_goal_only`` is the
intended mode: the θ-conditioned goal (n_states, n_static) is marginalized over the
current θ belief into a single terminal preference. Before the cue is visited θ is
uniform, so committing to any nav goal has no better expected preference than the
safe sink — the epistemic drive of the AIF-MP scheme is what pulls the agent to the
cue first.

Multiple planning methods are supported via separate agent classes; the ablation
focus is ``active-inference``.
"""

import jax.numpy as jnp
from jax import nn
from jax.scipy.special import logsumexp
from dataclasses import dataclass, replace

from inference.messages import safe_log
from inference.loopy_bp import loopy_bp_planning
from inference.region_extended_loopy_bp import region_extended_loopy_bp_planning
from inference.dyn_channel_loopy_bp import dyn_channel_loopy_bp_planning
from inference.nuijten_mp import nuijten_mp_planning
from inference.loopy_vbp import loopy_vbp_planning
from inference.vbp_channel import vbp_channel_planning
from inference.precise_info_seeking import precise_info_seeking_planning
from inference.active_inference import active_inference_planning

EPSILON = 1e-12


def _infer_state(
    q_current: jnp.ndarray,
    q_static: jnp.ndarray,
    transition_tensor: jnp.ndarray,
    observation_tensor: jnp.ndarray,
    obs: jnp.ndarray,
    action_onehot: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bayesian state and static-state update.

    1. Predict x_new from x_old via transition (marginalize θ with current belief).
    2. Incorporate the full observation likelihood to update both x and θ.

    Args:
        q_current: (n_states,) prior belief over state (before action)
        q_static: (n_static,) belief over goal identities
        transition_tensor: (n_states, n_states, n_static, n_actions)
        observation_tensor: (n_channels, n_obs_types, n_states, n_static) — full
        obs: (n_channels,) per-channel categorical outcome indices (0/1)
        action_onehot: (n_actions,) one-hot previous action

    Returns:
        q_new_state: (n_states,) posterior over state
        q_new_static: (n_static,) posterior over goal identity
    """
    n_states = q_current.shape[0]
    n_static = q_static.shape[0]
    n_channels = observation_tensor.shape[0]

    # Predict: P(x_new) = Σ_{x_old, θ, a} T(x_new|x_old,θ,a) · q(x_old) · q(θ) · a(a)
    predicted = jnp.einsum(
        "ijkl,j,k,l->i",
        transition_tensor, q_current, q_static, action_onehot,
    )
    predicted = predicted / (predicted.sum() + EPSILON)

    # Observation log-likelihood: Σ_c log P(obs_c | x, θ)
    obs_int = jnp.round(obs).astype(jnp.int32)
    log_lik_per_channel = jnp.log(
        observation_tensor[jnp.arange(n_channels), obs_int, :, :] + EPSILON
    )
    log_likelihood = log_lik_per_channel.sum(axis=0)

    # Joint posterior: P(x, θ | obs) ∝ P(obs | x, θ) · P(x) · P(θ)
    log_joint = (
        log_likelihood
        + jnp.log(predicted + EPSILON)[:, None]
        + jnp.log(q_static + EPSILON)[None, :]
    )

    log_q_x = jnp.logaddexp.reduce(log_joint, axis=1)
    q_new_state = nn.softmax(log_q_x)

    log_q_theta = jnp.logaddexp.reduce(log_joint, axis=0)
    q_new_static = nn.softmax(log_q_theta)

    return q_new_state, q_new_static


# ---------------------------------------------------------------------------
# Base agent with shared logic
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ReactivityMazeAgentBase:
    transition_tensor: jnp.ndarray   # (n_states, n_states, n_static, n_actions)
    observation_tensor: jnp.ndarray  # (n_location_channels + 1, n_theta+1, n_states, n_static)
    goal: jnp.ndarray               # (n_states, n_static)
    n_location_channels: int        # number of θ-independent location channels (split point)
    q_current_state: jnp.ndarray    # (n_states,)
    q_static_state: jnp.ndarray     # (n_static,)
    planning_horizon: int
    planning_iterations: int
    action_prior: jnp.ndarray | None
    last_action: int
    damping: float
    start_state_idx: int
    terminal_goal_only: bool

    @property
    def _cue_obs(self) -> jnp.ndarray:
        """θ-dependent cue observation channels (used for planning epistemics)."""
        return self.observation_tensor[self.n_location_channels:]

    def _planning_goal(self, q_static: jnp.ndarray) -> jnp.ndarray:
        """Goal for planning. If terminal_goal_only, marginalize 2D→1D over θ belief."""
        if self.terminal_goal_only and self.goal.ndim == 2:
            log_C = safe_log(self.goal)
            log_pref = logsumexp(log_C + safe_log(q_static)[None, :], axis=1)
            return nn.softmax(log_pref)
        return self.goal

    def reset(self):
        n_states = self.transition_tensor.shape[0]
        n_static = self.q_static_state.shape[0]
        q_start = jnp.zeros(n_states, dtype=jnp.float32).at[self.start_state_idx].set(1.0)
        return replace(self,
            q_current_state=q_start,
            q_static_state=jnp.ones(n_static) / n_static,
            last_action=-1,
        )

    def _plan(self, q_current, q_static, horizon):
        raise NotImplementedError

    def step(self, obs: jnp.ndarray, time_remaining: int):
        """Take a step given per-channel binary observations.

        Args:
            obs: (n_channels,) binary sensor readings
            time_remaining: steps left (for receding horizon)

        Returns:
            action: int
            updated_agent: agent with updated beliefs
        """
        n_actions = self.transition_tensor.shape[3]

        if self.last_action >= 0:
            action_onehot = jnp.zeros(n_actions).at[self.last_action].set(1.0)
        else:
            action_onehot = jnp.ones(n_actions) / n_actions

        q_current, q_static = _infer_state(
            self.q_current_state, self.q_static_state,
            self.transition_tensor, self.observation_tensor,
            obs, action_onehot,
        )

        horizon = min(time_remaining, self.planning_horizon)
        action_dist = self._plan(q_current, q_static, horizon)
        action = int(jnp.argmax(action_dist))

        return action, replace(self,
            q_current_state=q_current,
            q_static_state=q_static,
            last_action=action,
        )


# ---------------------------------------------------------------------------
# Method-specific agents
# ---------------------------------------------------------------------------

def _create(cls, transition_tensor, observation_tensor, goal,
            n_location_channels, start_state_idx,
            planning_horizon, planning_iterations, action_prior, damping=1.0,
            terminal_goal_only=False):
    """Shared factory for all Reactivity Maze agents."""
    n_states = transition_tensor.shape[0]
    n_static = transition_tensor.shape[2]
    return cls(
        transition_tensor=jnp.array(transition_tensor, dtype=jnp.float32),
        observation_tensor=jnp.array(observation_tensor, dtype=jnp.float32),
        goal=jnp.array(goal, dtype=jnp.float32),
        n_location_channels=n_location_channels,
        q_current_state=jnp.ones(n_states, dtype=jnp.float32) / n_states,
        q_static_state=jnp.ones(n_static, dtype=jnp.float32) / n_static,
        planning_horizon=planning_horizon,
        planning_iterations=planning_iterations,
        action_prior=jnp.array(action_prior, dtype=jnp.float32) if action_prior is not None else None,
        last_action=-1,
        damping=damping,
        start_state_idx=start_state_idx,
        terminal_goal_only=terminal_goal_only,
    )


@dataclass(frozen=True)
class ReactivityLoopyBPAgent(_ReactivityMazeAgentBase):
    """Loopy BP with θ as variable node."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityLoopyBPAgent, transition_tensor, observation_tensor,
                       goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        return loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )


@dataclass(frozen=True)
class ReactivityLoopyVBPAgent(_ReactivityMazeAgentBase):
    """Loopy VBP (value iteration, θ as variable node)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityLoopyVBPAgent, transition_tensor, observation_tensor,
                       goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        return loopy_vbp_planning(
            q_current, q_static, self.transition_tensor,
            self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
        )


@dataclass(frozen=True)
class ReactivityRegionExtendedAgent(_ReactivityMazeAgentBase):
    """Region-extended loopy BP with observation factors."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityRegionExtendedAgent, transition_tensor,
                       observation_tensor, goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = region_extended_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self._cue_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class ReactivityDynChannelAgent(_ReactivityMazeAgentBase):
    """Dyn-channel loopy BP (raw obs B + dyn channels, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityDynChannelAgent, transition_tensor,
                       observation_tensor, goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = dyn_channel_loopy_bp_planning(
            q_current, q_static, self.transition_tensor,
            self._cue_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class ReactivityNuijtenMPAgent(_ReactivityMazeAgentBase):
    """Nuijten MP (region beliefs + EFE, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityNuijtenMPAgent, transition_tensor,
                       observation_tensor, goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = nuijten_mp_planning(
            q_current, q_static, self.transition_tensor,
            self._cue_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior,
        )
        return action_dist


@dataclass(frozen=True)
class ReactivityVBPChannelAgent(_ReactivityMazeAgentBase):
    """VBP channel (action channel reparameterization, θ inferred)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityVBPChannelAgent, transition_tensor,
                       observation_tensor, goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _ = vbp_channel_planning(
            q_current, q_static, self.transition_tensor,
            self._cue_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class ReactivityPreciseInfoSeekingAgent(_ReactivityMazeAgentBase):
    """Precise info-seeking (VBP action channels + obs channels)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityPreciseInfoSeekingAgent, transition_tensor,
                       observation_tensor, goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = precise_info_seeking_planning(
            q_current, q_static, self.transition_tensor,
            self._cue_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


@dataclass(frozen=True)
class ReactivityActiveInferenceAgent(_ReactivityMazeAgentBase):
    """Active Inference (VBP action channels + dyn channels + obs channels)."""

    @staticmethod
    def create(transition_tensor, observation_tensor, goal,
               n_location_channels, start_state_idx,
               planning_horizon=5, planning_iterations=3, action_prior=None,
               damping=1.0, terminal_goal_only=False):
        return _create(ReactivityActiveInferenceAgent, transition_tensor,
                       observation_tensor, goal, n_location_channels, start_state_idx,
                       planning_horizon, planning_iterations, action_prior,
                       damping=damping, terminal_goal_only=terminal_goal_only)

    def _plan(self, q_current, q_static, horizon):
        action_dist, _, _ = active_inference_planning(
            q_current, q_static, self.transition_tensor,
            self._cue_obs, self._planning_goal(q_static),
            horizon=horizon, n_iterations=self.planning_iterations,
            action_prior=self.action_prior, damping=self.damping,
        )
        return action_dist


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_CLASSES = {
    "loopy_bp": ReactivityLoopyBPAgent,
    "loopy_vbp": ReactivityLoopyVBPAgent,
    "region_extended": ReactivityRegionExtendedAgent,
    "dyn_channel": ReactivityDynChannelAgent,
    "nuijten": ReactivityNuijtenMPAgent,
    "vbp_channel": ReactivityVBPChannelAgent,
    "precise_info_seeking": ReactivityPreciseInfoSeekingAgent,
    "active_inference": ReactivityActiveInferenceAgent,
}


def create_agent(
    method: str,
    transition_tensor,
    observation_tensor,
    goal,
    n_location_channels: int,
    start_state_idx: int,
    planning_horizon: int = 5,
    planning_iterations: int = 3,
    action_prior=None,
    damping: float = 1.0,
    terminal_goal_only: bool = False,
):
    """Create a Reactivity Maze agent for the given planning method."""
    if method not in AGENT_CLASSES:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(AGENT_CLASSES.keys())}")

    cls = AGENT_CLASSES[method]
    return cls.create(
        transition_tensor, observation_tensor, goal,
        n_location_channels, start_state_idx,
        planning_horizon, planning_iterations, action_prior,
        damping=damping, terminal_goal_only=terminal_goal_only,
    )
