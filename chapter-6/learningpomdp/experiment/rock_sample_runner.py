"""Per-episode multi-block runner for the RockSample BA-POMDP.

Distinct from the door_code runner because RockSample's contract differs:

- θ has both persistent and episode-reset components (sensor reliability
  vs rock quality). At each episode reset within a block, the
  episode_reset_components of the env's θ are re-sampled uniformly,
  matching the q(θ) reset performed on the agent's posterior.
- Reward is non-sparse: each successful SAMPLE contributes; the metric
  of interest is `cumulative_reward` over the episode, not a terminal
  ±1.
- There is no probe modality and no canonical SUCCESS state. The
  spatial-epistemic metrics here are sense counts and successful sample
  counts, not "probe visits".

Persistent across the block: q(θ_sensor), the env's θ_sensor components.
Per-episode reset: q(θ_quality), the env's θ_quality components.
"""

from __future__ import annotations

import inspect
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from learningpomdp.environments.rock_sample import (
    RockSampleEnv,
    action_role,
    decode_state,
)
from learningpomdp.experiment.door_code_runner import build_derived_metrics
from learningpomdp.factors import ThetaSpec
from learningpomdp.inference.state_inference import (
    bayesian_state_update,
    reset_episode_theta,
)
from learningpomdp.planning import GPU_PLANNER_TYPES
from learningpomdp.planning.base import Planner


def _call_plan(planner: Planner, *args, **kwargs) -> jnp.ndarray:
    """Filter kwargs to those the planner's plan() declares. Lets the
    runner pass ``registry``, ``reward_fn``, ``terminal_fn`` to BAMCP/
    BOSS without breaking AIF/VBP/Oracle/Random whose plan() doesn't
    accept them."""
    sig = inspect.signature(planner.plan)
    accepted = set(sig.parameters.keys())
    safe = {k: v for k, v in kwargs.items() if k in accepted}
    return planner.plan(*args, **safe)


def _build_rock_sample_reward_fn(env: RockSampleEnv) -> Callable[..., float]:
    """Closure mirroring ``RockSampleEnv.step`` reward logic, parameterised
    by (state, action, next_state, theta_components). Used by sample-based
    planners' rollouts to compute per-step reward without touching env state.
    """
    K = env.K
    n_events = env.layout.n_events
    rock_set = set(env.layout.rock_cells)
    cell_to_rock_idx = {c: i for i, c in enumerate(env.layout.rock_cells)}
    sample_reward = env.sample_reward

    def reward_fn(state: int, action: int, next_state: int, theta_components: tuple) -> float:
        pos, mask, _ = decode_state(state, K, n_events)
        role, _ = action_role(action, K)
        if role == "sample" and pos in rock_set:
            r = cell_to_rock_idx[pos]
            if not ((mask >> r) & 1):
                quality_r = theta_components[r]
                return sample_reward if quality_r == 1 else -sample_reward
        return 0.0

    return reward_fn


def _build_rock_sample_reward_table(env: RockSampleEnv) -> np.ndarray:
    """Vectorised reward table ``[n_states, n_actions, n_static]``.

    Equivalent to calling ``_build_rock_sample_reward_fn(env)`` over
    every (state, action, theta_flat) triple but ~1000× faster — uses
    numpy broadcasts instead of a Python triple loop. The Numba path
    needs this; the NumPy generic path can also use it via the same
    ``reward_table`` kwarg.
    """
    K = env.K
    n_events = env.layout.n_events
    n_states = env.n_states
    n_actions = env.n_actions
    n_static = env.n_static
    sample_reward = env.sample_reward
    rock_cells = env.layout.rock_cells
    cell_to_rock_idx = {c: i for i, c in enumerate(rock_cells)}

    # Decode (pos, mask) for each state in one pass.
    states = np.arange(n_states)
    block = (2 ** K) * n_events
    rest = states % block
    pos_per_state = states // block
    mask_per_state = rest // n_events

    # Action roles: only the SAMPLE action (action_idx == K + 4) ever
    # generates non-zero reward. Other actions contribute zero.
    sample_action = K + 4

    # For each (state, action_sample, theta_flat): is reward triggered?
    # Trigger: action == SAMPLE AND pos == rock_cell[r] AND mask[r] == 0,
    # for some rock r. Reward sign: + if theta_quality_r == 1, else -.
    table = np.zeros((n_states, n_actions, n_static), dtype=np.float64)
    component_sizes = env.theta_spec.component_sizes
    # Pre-decode theta_flat -> theta_components (just the K quality bits).
    theta_grid = np.array([
        np.unravel_index(t, component_sizes) for t in range(n_static)
    ], dtype=np.int32)  # (n_static, 2*K)
    quality_bits = theta_grid[:, :K]  # (n_static, K)

    for s in range(n_states):
        pos = int(pos_per_state[s])
        if pos not in cell_to_rock_idx:
            continue
        r = cell_to_rock_idx[pos]
        mask_bit = (int(mask_per_state[s]) >> r) & 1
        if mask_bit == 1:
            continue
        # quality_bits[:, r] gives 0/1 for each θ; reward = ±sample_reward.
        sign = quality_bits[:, r] * 2 - 1  # 0 -> -1, 1 -> +1
        table[s, sample_action, :] = sign * sample_reward
    return table


def _entropy(p: np.ndarray) -> float:
    safe = p[p > 0]
    return float(-np.sum(safe * np.log(safe)))


def _start_q_state_for(env: RockSampleEnv) -> jnp.ndarray:
    return jnp.zeros(env.n_states, dtype=jnp.float32).at[env.start_state_index].set(1.0)


def _select_action(
    action_dist: jnp.ndarray, deterministic: bool, rng: np.random.Generator,
) -> int:
    p = np.asarray(action_dist, dtype=np.float64)
    if deterministic:
        return int(p.argmax())
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def _call_prepare_episode(
    planner: Planner,
    q_static: jnp.ndarray,
    rng: np.random.Generator,
    theta_true: int,
    theta_spec: ThetaSpec,
) -> None:
    """Dispatch on the planner's `prepare_episode` signature.

    Pass ``theta_true`` only to planners that declare it (Oracle).
    Pass ``theta_spec`` only to planners that declare it (PSRL/BOSS in
    persistent-only sampling mode). Other planners get the legacy
    ``(q_static, rng)`` call. Explicit signature inspection so
    unexpected TypeErrors from a planner's body are not silently
    swallowed.
    """
    if not hasattr(planner, "prepare_episode"):
        return
    sig = inspect.signature(planner.prepare_episode)
    extra: dict = {}
    if "theta_true" in sig.parameters:
        extra["theta_true"] = theta_true
    if "theta_spec" in sig.parameters:
        extra["theta_spec"] = theta_spec
    planner.prepare_episode(q_static, rng, **extra)


# --- Per-component posterior summaries --------------------------------------


def _marginalize(q: np.ndarray, keep_axes: tuple[int, ...]) -> np.ndarray:
    """Sum over all axes except `keep_axes`, preserving the order of keep_axes."""
    drop = tuple(i for i in range(q.ndim) if i not in keep_axes)
    if drop:
        q = q.sum(axis=drop)
    # If keep_axes is not in ascending order we need to transpose; the
    # post-sum array has axes in ascending order of keep_axes, so when
    # keep_axes is sorted no transpose is needed. We require the caller
    # to pass keep_axes in ascending order to keep this contract simple.
    return q


def _posterior_summaries(
    q_static: jnp.ndarray,
    theta_spec: ThetaSpec,
    episode_theta_components: tuple[int, ...],
) -> dict[str, float]:
    """Compute joint, persistent-marginal, and episodic-marginal summaries.

    Returns a dict with:
      - posterior_mass_on_episode_theta : full-joint mass at the episode's
            true θ index.
      - posterior_mass_on_persistent_theta : persistent-marginal mass at
            the episode's persistent components of θ.
      - posterior_entropy : H(q(θ)) full joint.
      - posterior_entropy_persistent : H(q(θ_persistent)).
      - posterior_entropy_episodic : H(q(θ_episodic)).
    """
    q = np.asarray(q_static).reshape(theta_spec.component_sizes)
    q_flat = q.reshape(-1)
    ep_flat_idx = int(np.ravel_multi_index(
        episode_theta_components, theta_spec.component_sizes,
    ))

    persistent_axes = tuple(sorted(theta_spec.persistent_components))
    episodic_axes = tuple(sorted(theta_spec.episode_reset_components))

    if persistent_axes:
        q_persistent = _marginalize(q, persistent_axes)
        persistent_components = tuple(
            episode_theta_components[c] for c in persistent_axes
        )
        persistent_sizes = tuple(
            theta_spec.component_sizes[c] for c in persistent_axes
        )
        persistent_idx = int(np.ravel_multi_index(
            persistent_components, persistent_sizes,
        ))
        mass_persistent = float(q_persistent.reshape(-1)[persistent_idx])
        entropy_persistent = _entropy(q_persistent.reshape(-1))
    else:
        mass_persistent = 1.0
        entropy_persistent = 0.0

    if episodic_axes:
        q_episodic = _marginalize(q, episodic_axes)
        entropy_episodic = _entropy(q_episodic.reshape(-1))
    else:
        entropy_episodic = 0.0

    return {
        "posterior_mass_on_episode_theta": float(q_flat[ep_flat_idx]),
        "posterior_mass_on_persistent_theta": mass_persistent,
        "posterior_entropy": _entropy(q_flat),
        "posterior_entropy_persistent": entropy_persistent,
        "posterior_entropy_episodic": entropy_episodic,
    }


# --- Result containers ------------------------------------------------------


@dataclass(frozen=True)
class RockSampleEpisodeResult:
    n_steps: int
    actions: tuple[int, ...]
    states_visited: tuple[int, ...]
    observations: tuple[tuple[float, ...], ...]
    initial_q_static: np.ndarray
    final_q_static: np.ndarray
    cumulative_reward: float
    n_senses: int
    n_samples_successful: int
    n_good_sampled: int
    n_bad_sampled: int
    n_good_in_episode: int
    max_possible_reward: float
    episode_theta_true: int
    posterior_mass_on_episode_theta: float
    posterior_mass_on_persistent_theta: float
    posterior_entropy: float
    posterior_entropy_persistent: float
    posterior_entropy_episodic: float
    planning_time_s: float
    terminated_early: bool


@dataclass(frozen=True)
class RockSampleBlockResult:
    K: int
    grid_size: int
    theta_true: int
    n_episodes: int
    episodes: tuple[RockSampleEpisodeResult, ...]


# --- Single-episode driver --------------------------------------------------


def run_episode(
    env: RockSampleEnv,
    planner: Planner,
    q_static_init: jnp.ndarray,
    episode_theta_true: int,
    horizon: int,
    max_steps: int,
    deterministic: bool,
    rng: np.random.Generator,
    clip_horizon: bool = False,
) -> RockSampleEpisodeResult:
    """Run one RockSample episode with the env's θ pinned to
    `episode_theta_true` and the agent's prior at `q_static_init`.

    Plan-act-filter ordering matches the door_code runner: plan, act,
    update belief from the resulting observation. RockSample has no
    `learn_on_failure` flag because there is no terminal failure state
    — every step contributes to the posterior.
    """
    env.reset(theta=episode_theta_true)
    theta_spec = env.theta_spec
    _call_prepare_episode(
        planner, q_static_init, rng, episode_theta_true, theta_spec,
    )

    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32) for B_c in env.get_observation_tensors()
    ]
    # Precompute log(B) once per episode so the per-step Bayesian filter
    # doesn't recompute it on every call. Tuple — JAX treats it as a stable
    # PyTree key so the jitted filter doesn't retrace per call. EPSILON
    # matches state_inference.py.
    log_B_list = tuple(jnp.log(B_c + 1e-12) for B_c in B_list)
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    registry = env.get_modality_registry()
    theta_mask = registry.theta_dependent_mask
    t_factor_spec = env.t_factor_spec
    b_factor_specs = env.b_factor_specs
    n_actions = env.n_actions
    K = env.K
    n_events = env.layout.n_events
    reward_fn = _build_rock_sample_reward_fn(env)
    # Rock-sample is bounded-horizon (no terminal-state observation);
    # terminal_fn=None tells BAMCP/BOSS to never short-circuit a rollout.
    terminal_fn = None
    # Precompute reward_table once per episode so the Numba path doesn't
    # rebuild it per plan() call (a 30s+ saving at K=3+ where the table
    # is a 1000×8×64 = 512k-cell triple loop). Vectorised builder.
    reward_table = _build_rock_sample_reward_table(env)

    q_state = _start_q_state_for(env)
    q_static = q_static_init
    initial_q_static_np = np.asarray(q_static_init).copy()

    actions: list[int] = []
    states_visited: list[int] = [env.agent_state]
    observations: list[list[float]] = []
    cumulative_reward = 0.0
    n_senses = 0
    n_samples_successful = 0
    n_good_sampled = 0
    n_bad_sampled = 0
    planning_time_s = 0.0
    terminated = False

    for step in range(max_steps):
        if clip_horizon:
            effective_horizon = min(horizon, max_steps - step)
        else:
            effective_horizon = horizon
        _t0 = time.perf_counter()
        action_dist = _call_plan(
            planner, q_state, q_static, T, B_list, goal, effective_horizon,
            theta_mask=theta_mask,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
            registry=registry,
            reward_fn=reward_fn,
            terminal_fn=terminal_fn,
            reward_table=reward_table,
        )
        if hasattr(action_dist, "block_until_ready"):
            action_dist.block_until_ready()
        planning_time_s += time.perf_counter() - _t0
        action = _select_action(action_dist, deterministic, rng)

        # Track action role for inner-loop epistemic reporting. Sample
        # success is detected by mask change, not by reward sign (a "good"
        # rock sample and a no-op SAMPLE both have non-negative reward).
        pre_pos, pre_mask, _ = decode_state(env.agent_state, K, n_events)
        role, _ = action_role(action, K)

        env_step = env.step(action)
        actions.append(action)
        states_visited.append(env.agent_state)
        cumulative_reward += env_step.reward

        if role == "sense":
            n_senses += 1
        elif role == "sample":
            _, post_mask, _ = decode_state(env.agent_state, K, n_events)
            if post_mask != pre_mask:
                n_samples_successful += 1
                # Reward sign disambiguates good (+sample_reward) from
                # bad (-sample_reward); no-op SAMPLEs (no mask change)
                # are excluded by the outer guard.
                if env_step.reward > 0:
                    n_good_sampled += 1
                else:
                    n_bad_sampled += 1

        obs = jnp.asarray(env_step.obs, dtype=jnp.float32)
        observations.append(np.asarray(env_step.obs, dtype=np.float32).tolist())
        action_onehot = jnp.zeros(n_actions, dtype=jnp.float32).at[action].set(1.0)
        q_state, q_static = bayesian_state_update(
            q_state, q_static, T, log_B_list, obs, action_onehot,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
        )

        if hasattr(planner, "post_step_update"):
            planner.post_step_update(
                q_state, q_static, env_step.reward, env_step.terminated,
            )

        if env_step.terminated:
            terminated = True
            break

    q_static_np = np.asarray(q_static)
    ep_components = tuple(
        int(c) for c in np.unravel_index(
            episode_theta_true, theta_spec.component_sizes,
        )
    )
    summaries = _posterior_summaries(q_static_np, theta_spec, ep_components)

    # ep_components layout: (quality_0..quality_K-1, sensor_0..sensor_K-1).
    # Episodic axes are the K quality bits; sum gives n_good rocks present
    # this episode. Optimal reward = sample every good, sample no bad.
    n_good_in_episode = int(sum(ep_components[:K]))
    max_possible_reward = float(n_good_in_episode * env.sample_reward)

    return RockSampleEpisodeResult(
        n_steps=len(actions),
        actions=tuple(actions),
        states_visited=tuple(states_visited),
        observations=tuple(tuple(o) for o in observations),
        initial_q_static=initial_q_static_np,
        final_q_static=q_static_np.copy(),
        cumulative_reward=float(cumulative_reward),
        n_senses=n_senses,
        n_samples_successful=n_samples_successful,
        n_good_sampled=n_good_sampled,
        n_bad_sampled=n_bad_sampled,
        n_good_in_episode=n_good_in_episode,
        max_possible_reward=max_possible_reward,
        episode_theta_true=int(episode_theta_true),
        planning_time_s=planning_time_s,
        terminated_early=bool(terminated),
        **summaries,
    )


# --- Block driver -----------------------------------------------------------


def _resample_episode_theta(
    block_components: list[int],
    theta_spec: ThetaSpec,
    rng: np.random.Generator,
) -> int:
    """Re-sample the episode_reset_components of θ; persistent components
    stay at their block-level values. Returns the flat θ index."""
    if not theta_spec.episode_reset_components:
        return int(np.ravel_multi_index(
            tuple(block_components), theta_spec.component_sizes,
        ))
    ep_components = list(block_components)
    for c in theta_spec.episode_reset_components:
        ep_components[c] = int(rng.integers(0, theta_spec.component_sizes[c]))
    return int(np.ravel_multi_index(
        tuple(ep_components), theta_spec.component_sizes,
    ))


def run_block(
    env: RockSampleEnv,
    planner: Planner,
    block_theta_true: int,
    n_episodes: int,
    horizon: int = 3,
    max_steps: int = 30,
    deterministic: bool = True,
    seed: int = 0,
    initial_q_static: jnp.ndarray | None = None,
    clip_horizon: bool = False,
) -> RockSampleBlockResult:
    """Run n_episodes with the block's persistent θ components fixed by
    `block_theta_true`. Within the block:

      - q(θ_quality) is re-uniformed before each episode (M3
        ``reset_episode_theta``).
      - Env θ_quality components are re-sampled uniformly per episode.
      - q(θ_sensor) persists across episodes (the across-block
        epistemic story).
    """
    rng = np.random.default_rng(seed)
    theta_spec = env.theta_spec

    if initial_q_static is None:
        q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    else:
        q_static = jnp.asarray(initial_q_static, dtype=jnp.float32)

    block_components = list(
        int(c) for c in np.unravel_index(block_theta_true, theta_spec.component_sizes)
    )

    # Block-level planner setup, once before the episode loop (distinct
    # from prepare_episode, which fires per episode). BAMCP/BOSS reset
    # their rollout Q-hat here — learned within a block, dropped across
    # blocks. Planners without the hook are untouched.
    if hasattr(planner, "prepare_block"):
        planner.prepare_block(q_static, rng)

    episodes: list[RockSampleEpisodeResult] = []
    for _ep_idx in range(n_episodes):
        q_static = reset_episode_theta(q_static, theta_spec)
        ep_theta = _resample_episode_theta(block_components, theta_spec, rng)

        result = run_episode(
            env=env,
            planner=planner,
            q_static_init=q_static,
            episode_theta_true=ep_theta,
            horizon=horizon,
            max_steps=max_steps,
            deterministic=deterministic,
            rng=rng,
            clip_horizon=clip_horizon,
        )
        episodes.append(result)
        q_static = jnp.asarray(result.final_q_static, dtype=jnp.float32)

    return RockSampleBlockResult(
        K=env.K,
        grid_size=env.grid_size,
        theta_true=int(block_theta_true),
        n_episodes=n_episodes,
        episodes=tuple(episodes),
    )


# --- Outer-loop driver ------------------------------------------------------


@dataclass(frozen=True)
class RockSampleOuterLoopResult:
    config: dict[str, Any]
    n_seeds: int
    n_episodes: int
    theta_trues: tuple[int, ...]
    blocks: tuple[RockSampleBlockResult, ...]
    n_steps: np.ndarray
    cumulative_reward: np.ndarray
    n_senses: np.ndarray
    n_samples_successful: np.ndarray
    n_good_sampled: np.ndarray
    n_bad_sampled: np.ndarray
    n_good_in_episode: np.ndarray
    max_possible_reward: np.ndarray
    reward_normalised: np.ndarray
    posterior_mass_on_episode_theta: np.ndarray
    posterior_mass_on_persistent_theta: np.ndarray
    posterior_entropy: np.ndarray
    posterior_entropy_persistent: np.ndarray
    posterior_entropy_episodic: np.ndarray
    planning_time_s: np.ndarray

    def to_summary_dict(self) -> dict[str, Any]:
        per_episode = {
            "n_steps": self.n_steps.tolist(),
            "cumulative_reward": self.cumulative_reward.tolist(),
            "n_senses": self.n_senses.tolist(),
            "n_samples_successful": self.n_samples_successful.tolist(),
            "n_good_sampled": self.n_good_sampled.tolist(),
            "n_bad_sampled": self.n_bad_sampled.tolist(),
            "n_good_in_episode": self.n_good_in_episode.tolist(),
            "max_possible_reward": self.max_possible_reward.tolist(),
            "reward_normalised": self.reward_normalised.tolist(),
            "posterior_mass_on_episode_theta":
                self.posterior_mass_on_episode_theta.tolist(),
            "posterior_mass_on_persistent_theta":
                self.posterior_mass_on_persistent_theta.tolist(),
            "posterior_entropy": self.posterior_entropy.tolist(),
            "posterior_entropy_persistent":
                self.posterior_entropy_persistent.tolist(),
            "posterior_entropy_episodic":
                self.posterior_entropy_episodic.tolist(),
            "planning_time_s": self.planning_time_s.tolist(),
        }
        curves: dict[str, dict[str, list[float]]] = {}
        for name, arr in [
            ("n_steps", self.n_steps.astype(float)),
            ("cumulative_reward", self.cumulative_reward),
            ("n_senses", self.n_senses.astype(float)),
            ("n_samples_successful", self.n_samples_successful.astype(float)),
            ("n_good_sampled", self.n_good_sampled.astype(float)),
            ("n_bad_sampled", self.n_bad_sampled.astype(float)),
            ("n_good_in_episode", self.n_good_in_episode.astype(float)),
            ("max_possible_reward", self.max_possible_reward),
            ("reward_normalised", self.reward_normalised),
            ("posterior_mass_on_episode_theta",
                self.posterior_mass_on_episode_theta),
            ("posterior_mass_on_persistent_theta",
                self.posterior_mass_on_persistent_theta),
            ("posterior_entropy", self.posterior_entropy),
            ("posterior_entropy_persistent",
                self.posterior_entropy_persistent),
            ("posterior_entropy_episodic",
                self.posterior_entropy_episodic),
            ("planning_time_s", self.planning_time_s),
        ]:
            # nanmean/nanpercentile for reward_normalised: episodes with
            # n_good_in_episode == 0 yield NaN (no max_possible_reward to
            # divide by) and should not pull the average to 0 or inf.
            curves[name] = {
                "mean": np.nanmean(arr, axis=0).tolist(),
                "p05": np.nanpercentile(arr, 5, axis=0).tolist(),
                "p95": np.nanpercentile(arr, 95, axis=0).tolist(),
            }
        derived = build_derived_metrics(self.posterior_mass_on_persistent_theta)
        trajectories = [
            [
                {
                    "actions": list(ep.actions),
                    "observations": [list(o) for o in ep.observations],
                    "states_visited": list(ep.states_visited),
                    "initial_q_static": np.asarray(ep.initial_q_static).reshape(-1).tolist(),
                    "final_q_static": np.asarray(ep.final_q_static).reshape(-1).tolist(),
                    "episode_theta_true": int(ep.episode_theta_true),
                    "terminated_early": bool(ep.terminated_early),
                }
                for ep in block.episodes
            ]
            for block in self.blocks
        ]
        return {
            "config": self.config,
            "n_seeds": self.n_seeds,
            "n_episodes": self.n_episodes,
            "theta_trues": list(self.theta_trues),
            "per_episode": per_episode,
            "curves": curves,
            "derived": derived,
            "trajectories": trajectories,
        }


def _stack(blocks: list[RockSampleBlockResult], attr: str, dtype) -> np.ndarray:
    return np.array(
        [[getattr(ep, attr) for ep in b.episodes] for b in blocks],
        dtype=dtype,
    )


def _run_one_seed_worker(
    env_kwargs: dict[str, Any],
    planner_factory: Callable[[int], Planner],
    block_theta: int,
    n_episodes: int,
    horizon: int,
    max_steps: int,
    deterministic: bool,
    seed: int,
    clip_horizon: bool,
    seed_index: int,
) -> RockSampleBlockResult:
    """Subprocess entry point: builds a fresh env + planner, runs one
    seed's block. Spawned workers can't share the parent's planner
    instance (RNG state, JAX side effects), so this rebuilds locally."""
    env = RockSampleEnv(**env_kwargs)
    planner = planner_factory(seed_index)
    return run_block(
        env, planner, block_theta_true=block_theta, n_episodes=n_episodes,
        horizon=horizon, max_steps=max_steps, deterministic=deterministic,
        seed=seed, clip_horizon=clip_horizon,
    )


def run_outer_loop(
    env_kwargs: dict[str, Any],
    planner: Planner | None = None,
    n_seeds: int = 1,
    n_episodes: int = 1,
    horizon: int = 3,
    max_steps: int = 30,
    deterministic: bool = True,
    seed: int = 0,
    theta_seeds: list[int] | None = None,
    clip_horizon: bool = False,
    show_progress: bool = True,
    progress_desc: str = "rock-sample",
    env_factory: Callable[[dict[str, Any]], RockSampleEnv] | None = None,
    workers: int = 1,
    planner_factory: Optional[Callable[[int], Planner]] = None,
) -> RockSampleOuterLoopResult:
    """Run n_seeds blocks of n_episodes. Each block samples a fresh
    block-level θ and constructs a fresh env (independent obs RNG per
    seed). The block-level θ defines persistent components for the
    duration of the block; episodic components are re-sampled per
    episode within `run_block`.
    """
    if env_factory is None:
        env_factory = lambda kw: RockSampleEnv(**kw)

    if planner is None and planner_factory is None:
        raise ValueError("must provide either `planner` or `planner_factory`")

    # Decide whether to multiprocess. Force workers=1 for GPU planners
    # — spawning subprocesses would re-init JAX/CUDA per worker and OOM.
    # CPU-only planners (BAMCP/BOSS/Random) are safe to parallelise.
    if workers > 1:
        if planner_factory is None:
            raise ValueError(
                "`workers > 1` requires `planner_factory` since planner "
                "instances are not pickle-safe across processes."
            )
        # Probe planner type via the factory.
        probe = planner_factory(0)
        if isinstance(probe, GPU_PLANNER_TYPES):
            warnings.warn(
                f"Planner type {type(probe).__name__} uses JAX/GPU; "
                f"forcing workers=1 to avoid GPU re-init across spawned "
                f"workers.",
                stacklevel=2,
            )
            workers = 1

    # Pre-resolve block thetas so the parallel and sequential paths agree.
    rng = np.random.default_rng(seed)
    block_thetas: list[int] = []
    probe_env = env_factory(dict(env_kwargs))  # cheap; for n_static
    for s in range(n_seeds):
        if theta_seeds is not None:
            if len(theta_seeds) != n_seeds:
                raise ValueError(
                    f"theta_seeds length {len(theta_seeds)} != n_seeds {n_seeds}"
                )
            block_thetas.append(int(theta_seeds[s]))
        else:
            block_thetas.append(int(rng.integers(0, probe_env.n_static)))

    blocks: list[RockSampleBlockResult] = []
    theta_trues = list(block_thetas)

    if workers > 1:
        # Parallel path: ProcessPoolExecutor with spawn ctx so each worker
        # re-imports JAX cleanly. Each worker owns its own env + planner.
        ctx = multiprocessing.get_context("spawn")
        worker_kwargs_list = []
        for s in range(n_seeds):
            wk = dict(env_kwargs)
            wk["obs_seed"] = seed * 10_000 + s
            worker_kwargs_list.append(wk)

        iterator: Any = range(n_seeds)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc, unit="seed")

        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
        ) as pool:
            futures = [
                pool.submit(
                    _run_one_seed_worker,
                    worker_kwargs_list[s],
                    planner_factory,
                    block_thetas[s],
                    n_episodes,
                    horizon,
                    max_steps,
                    deterministic,
                    seed * 10_000 + s,
                    clip_horizon,
                    s,
                )
                for s in range(n_seeds)
            ]
            cum_reward_sum = 0.0
            cum_episodes = 0
            for s in iterator:
                block = futures[s].result()
                blocks.append(block)
                cum_reward_sum += sum(ep.cumulative_reward for ep in block.episodes)
                cum_episodes += n_episodes
                if show_progress:
                    iterator.set_postfix(
                        avg_rew=f"{cum_reward_sum / max(cum_episodes, 1):.2f}",
                    )
    else:
        # Sequential path. When `planner_factory` is provided, build a
        # fresh planner per seed so workers=1 is a true sequential
        # re-ordering of workers>1. The factory is handed the outer-seed
        # index `s`, which is what actually makes each seed's
        # `planner._rng` independent: a fresh instance alone is not
        # enough, since `default_rng(<same int>)` replays the same stream
        # and the per-plan() draw count is fixed by n_simulations. The
        # legacy `planner=` direct-instance API retains single-instance
        # reuse (and hence one shared stream — it is used only by tests).
        iterator = range(n_seeds)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc, unit="seed")
        cum_reward_sum = 0.0
        cum_episodes = 0
        for s in iterator:
            kwargs = dict(env_kwargs)
            kwargs["obs_seed"] = seed * 10_000 + s
            env = env_factory(kwargs)
            seed_planner = (
                planner_factory(s) if planner_factory is not None else planner
            )
            block = run_block(
                env, seed_planner, block_theta_true=block_thetas[s],
                n_episodes=n_episodes, horizon=horizon, max_steps=max_steps,
                deterministic=deterministic, seed=seed * 10_000 + s,
                clip_horizon=clip_horizon,
            )
            blocks.append(block)
            cum_reward_sum += sum(ep.cumulative_reward for ep in block.episodes)
            cum_episodes += n_episodes
            if show_progress:
                iterator.set_postfix(
                    avg_rew=f"{cum_reward_sum / max(cum_episodes, 1):.2f}",
                )

    config = {
        "env": env_kwargs,
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "horizon": horizon,
        "max_steps": max_steps,
        "deterministic": deterministic,
        "seed": seed,
        "clip_horizon": clip_horizon,
    }
    cumulative_reward = _stack(blocks, "cumulative_reward", float)
    max_possible_reward = _stack(blocks, "max_possible_reward", float)
    # Episodes with no good rocks have max_possible_reward == 0; emit
    # NaN there so they're excluded from the curve aggregates rather
    # than dragging the mean to ±∞.
    with np.errstate(divide="ignore", invalid="ignore"):
        reward_normalised = np.where(
            max_possible_reward > 0,
            cumulative_reward / max_possible_reward,
            np.nan,
        )
    return RockSampleOuterLoopResult(
        config=config,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        theta_trues=tuple(theta_trues),
        blocks=tuple(blocks),
        n_steps=_stack(blocks, "n_steps", int),
        cumulative_reward=cumulative_reward,
        n_senses=_stack(blocks, "n_senses", int),
        n_samples_successful=_stack(blocks, "n_samples_successful", int),
        n_good_sampled=_stack(blocks, "n_good_sampled", int),
        n_bad_sampled=_stack(blocks, "n_bad_sampled", int),
        n_good_in_episode=_stack(blocks, "n_good_in_episode", int),
        max_possible_reward=max_possible_reward,
        reward_normalised=reward_normalised,
        posterior_mass_on_episode_theta=_stack(
            blocks, "posterior_mass_on_episode_theta", float),
        posterior_mass_on_persistent_theta=_stack(
            blocks, "posterior_mass_on_persistent_theta", float),
        posterior_entropy=_stack(blocks, "posterior_entropy", float),
        posterior_entropy_persistent=_stack(
            blocks, "posterior_entropy_persistent", float),
        posterior_entropy_episodic=_stack(
            blocks, "posterior_entropy_episodic", float),
        planning_time_s=_stack(blocks, "planning_time_s", float),
    )
