"""Per-episode multi-block runner for the Factored Tiger BA-POMDP.

Mirrors `rock_sample_runner.py` with two differences:

- θ is purely persistent (`episode_reset_components=()`). The
  ``reset_episode_theta`` call within the block loop is a no-op for this
  env; the persistent posterior carries forward across episodes.
- Reward is denser than RockSample's: every LISTEN_k contributes a
  per-step cost and every unopened-door OPEN_k contributes
  ±open_reward. The metric of interest is ``cumulative_reward`` and the
  derived ``reward_normalised = cumulative / (n_safe · open_safe_reward)``.
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

from learningpomdp.environments.factored_tiger import (
    FactoredTigerEnv,
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
    """Filter kwargs to those the planner's plan() declares."""
    sig = inspect.signature(planner.plan)
    accepted = set(sig.parameters.keys())
    safe = {k: v for k, v in kwargs.items() if k in accepted}
    return planner.plan(*args, **safe)


def _build_factored_tiger_reward_fn(env: FactoredTigerEnv) -> Callable[..., float]:
    """Closure mirroring ``FactoredTigerEnv.step`` reward logic."""
    K = env.K
    listen_cost = env.listen_cost
    open_safe_reward = env.open_safe_reward
    open_tiger_penalty = env.open_tiger_penalty

    def reward_fn(
        state: int, action: int, next_state: int, theta_components: tuple,
    ) -> float:
        mask, _ = decode_state(state, K)
        role, payload = action_role(action, K)
        if role == "listen":
            return -listen_cost
        # open
        k = payload
        if (mask >> k) & 1:
            return 0.0
        theta_k = theta_components[k]
        return open_safe_reward if theta_k == 1 else -open_tiger_penalty

    return reward_fn


def _build_factored_tiger_reward_table(env: FactoredTigerEnv) -> np.ndarray:
    """Vectorised reward table ``[n_states, n_actions, n_static]``."""
    K = env.K
    n_states = env.n_states
    n_actions = env.n_actions
    n_static = env.n_static
    listen_cost = env.listen_cost
    open_safe_reward = env.open_safe_reward
    open_tiger_penalty = env.open_tiger_penalty

    n_events = 2 * K + 1
    states = np.arange(n_states)
    mask_per_state = states // n_events
    mask_bits = ((mask_per_state[:, None] >> np.arange(K)) & 1).astype(np.int32)
    # (n_static, K) — quality bits per θ index.
    component_sizes = (2,) * K
    theta_grid = np.array(
        [np.unravel_index(t, component_sizes) for t in range(n_static)],
        dtype=np.int32,
    )

    table = np.zeros((n_states, n_actions, n_static), dtype=np.float64)
    # LISTEN actions (0..K-1) — flat -listen_cost across (state, θ).
    table[:, :K, :] = -listen_cost
    # OPEN actions (K..2K-1).
    for k in range(K):
        action_open_k = K + k
        unopened = (mask_bits[:, k] == 0)  # (n_states,) bool
        theta_k_per_theta = theta_grid[:, k]  # (n_static,) ∈ {0,1}
        reward_per_theta = np.where(
            theta_k_per_theta == 1, open_safe_reward, -open_tiger_penalty,
        )  # (n_static,)
        table[unopened, action_open_k, :] = reward_per_theta[None, :]
    return table


def _entropy(p: np.ndarray) -> float:
    safe = p[p > 0]
    return float(-np.sum(safe * np.log(safe)))


def _start_q_state_for(env: FactoredTigerEnv) -> jnp.ndarray:
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
    Pass ``theta_spec`` only to planners that declare it (PSRL/BOSS).
    Other planners get the legacy ``(q_static, rng)`` call.
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
    drop = tuple(i for i in range(q.ndim) if i not in keep_axes)
    if drop:
        q = q.sum(axis=drop)
    return q


def _posterior_summaries(
    q_static: jnp.ndarray,
    theta_spec: ThetaSpec,
    episode_theta_components: tuple[int, ...],
) -> dict[str, float]:
    """Compute joint, persistent-marginal, and episodic-marginal summaries.

    For Factored Tiger ``episode_reset_components=()``, so the episodic
    marginal is trivial (entropy 0, mass 1) and persistent = joint.
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
class FactoredTigerEpisodeResult:
    n_steps: int
    actions: tuple[int, ...]
    states_visited: tuple[int, ...]
    observations: tuple[tuple[float, ...], ...]
    initial_q_static: np.ndarray
    final_q_static: np.ndarray
    cumulative_reward: float
    n_listens: int
    n_opens: int
    n_safe_opens: int
    n_tiger_opens: int
    n_safe_in_episode: int
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
class FactoredTigerBlockResult:
    K: int
    theta_true: int
    n_episodes: int
    episodes: tuple[FactoredTigerEpisodeResult, ...]


# --- Single-episode driver --------------------------------------------------


def run_episode(
    env: FactoredTigerEnv,
    planner: Planner,
    q_static_init: jnp.ndarray,
    episode_theta_true: int,
    horizon: int,
    max_steps: int,
    deterministic: bool,
    rng: np.random.Generator,
    clip_horizon: bool = False,
) -> FactoredTigerEpisodeResult:
    """Run one Factored Tiger episode with env θ pinned to
    ``episode_theta_true`` and agent prior at ``q_static_init``."""
    env.reset(theta=episode_theta_true)
    theta_spec = env.theta_spec
    _call_prepare_episode(
        planner, q_static_init, rng, episode_theta_true, theta_spec,
    )

    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32) for B_c in env.get_observation_tensors()
    ]
    log_B_list = tuple(jnp.log(B_c + 1e-12) for B_c in B_list)
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    registry = env.get_modality_registry()
    theta_mask = registry.theta_dependent_mask
    t_factor_spec = env.t_factor_spec
    b_factor_specs = env.b_factor_specs
    n_actions = env.n_actions
    K = env.K
    reward_fn = _build_factored_tiger_reward_fn(env)
    terminal_fn = None  # bounded horizon
    reward_table = _build_factored_tiger_reward_table(env)

    q_state = _start_q_state_for(env)
    q_static = q_static_init
    initial_q_static_np = np.asarray(q_static_init).copy()

    actions: list[int] = []
    states_visited: list[int] = [env.agent_state]
    observations: list[list[float]] = []
    cumulative_reward = 0.0
    n_listens = 0
    n_opens = 0
    n_safe_opens = 0
    n_tiger_opens = 0
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

        # Track action role + mask delta for inner-loop metrics.
        pre_mask, _ = decode_state(env.agent_state, K)
        role, _ = action_role(action, K)

        env_step = env.step(action)
        actions.append(action)
        states_visited.append(env.agent_state)
        cumulative_reward += env_step.reward

        if role == "listen":
            n_listens += 1
        else:  # open
            post_mask, _ = decode_state(env.agent_state, K)
            if post_mask != pre_mask:
                n_opens += 1
                # Reward sign disambiguates safe (+open_safe_reward) from
                # tiger (-open_tiger_penalty). Re-OPEN of opened door
                # (no mask change) is excluded by the outer guard.
                if env_step.reward > 0:
                    n_safe_opens += 1
                else:
                    n_tiger_opens += 1

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

    # All K components are the θ bits (no persistent/episodic split).
    n_safe_in_episode = int(sum(ep_components))
    max_possible_reward = float(n_safe_in_episode * env.open_safe_reward)

    return FactoredTigerEpisodeResult(
        n_steps=len(actions),
        actions=tuple(actions),
        states_visited=tuple(states_visited),
        observations=tuple(tuple(o) for o in observations),
        initial_q_static=initial_q_static_np,
        final_q_static=q_static_np.copy(),
        cumulative_reward=float(cumulative_reward),
        n_listens=n_listens,
        n_opens=n_opens,
        n_safe_opens=n_safe_opens,
        n_tiger_opens=n_tiger_opens,
        n_safe_in_episode=n_safe_in_episode,
        max_possible_reward=max_possible_reward,
        episode_theta_true=int(episode_theta_true),
        planning_time_s=planning_time_s,
        terminated_early=bool(terminated),
        **summaries,
    )


# --- Block driver -----------------------------------------------------------


def run_block(
    env: FactoredTigerEnv,
    planner: Planner,
    block_theta_true: int,
    n_episodes: int,
    horizon: int = 3,
    max_steps: int = 20,
    deterministic: bool = True,
    seed: int = 0,
    initial_q_static: jnp.ndarray | None = None,
    clip_horizon: bool = False,
) -> FactoredTigerBlockResult:
    """Run n_episodes with block θ fixed at ``block_theta_true``. All θ
    components are persistent, so q(θ) carries across episodes within
    the block (sharpens over time as listens/opens reveal information).
    """
    rng = np.random.default_rng(seed)
    theta_spec = env.theta_spec

    if initial_q_static is None:
        q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    else:
        q_static = jnp.asarray(initial_q_static, dtype=jnp.float32)

    # Block-level planner setup, once before the episode loop (distinct
    # from prepare_episode, which fires per episode). BAMCP/BOSS reset
    # their rollout Q-hat here — learned within a block, dropped across
    # blocks. Planners without the hook are untouched.
    if hasattr(planner, "prepare_block"):
        planner.prepare_block(q_static, rng)

    episodes: list[FactoredTigerEpisodeResult] = []
    for _ep_idx in range(n_episodes):
        # No-op for theta_spec.episode_reset_components=(), but keeps the
        # call-shape identical to rock_sample for future factor mixes.
        q_static = reset_episode_theta(q_static, theta_spec)

        result = run_episode(
            env=env,
            planner=planner,
            q_static_init=q_static,
            episode_theta_true=block_theta_true,
            horizon=horizon,
            max_steps=max_steps,
            deterministic=deterministic,
            rng=rng,
            clip_horizon=clip_horizon,
        )
        episodes.append(result)
        q_static = jnp.asarray(result.final_q_static, dtype=jnp.float32)

    return FactoredTigerBlockResult(
        K=env.K,
        theta_true=int(block_theta_true),
        n_episodes=n_episodes,
        episodes=tuple(episodes),
    )


# --- Outer-loop driver ------------------------------------------------------


@dataclass(frozen=True)
class FactoredTigerOuterLoopResult:
    config: dict[str, Any]
    n_seeds: int
    n_episodes: int
    theta_trues: tuple[int, ...]
    blocks: tuple[FactoredTigerBlockResult, ...]
    n_steps: np.ndarray
    cumulative_reward: np.ndarray
    n_listens: np.ndarray
    n_opens: np.ndarray
    n_safe_opens: np.ndarray
    n_tiger_opens: np.ndarray
    n_safe_in_episode: np.ndarray
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
            "n_listens": self.n_listens.tolist(),
            "n_opens": self.n_opens.tolist(),
            "n_safe_opens": self.n_safe_opens.tolist(),
            "n_tiger_opens": self.n_tiger_opens.tolist(),
            "n_safe_in_episode": self.n_safe_in_episode.tolist(),
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
            ("n_listens", self.n_listens.astype(float)),
            ("n_opens", self.n_opens.astype(float)),
            ("n_safe_opens", self.n_safe_opens.astype(float)),
            ("n_tiger_opens", self.n_tiger_opens.astype(float)),
            ("n_safe_in_episode", self.n_safe_in_episode.astype(float)),
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
            curves[name] = {
                "mean": np.nanmean(arr, axis=0).tolist(),
                "p05": np.nanpercentile(arr, 5, axis=0).tolist(),
                "p95": np.nanpercentile(arr, 95, axis=0).tolist(),
            }
        # Identification is defined on the *persistent* θ marginal; the
        # episodic marginal is meaningless for calibration since it
        # resamples each episode.
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


def _stack(
    blocks: list[FactoredTigerBlockResult], attr: str, dtype,
) -> np.ndarray:
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
) -> FactoredTigerBlockResult:
    """Subprocess entry point: builds fresh env + planner per worker."""
    env = FactoredTigerEnv(**env_kwargs)
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
    max_steps: int = 20,
    deterministic: bool = True,
    seed: int = 0,
    theta_seeds: list[int] | None = None,
    clip_horizon: bool = False,
    show_progress: bool = True,
    progress_desc: str = "factored-tiger",
    env_factory: Callable[[dict[str, Any]], FactoredTigerEnv] | None = None,
    workers: int = 1,
    planner_factory: Optional[Callable[[int], Planner]] = None,
) -> FactoredTigerOuterLoopResult:
    """Run n_seeds blocks of n_episodes; sequential or parallel by workers."""
    if env_factory is None:
        env_factory = lambda kw: FactoredTigerEnv(**kw)

    if planner is None and planner_factory is None:
        raise ValueError("must provide either `planner` or `planner_factory`")

    if workers > 1:
        if planner_factory is None:
            raise ValueError(
                "`workers > 1` requires `planner_factory` since planner "
                "instances are not pickle-safe across processes."
            )
        probe = planner_factory(0)
        if isinstance(probe, GPU_PLANNER_TYPES):
            warnings.warn(
                f"Planner type {type(probe).__name__} uses JAX/GPU; "
                f"forcing workers=1 to avoid GPU re-init across spawned workers.",
                stacklevel=2,
            )
            workers = 1

    # Pre-resolve block thetas so parallel and sequential paths agree.
    rng = np.random.default_rng(seed)
    block_thetas: list[int] = []
    probe_env = env_factory(dict(env_kwargs))
    for s in range(n_seeds):
        if theta_seeds is not None:
            if len(theta_seeds) != n_seeds:
                raise ValueError(
                    f"theta_seeds length {len(theta_seeds)} != n_seeds {n_seeds}"
                )
            block_thetas.append(int(theta_seeds[s]))
        else:
            block_thetas.append(int(rng.integers(0, probe_env.n_static)))

    blocks: list[FactoredTigerBlockResult] = []
    theta_trues = list(block_thetas)

    if workers > 1:
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
                cum_reward_sum += sum(
                    ep.cumulative_reward for ep in block.episodes
                )
                cum_episodes += n_episodes
                if show_progress:
                    iterator.set_postfix(
                        avg_rew=f"{cum_reward_sum / max(cum_episodes, 1):.2f}",
                    )
    else:
        iterator = range(n_seeds)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc, unit="seed")
        cum_reward_sum = 0.0
        cum_episodes = 0
        for s in iterator:
            kwargs = dict(env_kwargs)
            kwargs["obs_seed"] = seed * 10_000 + s
            env = env_factory(kwargs)
            # The factory takes the outer-seed index so each seed gets its
            # own planner RNG stream: a fresh instance alone is not enough,
            # since `default_rng(<same int>)` replays the same stream and
            # the per-plan() draw count is fixed by n_simulations.
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
    with np.errstate(divide="ignore", invalid="ignore"):
        reward_normalised = np.where(
            max_possible_reward > 0,
            cumulative_reward / max_possible_reward,
            np.nan,
        )
    return FactoredTigerOuterLoopResult(
        config=config,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        theta_trues=tuple(theta_trues),
        blocks=tuple(blocks),
        n_steps=_stack(blocks, "n_steps", int),
        cumulative_reward=cumulative_reward,
        n_listens=_stack(blocks, "n_listens", int),
        n_opens=_stack(blocks, "n_opens", int),
        n_safe_opens=_stack(blocks, "n_safe_opens", int),
        n_tiger_opens=_stack(blocks, "n_tiger_opens", int),
        n_safe_in_episode=_stack(blocks, "n_safe_in_episode", int),
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
