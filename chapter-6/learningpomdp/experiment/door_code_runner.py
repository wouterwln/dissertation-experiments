"""Persistent-theta multi-episode runner for the door-code env.

Implements the cross-episode identification protocol from M3 of
`door_code_system_id.md` and the m x n evaluation protocol from
"Evaluation Protocol":

- `run_episode`     : single episode with q(theta) initialized from a prior.
- `run_block`       : n episodes with theta pinned, q(theta) carried.
- `run_outer_loop`  : m blocks, each with a fresh theta_true sampled
                      uniformly. Returns aggregated per-episode metrics
                      suitable for paper plots.

q(x) resets to a START-prior one-hot at each episode (the env always
starts at START); q(theta) resets to uniform at the start of each
outer-loop seed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.inference.state_inference import bayesian_state_update
from learningpomdp.planning.base import Planner


def _write_video(frames: list[np.ndarray], path: Path, fps: int = 4) -> None:
    """Write a list of HxWx3 uint8 frames to `path` as mp4. Uses imageio
    with the bundled ffmpeg plugin so no system ffmpeg is required."""
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), np.stack(frames, axis=0), fps=fps, codec="libx264")


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    n_steps: int
    actions: tuple[int, ...]
    states_visited: tuple[int, ...]
    observations: tuple[tuple[float, ...], ...]
    probe_visits: int
    initial_q_static: np.ndarray
    final_q_static: np.ndarray
    posterior_mass_on_true_theta: float
    posterior_entropy: float
    terminal_reward: float
    cumulative_reward: float
    planning_time_s: float
    terminated_early: bool


@dataclass(frozen=True)
class BlockResult:
    K: int
    theta_true: int
    n_episodes: int
    episodes: tuple[EpisodeResult, ...]


def _start_q_state_for(env: Any) -> jnp.ndarray:
    return jnp.zeros(env.n_states, dtype=jnp.float32).at[env.start_state_index].set(1.0)


def _entropy(p: np.ndarray) -> float:
    safe = p[p > 0]
    return float(-np.sum(safe * np.log(safe)))


def _select_action(
    action_dist: jnp.ndarray,
    deterministic: bool,
    rng: np.random.Generator,
) -> int:
    p = np.asarray(action_dist, dtype=np.float64)
    if deterministic:
        return int(p.argmax())
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def _effective_horizons(
    horizon: int, max_steps: int, clip_horizon: bool,
) -> list[int]:
    """Every distinct horizon `run_episode` will hand `plan()`.

    Mirrors the `effective_horizon` expression in `run_episode`; keep the
    two in sync.
    """
    if not clip_horizon:
        return [horizon]
    return sorted({min(horizon, max_steps - step) for step in range(max_steps)})


def _warmup_planner(
    env: Any,
    planner: Planner,
    horizon: int,
    max_steps: int,
    clip_horizon: bool,
) -> None:
    """Pay every XLA compile the episode loop would otherwise pay inside
    the timed region.

    `AIFPlanner` / `VBPPlanner` jit on `horizon`, so under `clip_horizon`
    a single process compiles once per distinct `effective_horizon` the
    step loop reaches — up to `min(horizon, max_steps)` separate compiles.
    Every one of them lands inside `run_episode`'s `perf_counter` window
    and is charged to `planning_time_s`, which is what makes an un-warmed
    `planning_time_s` partly a compile measurement. One discarded
    `plan()` per distinct horizon here moves all of them outside every
    timed region.

    Only correct for planners whose `plan()` is a pure function of its
    arguments (AIF, VBP, Oracle, PSRL). The sample-based planners
    (BAMCP / BOSS / IDS / POMCP / BeliefQ / Random) draw from an internal
    RNG inside `plan()`, so a discarded warm-up call would shift their
    stream and change their results. Hence opt-in and off by default:
    only the timing stage, which runs AIF, asks for it.
    """
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32)
        for B_c in env.get_observation_tensors()
    ]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    q_state = _start_q_state_for(env)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    for h in _effective_horizons(horizon, max_steps, clip_horizon):
        action_dist = planner.plan(
            q_state, q_static, T, B_list, goal, h,
            theta_mask=env.get_modality_registry().theta_dependent_mask,
            theta_spec=env.theta_spec,
            t_factor_spec=env.t_factor_spec,
            b_factor_specs=env.b_factor_specs,
        )
        if hasattr(action_dist, "block_until_ready"):
            action_dist.block_until_ready()


# Environment variables that decide how much parallelism the process may
# use. Recorded, never set: a `planning_time_s` measured in a 2-core SLURM
# cgroup and one measured on a many-core workstation are different numbers
# for the same code, and the compute table has no other way to say which
# regime it is quoting.
_THREAD_ENV_VARS = (
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "XLA_FLAGS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
)


def _runtime_record(warmup_planner: bool) -> dict[str, Any]:
    """The timing regime of this process, for `config["runtime"]`.

    `schedulable_cpus` is the affinity-mask size, not the node's core
    count: it is what XLA's host thread pool and OpenMP actually size
    themselves from under a SLURM cgroup, so it — not `--cpus-per-task`
    alone — is the number that determines the regime.
    """
    if hasattr(os, "sched_getaffinity"):
        schedulable_cpus = len(os.sched_getaffinity(0))
    else:
        schedulable_cpus = os.cpu_count()
    return {
        "jax_backend": jax.default_backend(),
        "schedulable_cpus": schedulable_cpus,
        "thread_env": {n: os.environ.get(n) for n in _THREAD_ENV_VARS},
        "warmup_planner": bool(warmup_planner),
    }


def run_episode(
    env: DoorCodeEnv,
    planner: Planner,
    q_static_init: jnp.ndarray,
    theta_true: int,
    horizon: int,
    max_steps: int,
    deterministic: bool,
    rng: np.random.Generator,
    learn_on_failure: bool = True,
    clip_horizon: bool = False,
    video_path: Path | None = None,
    video_fps: int = 4,
) -> EpisodeResult:
    """Run a single episode with theta pinned and q(theta) initialized.

    Plan-act-filter ordering: at each step the agent plans from current
    beliefs, takes the action, and only then updates beliefs from the
    resulting observation. The very first observation (env.reset) is
    consistent with q_state=onehot(START) and q_static=q_static_init,
    so no t=0 filter call is needed; doing one with action=uniform
    would propagate q_static through T's theta-dependence as if some
    averaged action had been taken, which destroys a sharp q_static.

    Args:
        learn_on_failure: if False, skip the state-inference call on
            steps where the agent ends up in FAILURE (terminal_reward
            negative). The agent is "dead" and cannot extract Bayesian
            information from a transition it didn't survive to observe.
            Probe-derived q(theta) updates from earlier steps are
            retained. Default True (the standard Bayesian filter).
        clip_horizon: if True, the per-step planning horizon is capped
            at the number of steps remaining in the episode
            (max_steps - step). Receding-horizon control: the planner
            never plans past the actual budget, so it can't pick a
            first-action choice that only pays off across an imagined-
            but-unrealizable later commit. Default False (preserves
            backward-compat with prior runs).
    """
    env.reset(theta=theta_true)
    SUCCESS = env.success_state_index

    # Per-step rgb_array frames, only when video_path is set. The env
    # must have been constructed with render_mode='rgb_array' for this
    # to yield arrays. The first frame is the post-reset state.
    frames: list[np.ndarray] | None = [] if video_path is not None else None
    if frames is not None:
        first_frame = env.render()
        if first_frame is not None:
            frames.append(np.asarray(first_frame))

    # Episode-level planner setup (PSRL samples theta from q_static here;
    # OraclePlanner stashes theta_true; AIFPlanner / VBPPlanner do not
    # implement prepare_episode and are untouched).
    if hasattr(planner, "prepare_episode"):
        try:
            planner.prepare_episode(
                q_static_init, rng, theta_true=theta_true,
            )
        except TypeError:
            # Planner's prepare_episode does not accept theta_true
            # (PSRL, BAMCP, BeliefQ, BOSS, ...) — call without it.
            planner.prepare_episode(q_static_init, rng)

    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32) for B_c in env.get_observation_tensors()
    ]
    log_B_list = tuple(jnp.log(B_c + 1e-12) for B_c in B_list)
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    registry = env.get_modality_registry()
    theta_mask = registry.theta_dependent_mask
    theta_spec = env.theta_spec
    t_factor_spec = env.t_factor_spec
    b_factor_specs = env.b_factor_specs
    n_actions = env.n_actions
    probe_state_set = registry.get_modality_by_name("probe").informative_states or frozenset()

    q_state = _start_q_state_for(env)
    q_static = q_static_init
    initial_q_static_np = np.asarray(q_static_init).copy()

    actions: list[int] = []
    states_visited: list[int] = [env.agent_state]
    observations: list[list[float]] = []
    # Count distinct probe *cells* visited, not distinct probe states.
    # In spatial envs (MiniGridDoorCodeEnv) one probe cell maps to many
    # state indices across heading/stage, so adding raw states would
    # multi-count single visits. `env.state_to_cell` collapses those —
    # it is identity for graph envs (door_code) where each state already
    # corresponds to a single cell.
    probes_visited: set[int] = set()
    terminal_reward = 0.0
    cumulative_reward = 0.0
    planning_time_s = 0.0
    terminated = False

    for step in range(max_steps):
        if clip_horizon:
            effective_horizon = min(horizon, max_steps - step)
        else:
            effective_horizon = horizon
        _t0 = time.perf_counter()
        action_dist = planner.plan(
            q_state, q_static, T, B_list, goal, effective_horizon,
            theta_mask=theta_mask,
            theta_spec=theta_spec,
            t_factor_spec=t_factor_spec,
            b_factor_specs=b_factor_specs,
        )
        # JAX dispatches asynchronously; block on the result so timing
        # reflects actual compute rather than just queue-submit cost.
        if hasattr(action_dist, "block_until_ready"):
            action_dist.block_until_ready()
        planning_time_s += time.perf_counter() - _t0
        action = _select_action(action_dist, deterministic, rng)

        env_step = env.step(action)
        if frames is not None:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
        actions.append(action)
        states_visited.append(env.agent_state)
        if env.agent_state in probe_state_set:
            probes_visited.add(env.state_to_cell(env.agent_state))
        terminal_reward = env_step.reward
        cumulative_reward += float(env_step.reward)
        obs = jnp.asarray(env_step.obs, dtype=jnp.float32)
        observations.append(np.asarray(env_step.obs, dtype=np.float32).tolist())

        # Skip the state update if the agent died and learn_on_failure=False.
        # Probe-derived updates from earlier steps in this episode remain.
        if learn_on_failure or env_step.reward >= 0:
            action_onehot = jnp.zeros(n_actions, dtype=jnp.float32).at[action].set(1.0)
            q_state, q_static = bayesian_state_update(
                q_state, q_static, T, log_B_list, obs, action_onehot,
                theta_spec=theta_spec,
                t_factor_spec=t_factor_spec,
                b_factor_specs=b_factor_specs,
            )

        # Optional post-step hook for RL agents (e.g. belief-state Q-learning
        # does its TD update here). AIF / VBP / PSRL don't implement it.
        if hasattr(planner, "post_step_update"):
            planner.post_step_update(
                q_state, q_static, env_step.reward, env_step.terminated,
            )

        if env_step.terminated:
            terminated = True
            break

    if frames is not None and video_path is not None and len(frames) > 0:
        _write_video(frames, video_path, fps=video_fps)

    # q_static is rank-K (theta_spec.component_sizes); flatten for the
    # single-int theta_true index used by M1/M2. Multi-component envs
    # would need a tuple-index, but those aren't yet wired into the
    # outer-loop driver.
    q_static_np = np.asarray(q_static)
    q_static_flat = q_static_np.reshape(-1)
    return EpisodeResult(
        success=(env.agent_state == SUCCESS),
        n_steps=len(actions),
        actions=tuple(actions),
        states_visited=tuple(states_visited),
        observations=tuple(tuple(o) for o in observations),
        probe_visits=len(probes_visited),
        initial_q_static=initial_q_static_np,
        final_q_static=q_static_np.copy(),
        posterior_mass_on_true_theta=float(q_static_flat[theta_true]),
        posterior_entropy=_entropy(q_static_flat),
        terminal_reward=float(terminal_reward),
        cumulative_reward=cumulative_reward,
        planning_time_s=planning_time_s,
        terminated_early=bool(terminated),
    )


def run_block(
    env: DoorCodeEnv,
    planner: Planner,
    theta_true: int,
    n_episodes: int,
    horizon: int = 3,
    max_steps: int = 6,
    deterministic: bool = True,
    seed: int = 0,
    initial_q_static: jnp.ndarray | None = None,
    learn_on_failure: bool = True,
    clip_horizon: bool = False,
    video_path_for_episode: Callable[[int], Path | None] | None = None,
    video_fps: int = 4,
) -> BlockResult:
    """Run n_episodes with theta pinned and q(theta) carried across resets.

    Args:
        learn_on_failure: see run_episode docstring (default True).
        clip_horizon: see run_episode docstring (default False).
        video_path_for_episode: optional callable mapping the inner
            episode index to a per-episode mp4 output path; None to
            skip recording. The env must have been constructed with
            `render_mode='rgb_array'` for frames to be captured.
    """
    rng = np.random.default_rng(seed)
    if initial_q_static is None:
        q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    else:
        q_static = jnp.asarray(initial_q_static, dtype=jnp.float32)

    # Block-level planner setup, called once before the episode loop —
    # distinct from prepare_episode, which fires per episode. BAMCP/BOSS
    # reset their rollout Q-hat here: it is learned across the episodes of
    # one block (theta_true pinned) and must not survive into the next
    # block (theta_true redrawn). Planners without the hook are untouched.
    if hasattr(planner, "prepare_block"):
        planner.prepare_block(q_static, rng)

    episodes: list[EpisodeResult] = []
    for ep_idx in range(n_episodes):
        video_path = (
            video_path_for_episode(ep_idx)
            if video_path_for_episode is not None else None
        )
        result = run_episode(
            env=env,
            planner=planner,
            q_static_init=q_static,
            theta_true=theta_true,
            horizon=horizon,
            max_steps=max_steps,
            deterministic=deterministic,
            rng=rng,
            learn_on_failure=learn_on_failure,
            clip_horizon=clip_horizon,
            video_path=video_path,
            video_fps=video_fps,
        )
        episodes.append(result)
        q_static = jnp.asarray(result.final_q_static, dtype=jnp.float32)

    return BlockResult(
        K=env.K,
        theta_true=theta_true,
        n_episodes=n_episodes,
        episodes=tuple(episodes),
    )


@dataclass(frozen=True)
class OuterLoopResult:
    """Aggregated outcome of m blocks, each with a sampled theta_true.

    Per-episode metric arrays have shape (n_seeds, n_episodes); the
    outer axis is the seed (theta_true) index, the inner axis is
    episode index within the block. These are the arrays the paper
    plots: mean +/- 90% CI across seeds, per episode.

    `blocks` retains the full BlockResult per seed for ad-hoc analysis.
    """

    config: dict[str, Any]
    n_seeds: int
    n_episodes: int
    theta_trues: tuple[int, ...]
    blocks: tuple[BlockResult, ...]
    success: np.ndarray            # (n_seeds, n_episodes) int
    n_steps: np.ndarray            # (n_seeds, n_episodes) int
    probe_visits: np.ndarray       # (n_seeds, n_episodes) int
    posterior_mass_true: np.ndarray  # (n_seeds, n_episodes) float
    posterior_entropy: np.ndarray  # (n_seeds, n_episodes) float
    terminal_reward: np.ndarray    # (n_seeds, n_episodes) float
                                   # +1 success, -1 catastrophic failure,
                                   #  0 truncation (ran out of steps)
    cumulative_reward: np.ndarray  # (n_seeds, n_episodes) float
    planning_time_s: np.ndarray    # (n_seeds, n_episodes) float

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary (config + per-episode arrays as
        nested lists, plus aggregate curves with mean and 5/95 percentiles).

        Also emits a `trajectories` block: per-seed, per-episode list of
        dicts with the action sequence, observations, states visited,
        initial / final q(theta), and terminated_early flag. Replay-
        sufficient: with `inference/state_inference.py.bayesian_state_update`
        these reproduce every per-step belief without rerunning the planner.
        """
        per_episode = {
            "success": self.success.tolist(),
            "n_steps": self.n_steps.tolist(),
            "probe_visits": self.probe_visits.tolist(),
            "posterior_mass_on_true_theta": self.posterior_mass_true.tolist(),
            "posterior_entropy": self.posterior_entropy.tolist(),
            "terminal_reward": self.terminal_reward.tolist(),
            "cumulative_reward": self.cumulative_reward.tolist(),
            "planning_time_s": self.planning_time_s.tolist(),
        }
        curves = {}
        for name, arr in [
            ("success", self.success.astype(float)),
            ("n_steps", self.n_steps.astype(float)),
            ("probe_visits", self.probe_visits.astype(float)),
            ("posterior_mass_on_true_theta", self.posterior_mass_true),
            ("posterior_entropy", self.posterior_entropy),
            ("cumulative_reward", self.cumulative_reward),
            ("planning_time_s", self.planning_time_s),
        ]:
            curves[name] = {
                "mean": arr.mean(axis=0).tolist(),
                "p05": np.percentile(arr, 5, axis=0).tolist(),
                "p95": np.percentile(arr, 95, axis=0).tolist(),
            }
        derived = build_derived_metrics(
            posterior_mass_true=self.posterior_mass_true,
            terminal_reward=self.terminal_reward,
        )
        trajectories = [
            [
                {
                    "actions": list(ep.actions),
                    "observations": [list(o) for o in ep.observations],
                    "states_visited": list(ep.states_visited),
                    "initial_q_static": np.asarray(ep.initial_q_static).reshape(-1).tolist(),
                    "final_q_static": np.asarray(ep.final_q_static).reshape(-1).tolist(),
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


def _stack_metric(blocks: list[BlockResult], attr: str, dtype) -> np.ndarray:
    return np.array(
        [[getattr(ep, attr) for ep in b.episodes] for b in blocks],
        dtype=dtype,
    )


# Threshold used for the "episodes-to-identify" scalar. 0.9 mass on the
# true theta is the canonical "the agent has identified the system"
# criterion used in the paper's identification protocol.
_IDENTIFY_THRESHOLD = 0.9


def _survival_quantile(
    fraction_by_episode: list[float], q: float,
) -> float | None:
    """First episode index at which the crossing fraction reaches `q`.

    None when the fraction never reaches `q` within the budget, i.e. the
    quantile is right-censored at the episode budget.
    """
    for k, frac in enumerate(fraction_by_episode):
        if frac >= q:
            return float(k)
    return None


def episodes_to_identify(
    posterior_mass_true: np.ndarray,
    threshold: float = _IDENTIFY_THRESHOLD,
) -> dict[str, Any]:
    """For each seed, find the first episode index where posterior mass
    on the true theta is >= threshold. Seeds that never cross the
    threshold contribute NaN.

    Two families of summary come out of the same first-crossing times:

    `median` / `p25` / `p75` are **conditional on crossing** — computed
    over the seeds that crossed, with `fraction_identified` reporting
    what share those were. Comparing them across methods compares
    different subpopulations of blocks whenever the crossing fractions
    differ, so they are kept for backward compatibility (the figures read
    them) but are not the primary statistic.

    `fraction_identified_by_episode` / `censored_median` /
    `censored_p25` / `censored_p75` treat non-crossing seeds as
    right-censored at the episode budget instead of dropping them.
    `fraction_identified_by_episode[k]` is the share of all seeds that
    have crossed by episode index k (a survival curve, complement of
    S(k)); the censored quantiles are the first index where that curve
    reaches 0.25 / 0.5 / 0.75, and are None when it never does — which is
    the honest answer for a method that identifies fewer than half its
    blocks within the budget.

    Censoring here is purely administrative and common to every seed (the
    episode budget), and it happens after every observed event, so the
    Kaplan-Meier product-limit estimator of S coincides exactly with
    1 - ECDF of the first-crossing times; no product-limit loop is
    needed and none is used.
    """
    crossed = posterior_mass_true >= threshold
    n_episodes = posterior_mass_true.shape[1]
    first_idx = np.where(crossed.any(axis=1), crossed.argmax(axis=1), -1)
    per_seed = [int(i) if i >= 0 else None for i in first_idx]
    crossed_idx = first_idx[first_idx >= 0].astype(float)
    if crossed_idx.size > 0:
        median = float(np.median(crossed_idx))
        p25 = float(np.percentile(crossed_idx, 25))
        p75 = float(np.percentile(crossed_idx, 75))
    else:
        median = None
        p25 = None
        p75 = None
    ever = first_idx >= 0
    fraction_by_episode = [
        float((ever & (first_idx <= k)).mean()) for k in range(n_episodes)
    ]
    return {
        "threshold": threshold,
        "n_episodes_budget": int(n_episodes),
        "per_seed": per_seed,
        "median": median,
        "p25": p25,
        "p75": p75,
        "fraction_identified": float(ever.mean()),
        "fraction_identified_by_episode": fraction_by_episode,
        "censored_median": _survival_quantile(fraction_by_episode, 0.5),
        "censored_p25": _survival_quantile(fraction_by_episode, 0.25),
        "censored_p75": _survival_quantile(fraction_by_episode, 0.75),
    }


def calibration_curves(posterior_mass: np.ndarray) -> dict[str, dict]:
    """Brier = (1 - p_true)**2 and NLL = -log(p_true), averaged across
    seeds per episode. Same mean/p05/p95 layout as `curves`."""
    p = np.clip(posterior_mass, 1e-12, 1.0)
    brier = (1.0 - posterior_mass) ** 2
    nll = -np.log(p)
    out = {}
    for name, arr in [("brier", brier), ("nll", nll)]:
        out[name] = {
            "mean": arr.mean(axis=0).tolist(),
            "p05": np.percentile(arr, 5, axis=0).tolist(),
            "p95": np.percentile(arr, 95, axis=0).tolist(),
        }
    return out


def build_derived_metrics(
    posterior_mass_true: np.ndarray,
    terminal_reward: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bundle calibration curves and the episodes-to-identify scalar.

    Plot scripts compute regret post-hoc against an oracle JSON, so it
    is not added here.
    """
    derived: dict[str, Any] = {}
    derived.update(calibration_curves(posterior_mass_true))
    derived["episodes_to_identify"] = episodes_to_identify(
        posterior_mass_true,
    )
    return derived


def run_outer_loop(
    env_kwargs: dict[str, Any],
    planner: Planner,
    n_seeds: int,
    n_episodes: int,
    horizon: int = 3,
    max_steps: int = 2,
    deterministic: bool = True,
    seed: int = 0,
    theta_seeds: list[int] | None = None,
    learn_on_failure: bool = True,
    clip_horizon: bool = False,
    show_progress: bool = True,
    progress_desc: str = "outer-loop",
    env_factory: Callable[[dict[str, Any]], Any] | None = None,
    video_path_for: Callable[[int, int], Path | None] | None = None,
    video_fps: int = 4,
    warmup_planner: bool = False,
) -> OuterLoopResult:
    """Run m = n_seeds blocks, each with a sampled theta_true and a
    fresh env (independent obs RNG per seed).

    Args:
        env_kwargs: Keyword args passed to env_factory (K, M, probe_noise,
            partitions, ...). `obs_seed` is overridden per outer-seed.
        n_seeds: Number of outer-loop seeds (m).
        n_episodes: Episodes per block (n).
        theta_seeds: Optional explicit list of theta_true values
            (length must match n_seeds). If None, sampled uniformly
            using `seed`.
        learn_on_failure: see run_episode docstring. Default True.
        show_progress: display a tqdm progress bar over outer seeds.
        progress_desc: tqdm bar label.
        env_factory: Callable mapping the per-seed kwargs dict to an env
            instance. Defaults to DoorCodeEnv(**kwargs); pass a different
            factory (e.g. lambda kw: MiniGridDoorCodeEnv(**kw)) to run
            the same protocol against a different env.
        warmup_planner: if True, run one discarded `plan()` per distinct
            planning horizon before the first timed decision, so XLA
            compiles are not charged to `planning_time_s`. Default False;
            see `_warmup_planner` for why it is not safe to turn on for
            the sample-based planners.

    Returns:
        OuterLoopResult with stacked per-episode metrics.
    """
    if env_factory is None:
        env_factory = lambda kw: DoorCodeEnv(**kw)

    rng = np.random.default_rng(seed)
    blocks: list[BlockResult] = []
    theta_trues: list[int] = []

    iterator = range(n_seeds)
    if show_progress:
        iterator = tqdm(iterator, desc=progress_desc, unit="seed")

    cum_success = 0
    cum_deaths = 0
    for s in iterator:
        kwargs = dict(env_kwargs)
        kwargs["obs_seed"] = seed * 10_000 + s
        env = env_factory(kwargs)
        if warmup_planner and s == 0:
            # Once per process is enough: the AIF/VBP jit cache is
            # module-level and keyed on shapes and horizon, both of which
            # are fixed across seeds.
            _warmup_planner(env, planner, horizon, max_steps, clip_horizon)
        if theta_seeds is not None:
            if len(theta_seeds) != n_seeds:
                raise ValueError(
                    f"theta_seeds length {len(theta_seeds)} != n_seeds {n_seeds}"
                )
            theta_true = int(theta_seeds[s])
        else:
            theta_true = int(rng.integers(0, env.n_static))
        ep_video_path = (
            (lambda ep, _s=s: video_path_for(_s, ep))
            if video_path_for is not None else None
        )
        block = run_block(
            env, planner, theta_true=theta_true, n_episodes=n_episodes,
            horizon=horizon, max_steps=max_steps, deterministic=deterministic,
            seed=seed * 10_000 + s,
            learn_on_failure=learn_on_failure,
            clip_horizon=clip_horizon,
            video_path_for_episode=ep_video_path,
            video_fps=video_fps,
        )
        blocks.append(block)
        theta_trues.append(theta_true)

        cum_success += sum(1 for ep in block.episodes if ep.success)
        cum_deaths += sum(1 for ep in block.episodes if ep.terminal_reward < 0)
        if show_progress:
            n_seen = (s + 1) * n_episodes
            iterator.set_postfix(
                succ=f"{cum_success / n_seen:.2f}",
                deaths=f"{cum_deaths / (s + 1):.1f}/blk",
            )

    config = {
        "env": env_kwargs,
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "horizon": horizon,
        "max_steps": max_steps,
        "deterministic": deterministic,
        "seed": seed,
        "learn_on_failure": learn_on_failure,
        "clip_horizon": clip_horizon,
        "runtime": _runtime_record(warmup_planner),
    }
    return OuterLoopResult(
        config=config,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        theta_trues=tuple(theta_trues),
        blocks=tuple(blocks),
        success=_stack_metric(blocks, "success", int),
        n_steps=_stack_metric(blocks, "n_steps", int),
        probe_visits=_stack_metric(blocks, "probe_visits", int),
        posterior_mass_true=_stack_metric(
            blocks, "posterior_mass_on_true_theta", float),
        posterior_entropy=_stack_metric(blocks, "posterior_entropy", float),
        terminal_reward=_stack_metric(blocks, "terminal_reward", float),
        cumulative_reward=_stack_metric(blocks, "cumulative_reward", float),
        planning_time_s=_stack_metric(blocks, "planning_time_s", float),
    )
