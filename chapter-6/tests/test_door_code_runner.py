"""M3 gate: persistent-theta multi-episode runner on the door-code env.

Gate from `door_code_system_id.md`:
- Posterior entropy H[q(theta)] drops across episodes within a block.
- Success rate rises across episodes.
- Probe-visit rate per episode falls after identification.

The empirical regime targeted by the paper plan is K=4 with noisy probes
(probe_noise=0.2) and tight max_steps=2, which produces a ~20-episode
identification curve. Fast deterministic regimes (K=3 noiseless) are
also covered as sanity checks.
"""

import math

import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import (
    run_block,
    run_episode,
)
from learningpomdp.planning.aif_planner import AIFPlanner


@pytest.fixture(scope="module")
def planner() -> AIFPlanner:
    return AIFPlanner()


def _aggregate_block_seeds(
    *,
    K: int,
    probe_noise: float,
    max_steps: int,
    n_episodes: int,
    n_seeds: int,
    horizon: int,
    planner: AIFPlanner,
) -> dict[str, np.ndarray]:
    """Run multiple seeds (each pinning a distinct theta_true) and stack
    per-episode metrics into (n_seeds, n_episodes) arrays."""
    successes = []
    masses = []
    probes = []
    entropies = []
    for trial in range(n_seeds):
        env = DoorCodeEnv(K=K, probe_noise=probe_noise, obs_seed=trial * 1000)
        theta_true = trial % env.n_static
        block = run_block(
            env, planner, theta_true=theta_true, n_episodes=n_episodes,
            horizon=horizon, max_steps=max_steps, deterministic=True,
            seed=trial,
        )
        successes.append([int(ep.success) for ep in block.episodes])
        masses.append([ep.posterior_mass_on_true_theta for ep in block.episodes])
        probes.append([ep.probe_visits for ep in block.episodes])
        entropies.append([ep.posterior_entropy for ep in block.episodes])
    return {
        "success": np.array(successes),
        "mass": np.array(masses),
        "probes": np.array(probes),
        "entropy": np.array(entropies),
    }


# ---------- Fast deterministic regime (K=3, noiseless, plenty of steps) ----------

def test_fast_regime_K3_solves_in_one_episode(planner: AIFPlanner) -> None:
    """K=3 with deterministic probes and max_steps=4 fully identifies in
    one episode for every theta. Sanity check that the cycle still works
    after the env redesign."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    for theta_true in range(env.n_static):
        block = run_block(
            env, planner, theta_true=theta_true, n_episodes=4,
            horizon=3, max_steps=4, deterministic=True, seed=0,
        )
        assert block.episodes[0].success
        for ep in block.episodes[1:]:
            assert ep.success
            assert ep.n_steps == 1
            assert ep.probe_visits == 0


# ---------- Paper regime: K=4, noise=0.2, max_steps=2 ----------

@pytest.fixture(scope="module")
def paper_block(planner: AIFPlanner) -> dict[str, np.ndarray]:
    return _aggregate_block_seeds(
        K=4, probe_noise=0.2, max_steps=2,
        n_episodes=25, n_seeds=8, horizon=3, planner=planner,
    )


def test_paper_regime_success_rate_rises(paper_block: dict[str, np.ndarray]) -> None:
    """Episode-0 success rate is near zero (theta unknown); late-block
    success rate is well above chance and substantially higher."""
    success = paper_block["success"]
    early = success[:, :3].mean()
    late = success[:, -5:].mean()

    chance = 1.0 / 4.0  # K=4 doors
    assert early < chance + 0.1, f"early success rate {early:.2f} too high"
    assert late > 0.6, f"late success rate {late:.2f} too low"
    assert late > early + 0.3, (
        f"success rate did not rise enough: early={early:.2f}, late={late:.2f}"
    )


def test_paper_regime_posterior_mass_rises(
    paper_block: dict[str, np.ndarray],
) -> None:
    """Posterior mass on the true theta increases over episodes."""
    mass = paper_block["mass"]
    early = mass[:, :3].mean()
    late = mass[:, -5:].mean()

    uniform_mass = 1.0 / 24.0
    assert early < 0.3, (
        f"early posterior mass on true theta is too high "
        f"(should start near uniform={uniform_mass:.3f}): {early:.3f}"
    )
    assert late > 0.7, (
        f"late posterior mass on true theta should be high: {late:.3f}"
    )


def test_paper_regime_probe_visits_decline(
    paper_block: dict[str, np.ndarray],
) -> None:
    """Probe-visit rate falls across the block — early episodes probe
    actively, late episodes commit directly."""
    probes = paper_block["probes"]
    early = probes[:, :3].mean()
    late = probes[:, -5:].mean()

    assert early > 0.5, f"agent should probe in early episodes: {early:.2f}"
    assert late < early, (
        f"probe rate did not drop: early={early:.2f}, late={late:.2f}"
    )


def test_paper_regime_entropy_drops(paper_block: dict[str, np.ndarray]) -> None:
    """Posterior entropy decreases on average."""
    entropy = paper_block["entropy"]
    initial = math.log(24)
    early = entropy[:, :3].mean()
    late = entropy[:, -5:].mean()
    assert late < early, (
        f"entropy did not drop: early={early:.2f}, late={late:.2f} "
        f"(initial uniform = {initial:.2f})"
    )


# ---------- Persistence and sanity ----------

def test_q_static_persists_across_resets(planner: AIFPlanner) -> None:
    """The block runner must carry q_static into the next episode.

    A second-episode probe-zero count is only possible if q_static was
    handed across the env reset; otherwise ep 1 starts uniform and probes
    again."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    block = run_block(
        env, planner, theta_true=2, n_episodes=2,
        horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    assert block.episodes[1].probe_visits == 0
    assert block.episodes[1].n_steps == 1


def test_run_episode_with_pinned_q_static_misleads_agent(
    planner: AIFPlanner,
) -> None:
    """run_episode with q_static sharp on the wrong theta opens the wrong
    door — sanity check that the input is genuinely consumed."""
    import jax.numpy as jnp
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    theta_true = 0   # perm = (0,1,2), correct door = 0
    theta_decoy = 4  # perm = (2,0,1), correct door = 2
    q_static_decoy = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_decoy].set(1.0)
    rng = np.random.default_rng(0)

    result = run_episode(
        env=env, planner=planner, q_static_init=q_static_decoy,
        theta_true=theta_true, horizon=3, max_steps=2,
        deterministic=True, rng=rng,
    )
    # Agent commits to door 2 (decoy's correct door, true's wrong door).
    assert not result.success
    assert result.terminal_reward == -1.0


def test_learn_on_failure_false_freezes_q_static_on_death(
    planner: AIFPlanner,
) -> None:
    """With learn_on_failure=False, a failed door commit must not move
    q(theta) — the agent is dead and cannot update.

    Setup: pin q_static sharp on the wrong theta so the agent commits
    the wrong door (failure). Compare q_static after the episode under
    learn_on_failure True vs False:
    - True: q_static updates (the FAILURE transition is theta-dependent
      and rules out perms with that pi(0)).
    - False: q_static is unchanged from q_static_init.
    """
    import jax.numpy as jnp
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    theta_true = 0      # perm = (0,1,2), correct door = 0
    theta_decoy = 4     # perm = (2,0,1), correct door = 2
    q_static_decoy = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_decoy].set(1.0)

    res_learn = run_episode(
        env=env, planner=planner, q_static_init=q_static_decoy,
        theta_true=theta_true, horizon=3, max_steps=2,
        deterministic=True, rng=np.random.default_rng(0),
        learn_on_failure=True,
    )
    res_die = run_episode(
        env=env, planner=planner, q_static_init=q_static_decoy,
        theta_true=theta_true, horizon=3, max_steps=2,
        deterministic=True, rng=np.random.default_rng(0),
        learn_on_failure=False,
    )

    # Both fail.
    assert not res_learn.success
    assert not res_die.success

    # learn_on_failure=False: q_static must equal q_static_init.
    np.testing.assert_allclose(
        res_die.final_q_static, np.asarray(q_static_decoy), atol=1e-6
    )
    # learn_on_failure=True: q_static moved.
    assert not np.allclose(res_learn.final_q_static, np.asarray(q_static_decoy))


def test_clip_horizon_passes_decreasing_horizon_to_planner() -> None:
    """When clip_horizon=True, run_episode should call planner.plan with
    a horizon that decreases each step toward 1 at the final step."""
    import jax.numpy as jnp
    env = DoorCodeEnv(K=3, probe_noise=0.0)

    seen_horizons: list[int] = []

    class HorizonRecorder:
        """Stand-in planner that records the horizon arg and returns a
        valid one-hot action distribution."""
        def plan(self, q_state, q_static, T, B_list, goal, horizon, theta_mask=None, **_):
            seen_horizons.append(int(horizon))
            n_actions = T.shape[3]
            d = np.zeros(n_actions, dtype=np.float32)
            d[0] = 1.0  # always VISIT_PROBE_0 (safe, non-terminal)
            return jnp.asarray(d)

    rng = np.random.default_rng(0)
    run_episode(
        env=env, planner=HorizonRecorder(),
        q_static_init=jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static,
        theta_true=0, horizon=4, max_steps=4,
        deterministic=True, rng=rng,
        clip_horizon=True,
    )
    # Should have planned 4 times with horizons [4, 3, 2, 1].
    assert seen_horizons == [4, 3, 2, 1], f"got {seen_horizons}"


def test_clip_horizon_off_keeps_horizon_constant() -> None:
    """Without clip_horizon, the planner is called with the same horizon
    at every step regardless of remaining max_steps."""
    import jax.numpy as jnp
    env = DoorCodeEnv(K=3, probe_noise=0.0)

    seen_horizons: list[int] = []

    class HorizonRecorder:
        def plan(self, q_state, q_static, T, B_list, goal, horizon, theta_mask=None, **_):
            seen_horizons.append(int(horizon))
            n_actions = T.shape[3]
            d = np.zeros(n_actions, dtype=np.float32)
            d[0] = 1.0
            return jnp.asarray(d)

    rng = np.random.default_rng(0)
    run_episode(
        env=env, planner=HorizonRecorder(),
        q_static_init=jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static,
        theta_true=0, horizon=4, max_steps=4,
        deterministic=True, rng=rng,
        clip_horizon=False,
    )
    assert seen_horizons == [4, 4, 4, 4], f"got {seen_horizons}"


def test_episode_result_serializable(planner: AIFPlanner) -> None:
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    block = run_block(
        env, planner, theta_true=0, n_episodes=1,
        horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    ep = block.episodes[0]
    assert isinstance(ep.final_q_static, np.ndarray)
    assert ep.final_q_static.shape == (env.n_static,)
    np.testing.assert_allclose(ep.final_q_static.sum(), 1.0, atol=1e-5)
