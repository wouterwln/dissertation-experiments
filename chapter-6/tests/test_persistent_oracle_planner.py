"""Tests for PersistentOraclePlanner — θ_persistent given, θ_episodic inferred.

Covers:
- prepare_episode's runner contract (theta_true + theta_spec required).
- plan() raises if prepare_episode was never called.
- Degenerates to the full OraclePlanner when every θ component is
  persistent (door_code) — the reason it is wired on rock_sample only.
- Warns, instead of silently degrading, when the filter underflows the
  true persistent value to zero mass.
- On rock_sample the persistent axes are pinned to the truth at every
  step while the episodic posterior still moves within the episode.
- Opt-in (`-m slow`): where the two arms land relative to the full Oracle
  and AIF-MP on cumulative reward. Ordering only, and measured — these
  are reference arms, not bounds (see the planner's module docstring).
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.environments.rock_sample import RockSampleEnv
from learningpomdp.experiment.rock_sample_runner import run_block, run_outer_loop
from learningpomdp.factors import ThetaSpec
from learningpomdp.planning import GPU_PLANNER_TYPES
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.oracle_planner import OraclePlanner
from learningpomdp.planning.persistent_oracle_planner import PersistentOraclePlanner
from learningpomdp.planning.vbp_planner import VBPPlanner


class _RecordingBase:
    """Base-planner proxy that records the q_static handed to it.

    With ``base=None`` it also stands in for the inner solve, so tests can
    exercise the conditioning without paying for a BP pass (and may pass
    None for the model tensors, which are only forwarded).
    """

    def __init__(self, base=None) -> None:
        self._base = base
        self.seen: list[np.ndarray] = []

    def plan(self, q_state, q_static, T, B_list, goal, horizon, **kwargs):
        self.seen.append(np.asarray(q_static, dtype=np.float64))
        if self._base is None:
            return jnp.ones(2, dtype=jnp.float32) / 2
        return self._base.plan(
            q_state, q_static, T, B_list, goal, horizon, **kwargs,
        )


# ---------- Runner contract ----------

def test_registered_as_gpu_planner_type() -> None:
    """The rock_sample runner downgrades to workers=1 for GPU_PLANNER_TYPES.
    Missing from that tuple, this planner would be handed to a spawn pool
    and re-init JAX per worker.
    """
    assert isinstance(
        PersistentOraclePlanner(VBPPlanner(n_iterations=10)), GPU_PLANNER_TYPES,
    )


def test_prepare_episode_requires_theta_true_and_theta_spec() -> None:
    planner = PersistentOraclePlanner(VBPPlanner(n_iterations=10))
    rng = np.random.default_rng(0)
    spec = ThetaSpec(component_sizes=(2, 2), episode_reset_components=(0,))
    q_static = jnp.ones(4, dtype=jnp.float32) / 4

    with pytest.raises(RuntimeError, match="theta_true and theta_spec"):
        planner.prepare_episode(q_static, rng, theta_spec=spec)
    with pytest.raises(RuntimeError, match="theta_true and theta_spec"):
        planner.prepare_episode(q_static, rng, theta_true=0)
    with pytest.raises(RuntimeError, match="theta_true and theta_spec"):
        planner.prepare_episode(q_static, rng)


def test_plan_raises_before_prepare_episode() -> None:
    planner = PersistentOraclePlanner(VBPPlanner(n_iterations=10))
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    with pytest.raises(RuntimeError, match="prepare_episode"):
        planner.plan(
            q_state, q_static, T, B_list, goal, horizon=4,
            theta_spec=env.theta_spec,
        )


# ---------- Degeneracy when θ is purely persistent ----------

def test_equals_full_oracle_when_all_components_persistent() -> None:
    """With no episodic components there is nothing left to infer, so
    conditioning on θ_true_persistent collapses q(θ) onto θ_true and the
    two oracles must plan identically. This is why persistent_oracle is
    wired on rock_sample only — door_code, minigrid and factored_tiger
    all land in this branch.
    """
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1)
    assert env.theta_spec.episode_reset_components == (), (
        "premise of this test: door_code's θ is purely persistent"
    )
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
    rng = np.random.default_rng(0)

    # A deliberately misleading belief: sharp on the wrong θ. Both oracles
    # must ignore it.
    for theta_true in [0, 2, 4]:
        wrong = (theta_true + 1) % env.n_static
        q_static = jnp.zeros(env.n_static, dtype=jnp.float32).at[wrong].set(1.0)

        plan_kwargs = dict(
            theta_mask=theta_mask,
            theta_spec=env.theta_spec,
            t_factor_spec=env.t_factor_spec,
            b_factor_specs=env.b_factor_specs,
        )
        oracle = OraclePlanner(VBPPlanner(n_iterations=80))
        oracle.prepare_episode(q_static, rng, theta_true=theta_true)
        expected = np.asarray(oracle.plan(
            q_state, q_static, T, B_list, goal, 4, **plan_kwargs,
        ))

        persistent = PersistentOraclePlanner(VBPPlanner(n_iterations=80))
        persistent.prepare_episode(
            q_static, rng, theta_true=theta_true, theta_spec=env.theta_spec,
        )
        got = np.asarray(persistent.plan(
            q_state, q_static, T, B_list, goal, 4, **plan_kwargs,
        ))

        np.testing.assert_allclose(got, expected, atol=1e-5)


# ---------- Zero-mass on the pinned truth ----------

def test_warns_when_true_persistent_value_has_zero_mass() -> None:
    """`_condition_on_persistent` answers zero mass with a *uniform*
    episodic conditional, which throws away the within-episode SENSE
    evidence. PSRL cannot reach that branch (it samples the persistent
    value from q, so it always has mass) but pinning to the truth can,
    since the filter's float32 softmax can underflow it — so it must warn
    rather than quietly hand the inner solve a weaker belief.
    """
    spec = ThetaSpec(component_sizes=(2, 2, 2, 2), episode_reset_components=(2, 3))
    theta_true = spec.ravel((0, 1, 0, 1))
    base = _RecordingBase()
    planner = PersistentOraclePlanner(base)
    planner.prepare_episode(
        jnp.ones(spec.n_static, dtype=jnp.float32) / spec.n_static,
        np.random.default_rng(0),
        theta_true=theta_true, theta_spec=spec,
    )

    # All mass on the wrong sensor pattern — as an underflowed softmax
    # would leave it. The model tensors are unused by _RecordingBase.
    underflowed = jnp.zeros(spec.n_static, dtype=jnp.float32).at[
        spec.ravel((1, 0, 0, 0))
    ].set(1.0)
    with pytest.warns(RuntimeWarning, match="zero mass on the true"):
        planner.plan(None, underflowed, None, None, None, 3, theta_spec=spec)
    episodic = base.seen[-1].reshape(spec.component_sizes)[0, 1]
    np.testing.assert_allclose(episodic, np.full((2, 2), 0.25), atol=1e-6)

    # With mass on the truth there is nothing to warn about.
    ok = jnp.zeros(spec.n_static, dtype=jnp.float32).at[theta_true].set(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        planner.plan(None, ok, None, None, None, 3, theta_spec=spec)


# ---------- Persistent pinned, episodic free ----------

def test_persistent_pinned_while_episodic_posterior_moves() -> None:
    """Every step the inner planner must see persistent mass 1.0 on the
    true sensor pattern, and the episodic (rock-quality) marginal must
    still change as observations arrive — otherwise SENSE would be
    worthless and this would just be the full Oracle again.
    """
    env = RockSampleEnv(K=2, grid_size=3, max_steps=12)
    spec = env.theta_spec
    episodic = tuple(spec.episode_reset_components)
    persistent = spec.persistent_components
    assert episodic and persistent, "premise: rock_sample θ has both kinds"

    theta_true = spec.ravel((0, 1, 0, 1))
    base = _RecordingBase(VBPPlanner(n_iterations=40))
    run_block(
        env=env,
        planner=PersistentOraclePlanner(base),
        block_theta_true=theta_true,
        n_episodes=1,
        # horizon=5 is deep enough that the inner VBP solve actually
        # chooses to SENSE on this instance; at horizon=3 it never does
        # and there is no episodic evidence to move the posterior with.
        horizon=5,
        max_steps=12,
        deterministic=True,
        seed=0,
    )
    assert len(base.seen) > 1, "need several within-episode steps to compare"

    truth = spec.unravel(theta_true)
    true_persistent = tuple(truth[k] for k in persistent)
    episodic_margs = []
    for q in base.seen:
        qf = q.reshape(spec.component_sizes)
        assert qf.sum(axis=episodic)[true_persistent] == pytest.approx(1.0, abs=1e-5)
        episodic_margs.append(qf.sum(axis=persistent))

    drift = max(
        float(np.abs(m - episodic_margs[0]).max()) for m in episodic_margs[1:]
    )
    assert drift > 1e-3, (
        f"episodic posterior never moved within the episode (drift={drift})"
    )


# ---------- Position relative to Oracle and AIF-MP ----------

@pytest.mark.slow
def test_cumulative_reward_ordering_of_both_arms() -> None:
    """Where the arms land on mean cumulative reward. `persistent_oracle`
    (AIF inner) differs from `aif` in exactly one respect — θ_persistent
    is handed to it — and still lacks θ_episodic, so it should sit between
    `aif` and the full Oracle. This asserts that ordering only, not a
    decomposition of the interval: the `aif`→`persistent_oracle` margin is
    the value of knowing θ_persistent, while `persistent_oracle`→`oracle`
    also switches the inner solver (AIF→VBP) and so measures nothing
    clean. The `_vbp` arm only has to fall short of the full Oracle: it
    never scores above plain `vbp`, so it bounds nothing (planner
    docstring).

    Measured at K=2, grid_size=3, horizon=5, 8 seeds x 2 episodes
    (deterministic, seed=0): oracle 1.31, persistent_oracle 1.13, aif
    0.63, persistent_oracle_vbp 0.56 — margins +0.19 (oracle over
    persistent) and +0.50 (persistent over aif) on a ±1 per-SAMPLE reward
    scale, and the same ordering holds at 2, 4, 6 and 12 seeds. Opt-in
    `slow` (4-7 min on CPU): a measured property of two approximate
    planners, not a law, so it does not gate every test run.
    """
    common = dict(
        env_kwargs=dict(K=2, grid_size=3, max_steps=12),
        n_seeds=8, n_episodes=2, horizon=5,
        max_steps=12, deterministic=True, seed=0, show_progress=False,
    )

    def mean_cumulative_reward(planner) -> float:
        result = run_outer_loop(planner=planner, **common)
        return float(np.mean(result.cumulative_reward))

    oracle = mean_cumulative_reward(OraclePlanner(VBPPlanner(n_iterations=40)))
    persistent = mean_cumulative_reward(
        PersistentOraclePlanner(AIFPlanner(n_iterations=40))
    )
    aif = mean_cumulative_reward(AIFPlanner(n_iterations=40))
    persistent_vbp = mean_cumulative_reward(
        PersistentOraclePlanner(VBPPlanner(n_iterations=40))
    )

    assert aif < persistent < oracle, (
        f"expected aif < persistent_oracle < oracle, got aif={aif:.4f} "
        f"persistent_oracle={persistent:.4f} oracle={oracle:.4f}"
    )
    assert persistent_vbp < oracle, (
        f"expected persistent_oracle_vbp < oracle, got "
        f"persistent_oracle_vbp={persistent_vbp:.4f} oracle={oracle:.4f}"
    )
