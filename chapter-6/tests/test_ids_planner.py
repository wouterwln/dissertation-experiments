"""Tests for IDSPlanner — Information-Directed Sampling baseline.

Covers:
- plan() returns a normalised one-hot distribution.
- Sharp q(θ) collapses I(a)=0 ⇒ IDS reduces to BAMCP argmax-Q.
- Info gain is exactly zero on θ-independent channels (synthetic env).
- Info gain is strictly positive under a diffuse prior on door-code
  (the probe channel must move q(θ) for the baseline to be defensible
  as 'information-directed').
- run_outer_loop integration.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_outer_loop
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.ids_planner import IDSPlanner


def _make_ids(n_simulations: int = 64, depth: int = 4, seed: int = 0) -> IDSPlanner:
    return IDSPlanner(
        bamcp=BAMCPPlanner(
            n_simulations=n_simulations, depth=depth, rollout_seed=seed,
            resample_theta_per_sim=True,
        ),
        epsilon_info=1.0e-6,
    )


def _door_code_args(env: DoorCodeEnv):
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32)
        for B_c in env.get_observation_tensors()
    ]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    return T, B_list, goal


def test_sharp_posterior_matches_bamcp_argmax() -> None:
    """With q(θ) one-hot, I(a)=0 ∀a → ratio is monotone in Δ → argmin
    is Q-greedy, which is BAMCP's argmax-Q. IDS must match BAMCP's
    selection. Uses identical RNG seeds and sim budgets on both
    planners so the Q estimates align."""
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1)
    T, B_list, goal = _door_code_args(env)
    theta_mask = env.get_modality_registry().theta_dependent_mask
    common = dict(
        n_simulations=256, depth=4, rollout_seed=0, resample_theta_per_sim=True,
    )

    for theta_idx in [0, 4]:
        env.reset(theta=theta_idx)
        correct = env.correct_door
        q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
        q_static = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_idx].set(1.0)

        ids = IDSPlanner(BAMCPPlanner(**common), epsilon_info=1.0e-6)
        ids_dist = np.asarray(ids.plan(
            q_state, q_static, T, B_list, goal, horizon=4,
            theta_mask=theta_mask,
            theta_spec=env.theta_spec,
            t_factor_spec=env.t_factor_spec,
            b_factor_specs=env.b_factor_specs,
        ))
        # Under a sharp prior the optimal action is open(correct_door).
        # IDS should pick that since I=0 and BAMCP's Q is highest there.
        assert int(ids_dist.argmax()) == env.M + correct, (
            f"theta={theta_idx}: IDS expected OPEN_DOOR_{correct} "
            f"(action {env.M + correct}); got {int(ids_dist.argmax())} "
            f"dist={ids_dist}"
        )


def test_info_gain_zero_when_neither_T_nor_B_depend_on_theta() -> None:
    """If both T and every B channel are θ-independent, no observation
    can move q(θ) — I(a) must be exactly zero for every action and
    IDS falls back to Q-greedy via the ratio's degenerate limit."""
    n_states = 3
    n_actions = 2
    n_static = 2

    # T is θ-independent. From state 0, action 0 → state 1, action 1 →
    # state 2. States 1 and 2 are absorbing self-loops.
    T = np.zeros((n_states, n_states, n_actions), dtype=np.float64)
    for action in range(n_actions):
        T[1 if action == 0 else 2, 0, action] = 1.0
        T[1, 1, action] = 1.0
        T[2, 2, action] = 1.0

    # Two θ-independent channels.
    B_pos = np.eye(n_states, dtype=np.float64)
    B_outcome = np.zeros((3, n_states), dtype=np.float64)
    B_outcome[0, :] = 1.0  # NEUTRAL everywhere
    B_list = [B_pos, B_outcome]

    theta_spec = ThetaSpec(component_sizes=(n_static,))
    t_factor_spec = FactorSpec(theta_components=())  # θ-indep T
    b_factor_specs = (
        FactorSpec(theta_components=()),  # B_pos θ-independent
        FactorSpec(theta_components=()),  # B_outcome θ-independent
    )

    ids = _make_ids(n_simulations=8, depth=2)
    info_gain = ids._info_gain_per_action(
        q_state=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        q_static=np.ones(n_static) / n_static,
        T=T, B_list=B_list, n_actions=n_actions,
        theta_spec=theta_spec,
        t_factor_spec=t_factor_spec,
        b_factor_specs=b_factor_specs,
    )
    np.testing.assert_allclose(info_gain, 0.0, atol=1e-12)


def test_theta_indep_obs_still_informs_under_theta_dep_T() -> None:
    """The reverse: under θ-dependent T with θ-independent B, an
    observation that pins down x_new still moves q(θ) because different
    θs route to different x_new under the same action. Skipping these
    channels would zero out info gain on actions whose discriminative
    power flows through T (e.g. door-opening outcomes). This is the
    test that pins the original bug fix.
    """
    n_states = 3
    n_actions = 2
    n_static = 2

    # T is θ-dependent: action 0 under θ=0 → state 1; under θ=1 → state 2.
    # Action 1 is symmetric (swap). State 1 and 2 self-loop.
    T = np.zeros((n_states, n_states, n_static, n_actions), dtype=np.float64)
    T[1, 0, 0, 0] = 1.0
    T[2, 0, 1, 0] = 1.0
    T[2, 0, 0, 1] = 1.0
    T[1, 0, 1, 1] = 1.0
    for theta in range(n_static):
        for action in range(n_actions):
            T[1, 1, theta, action] = 1.0
            T[2, 2, theta, action] = 1.0

    # θ-independent observations — but B_pos uniquely identifies x_new.
    B_pos = np.eye(n_states, dtype=np.float64)
    B_list = [B_pos]

    theta_spec = ThetaSpec(component_sizes=(n_static,))
    t_factor_spec = FactorSpec(theta_components=(0,))
    b_factor_specs = (FactorSpec(theta_components=()),)

    ids = _make_ids(n_simulations=8, depth=2)
    info_gain = ids._info_gain_per_action(
        q_state=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        q_static=np.ones(n_static) / n_static,
        T=T, B_list=B_list, n_actions=n_actions,
        theta_spec=theta_spec,
        t_factor_spec=t_factor_spec,
        b_factor_specs=b_factor_specs,
    )
    # Observing x_new=1 vs x_new=2 perfectly identifies θ → KL is log(2)
    # weighted by the marginal p(x_new=1) = p(x_new=2) = 0.5.
    # I(a) = log(2) for each action.
    np.testing.assert_allclose(info_gain, np.log(2.0), atol=1e-8)


def test_info_gain_positive_under_diffuse_prior_door_code() -> None:
    """On door-code under a uniform prior, at least one action must
    have strictly positive information gain — otherwise the IDS ratio
    is uniformly Δ²/ε and the 'information-directed' label is empty.
    Specifically, probe actions move q(θ) (the probe channel is
    θ-conditional) so I(probe) > 0."""
    env = DoorCodeEnv(K=3, probe_noise=0.2)
    T_np = np.asarray(env.get_transition_tensor(), dtype=np.float64)
    B_list_np = [
        np.asarray(B_c, dtype=np.float64) for B_c in env.get_observation_tensors()
    ]
    q_state_np = np.zeros(env.n_states, dtype=np.float64)
    q_state_np[env._START] = 1.0
    q_static_np = np.ones(env.n_static, dtype=np.float64) / env.n_static

    ids = _make_ids(n_simulations=8, depth=2)
    info_gain = ids._info_gain_per_action(
        q_state=q_state_np, q_static=q_static_np,
        T=T_np, B_list=B_list_np, n_actions=env.n_actions,
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
    )
    assert info_gain.max() > 1e-3, (
        f"no action has appreciable information gain on door-code "
        f"under a diffuse prior — info_gain={info_gain}"
    )
    # And info gain must be non-negative everywhere (it's an expected KL).
    assert (info_gain >= -1e-9).all(), f"negative info gain detected: {info_gain}"


def test_run_outer_loop_with_ids() -> None:
    """Full runner integration — IDS plays through several episodes on
    door-code. Smoke-test that prepare_episode / plan are correctly
    invoked and that the planner doesn't crash mid-block."""
    ids = _make_ids(n_simulations=32, depth=4)
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.2),
        planner=ids, n_seeds=2, n_episodes=3,
        horizon=4, max_steps=4, deterministic=True, seed=0,
        show_progress=False,
    )
    assert result.success.shape == (2, 3)
