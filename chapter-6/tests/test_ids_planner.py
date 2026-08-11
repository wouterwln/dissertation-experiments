"""Tests for IDSPlanner — Information-Directed Sampling baseline.

Covers:
- plan() returns a normalised one-hot distribution.
- Sharp q(θ) collapses I(a)=0 ⇒ IDS reduces to BAMCP argmax-Q.
- The information ratio is a no-op on every *reachable* Δ, so IDS's
  selection is its BAMCP's argmax-Q. This is a limitation the planner's
  docstring states plainly, and pinning it here is what stops a future
  edit from re-advertising a stochastic policy that cannot exist under
  ``Δ(a) = max_a' Q(a') - Q(a)``.
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


# --- The information ratio on reachable inputs --------------------------------
#
# `plan()` builds Δ as `q_values.max() - q_values`, so Δ ALWAYS contains an
# exact 0.0 at the greedy action (float64 x - x). Δ²/(I+ε) is non-negative,
# so that action attains the global minimum 0 and no information term can
# outrank it. An earlier revision of this file exercised the ratio only on
# hand-built deltas with no zero entry (e.g. [0.10, 1.00]) — inputs `plan()`
# cannot produce — and concluded from them that IDS played a genuine
# mixture. The two tests below use reachable deltas instead.


def test_information_ratio_is_a_no_op_on_reachable_deltas() -> None:
    """On a Δ built the way ``plan`` builds it, I(a) cannot move the action.

    Reachable Δ means ``Δ = max(Q) - Q``, which contains an exact 0.0.
    Randomised over Q and over information vectors including adversarial
    ones (huge info on a non-greedy action, zero info on the greedy one):
    the argmin of Δ²/(I+ε) is the lowest-index greedy action every time.
    So the objective's only content is the tie-break, and the docstring's
    claim that IDS's rule reduces to argmax-Q is exact, not approximate.
    """
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(2, 9))
        q_values = rng.normal(size=n)
        delta = q_values.max() - q_values
        assert (delta == 0.0).any(), "reachable Δ must contain an exact zero"

        info = np.abs(rng.normal(size=n)) * 10.0 ** rng.integers(-3, 4)
        info[int(rng.integers(0, n))] = 0.0
        denom = info + 1e-6

        ratio = (delta * delta) / denom
        assert int(np.argmin(ratio)) == int(np.argmax(q_values)), (
            f"the information term changed the selection: Q={q_values}, "
            f"I={info}, ratio={ratio}"
        )


def test_plan_selects_its_own_bamcp_argmax_q() -> None:
    """End-to-end: IDS's action is the argmax-Q of the tree it composed.

    This is the honest statement of reviewer item 9's residue — IDS is a
    different Monte-Carlo draw of BAMCP's estimator, not a different
    decision rule — and it is the regression guard: if a future edit gives
    IDS a Δ that can actually reach the mixture regime (per-θ root Q), this
    test is the one that must be rewritten alongside the docstring.
    """
    env = DoorCodeEnv(K=3, probe_noise=0.2)
    T, B_list, goal = _door_code_args(env)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    common = dict(
        n_simulations=256, depth=4, rollout_seed=7919,
        resample_theta_per_sim=True,
    )
    plan_args = dict(
        theta_mask=env.get_modality_registry().theta_dependent_mask,
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
    )

    ids = IDSPlanner(BAMCPPlanner(**common), epsilon_info=1.0e-6)
    dist = np.asarray(ids.plan(
        q_state, q_static, T, B_list, goal, horizon=4, **plan_args,
    ), dtype=np.float64)

    # A freshly seeded twin reproduces the same tree, so its root Q is the
    # Q that IDS just minimised the ratio over.
    q_ref, _ = BAMCPPlanner(**common).plan(
        q_state, q_static, T, B_list, goal, horizon=4,
        return_q_values=True, **plan_args,
    )
    assert np.count_nonzero(dist) == 1, f"not a one-hot: {dist}"
    assert int(dist.argmax()) == int(np.asarray(q_ref).argmax()), (
        f"IDS picked {int(dist.argmax())} but its BAMCP's argmax-Q is "
        f"{int(np.asarray(q_ref).argmax())} (Q={np.asarray(q_ref)})"
    )


def test_plan_returns_normalised_distribution_on_door_code() -> None:
    """End-to-end shape/normalisation contract through the real planner."""
    env = DoorCodeEnv(K=3, probe_noise=0.2)
    T, B_list, goal = _door_code_args(env)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    ids = _make_ids(n_simulations=64, depth=4)
    dist = np.asarray(ids.plan(
        q_state, q_static, T, B_list, goal, horizon=4,
        theta_mask=env.get_modality_registry().theta_dependent_mask,
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
    ), dtype=np.float64)
    assert dist.shape == (env.n_actions,)
    assert np.all(dist >= -1e-6)
    assert dist.sum() == pytest.approx(1.0, abs=1e-5)


def test_independent_rollout_seed_decorrelates_ids_from_bamcp() -> None:
    """IDS constructed with its own rollout_seed must not reproduce the
    BAMCP arm's tree. Together with ``ids_bamcp_depth`` this is the ONLY
    thing separating the two arms — the information ratio cannot (see
    ``test_information_ratio_is_a_no_op_on_reachable_deltas``) — so it is
    all that remains of reviewer item 9."""
    env = DoorCodeEnv(K=3, probe_noise=0.2)
    T, B_list, goal = _door_code_args(env)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    shared = BAMCPPlanner(
        n_simulations=256, depth=4, rollout_seed=0, resample_theta_per_sim=True,
    )
    independent = BAMCPPlanner(
        n_simulations=256, depth=4, rollout_seed=7919,  # --ids-rollout-seed
        resample_theta_per_sim=True,
    )
    q_shared, _ = shared.plan(
        q_state, q_static, T, B_list, goal, horizon=4, return_q_values=True,
    )
    q_independent, _ = independent.plan(
        q_state, q_static, T, B_list, goal, horizon=4, return_q_values=True,
    )
    assert not np.allclose(
        np.asarray(q_shared), np.asarray(q_independent), atol=1e-6,
    ), "distinct rollout_seed produced an identical Q estimate"


def test_prepare_block_forwards_to_the_composed_bamcp() -> None:
    """IDS must forward block boundaries, or its BAMCP's rollout Q-hat
    would outlive the theta_true it was learned under."""
    ids = _make_ids(n_simulations=16, depth=4)
    ids._bamcp.rollout_policy = "eps_greedy_q"
    ids._bamcp._q_rollout = np.zeros((3, 3), dtype=np.float64)
    ids.prepare_block(
        jnp.ones(6, dtype=jnp.float32) / 6, np.random.default_rng(0),
    )
    assert ids._bamcp._q_rollout is None


# --- reachable-state restriction: equivalence against a brute-force oracle ---
#
# `_info_gain_per_action` restricts every state axis to the states whose
# predictive mass exceeds `_STATE_MASS_TOL`, because the unrestricted
# `unnorm` tensor is 398 MB at RockSample K=4 while all but a handful of its
# state slices hold ~3e-12 -- the floor `bayesian_state_update`'s softmax
# leaves on analytically-impossible states. That is a tolerance, not an
# identity, so it needs a reference independent of the code path it
# validates: comparing the production path against itself under a perturbed
# model cannot do that.
#
# `_brute_force_info_gain` below is therefore a direct transcription of
# I(a) = E_o[KL(q(θ|o,a) || q(θ))]: enumerate the full joint observation
# space over the full state space, no restriction, no chunking, no
# vectorised broadcast tricks. It validates the restriction *and* the
# vectorised algebra around it.


def _channel_matrix(B_c, spec, theta_spec, obs: int, n_states: int) -> np.ndarray:
    """`(n_states, n_static)` likelihood of `obs` on one channel."""
    B_c = np.asarray(B_c, dtype=np.float64)
    n_static = theta_spec.n_static
    out = np.empty((n_states, n_static), dtype=np.float64)
    for theta_flat in range(n_static):
        components = theta_spec.unravel(theta_flat)
        if spec.is_theta_independent:
            out[:, theta_flat] = B_c[obs]
        else:
            idx = tuple(components[k] for k in spec.theta_components)
            out[:, theta_flat] = B_c[(obs, slice(None), *idx)]
    return out


def _brute_force_info_gain(
    q_state, q_static, T, B_list, n_actions,
    theta_spec, t_factor_spec, b_factor_specs,
) -> np.ndarray:
    """I(a) = E_o[KL(q(θ|o,a) || q(θ))], enumerated exhaustively."""
    import itertools

    T = np.asarray(T, dtype=np.float64)
    q_state = np.asarray(q_state, dtype=np.float64)
    n_states = q_state.shape[0]
    n_static = theta_spec.n_static
    q_theta = np.asarray(q_static, dtype=np.float64).reshape(-1)
    q_theta = q_theta / q_theta.sum()

    cards = [np.asarray(b).shape[0] for b in B_list]
    # Per (channel, obs) likelihood, so the tuple loop is a product of
    # precomputed (n_states, n_static) matrices.
    mats = [
        [
            _channel_matrix(B_list[c], b_factor_specs[c], theta_spec, o, n_states)
            for o in range(cards[c])
        ]
        for c in range(len(B_list))
    ]

    info = np.zeros(n_actions, dtype=np.float64)
    for a in range(n_actions):
        # p(x' , θ | a) with no factorisation shortcuts.
        joint = np.zeros((n_states, n_static), dtype=np.float64)
        for theta_flat in range(n_static):
            components = theta_spec.unravel(theta_flat)
            if t_factor_spec.is_theta_independent:
                T_a = T[:, :, a]
            else:
                idx = tuple(components[k] for k in t_factor_spec.theta_components)
                T_a = T[(slice(None), slice(None), *idx, a)]
            joint[:, theta_flat] = (T_a @ q_state) * q_theta[theta_flat]
        joint /= joint.sum()

        for obs_tuple in itertools.product(*[range(c) for c in cards]):
            lik = np.ones((n_states, n_static), dtype=np.float64)
            for c, o in enumerate(obs_tuple):
                lik = lik * mats[c][o]
            unnorm = lik * joint
            p_o = unnorm.sum()
            if p_o <= 0.0:
                continue
            q_new = unnorm.sum(axis=0) / p_o
            nz = q_new > 0.0
            kl = float(
                (q_new[nz] * np.log(q_new[nz] / q_theta[nz])).sum()
            )
            info[a] += p_o * max(kl, 0.0)
    return info


def _oracle_case(env, q_state=None):
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(b, dtype=jnp.float32)
              for b in env.get_observation_tensors()]
    if q_state is None:
        q_state = np.zeros(env.n_states)
        q_state[env.start_state_index] = 1.0
    q_static = np.ones(env.theta_spec.n_static) / env.theta_spec.n_static
    specs = dict(
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
    )
    got = np.asarray(IDSPlanner(
        bamcp=BAMCPPlanner(n_simulations=1, depth=1),
    )._info_gain_per_action(
        jnp.asarray(q_state, dtype=jnp.float32),
        jnp.asarray(q_static, dtype=jnp.float32),
        T, B_list, env.n_actions, **specs,
    ))
    want = _brute_force_info_gain(
        q_state, q_static, np.asarray(T, dtype=np.float64),
        [np.asarray(b, dtype=np.float64) for b in B_list],
        env.n_actions, **specs,
    )
    # The production path floors probabilities at EPSILON=1e-12 in two
    # places, so agreement is to ~1e-11 absolute, not exact. Values here
    # are O(0.1), so this still pins every digit that matters.
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)
    return got


@pytest.mark.parametrize("lock_length", [1, 2])
def test_info_gain_matches_oracle_theta_dep_T(lock_length: int) -> None:
    """door_code: theta-DEPENDENT T branch of the reachability mask."""
    env = DoorCodeEnv(K=3, probe_noise=0.2, lock_length=lock_length)
    env.reset(theta=0)
    _oracle_case(env)


def test_info_gain_matches_oracle_theta_indep_T_paired_components() -> None:
    """RockSample: theta-INDEPENDENT T, channels seeing component *pairs*.

    This is the shape the restriction was written for -- the unrestricted
    tensor is (*inner_cards, n_states, *component_sizes), 398 MB at the
    canonical K=4. Pinned at K=2 to keep the oracle affordable.
    """
    from learningpomdp.environments.rock_sample import RockSampleEnv

    env = RockSampleEnv(K=2, grid_size=4, sensor_noise_eta=0.3)
    env.reset(seed=0)
    info = _oracle_case(env)
    # Physically: MOVE and SAMPLE emit `no_info`, only SENSE can inform θ.
    K = 2
    np.testing.assert_allclose(info[:4], 0.0, atol=1e-9)
    assert (info[4:4 + K] > 1e-6).all()
    assert info[env.n_actions - 1] == pytest.approx(0.0, abs=1e-9)


def test_info_gain_matches_oracle_factored_tiger() -> None:
    """Factored Tiger: singleton theta-dep channels, plus an absolute check.

    OPEN_k resolves theta_k outright, so under a uniform prior over a
    binary component its expected KL is exactly ln 2. That is an absolute
    anchor: a restriction that silently dropped contributing states would
    land below it, which a self-consistency test could not detect.
    """
    from learningpomdp.environments.factored_tiger import FactoredTigerEnv

    K = 2
    env = FactoredTigerEnv(K=K, listen_accuracy=0.85)
    env.reset(seed=0)
    info = _oracle_case(env)
    np.testing.assert_allclose(info[K:], np.log(2.0), rtol=1e-6)
    assert (info[:K] < np.log(2.0)).all(), "a noisy listen cannot beat a reveal"


def test_restriction_keeps_full_support_on_a_spread_belief() -> None:
    """A spread q(x) is the MiniGrid regime: nothing may be dropped.

    Every state carries mass, so the mask must retain the whole axis and
    the result must still match the oracle.
    """
    env = DoorCodeEnv(K=3, probe_noise=0.2, lock_length=1)
    env.reset(theta=0)
    _oracle_case(env, q_state=np.ones(env.n_states) / env.n_states)


def test_state_mass_tolerance_drops_only_the_filter_floor() -> None:
    """A sub-tolerance tail must be dropped, and must not move I(a).

    `_STATE_MASS_TOL` is a heuristic, so its effect is measured rather than
    asserted: build the belief the filter actually produces -- one state at
    ~1.0 plus the softmax's ~3e-12 floor everywhere else -- and check the
    pruned result still matches the unpruned brute-force oracle. Also pins
    that the tolerance is doing something, so a future change that made it
    inert would fail loudly instead of silently costing 59% of the slate.
    """
    from learningpomdp.planning.ids_planner import _STATE_MASS_TOL

    env = DoorCodeEnv(K=3, probe_noise=0.2, lock_length=1)
    env.reset(theta=0)
    q_state = np.full(env.n_states, 3e-12)
    q_state[env.start_state_index] = 1.0 - 3e-12 * (env.n_states - 1)
    assert (q_state < _STATE_MASS_TOL).sum() == env.n_states - 1

    # The oracle enumerates every state, tail included; the production path
    # drops the tail. Agreement is the claim the tolerance rests on.
    _oracle_case(env, q_state=q_state)
