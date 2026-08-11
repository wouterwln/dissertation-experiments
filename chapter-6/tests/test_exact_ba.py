"""Tests for the exact Bayes-adaptive expectimax reference."""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_outer_loop
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.planning.exact_ba import (
    ExactBAPlanner,
    NodeCapExceeded,
    optimal_action_set,
    solve_exact,
)


# --- Fixtures / builders ----------------------------------------------------


def _uninformative_kwargs(n_states: int, n_actions: int, n_theta: int) -> dict:
    """Single-state-observation model: T self-loops, one flat θ component."""
    T = np.zeros((n_states, n_states, n_actions))
    T[0, 0, :] = 1.0
    return dict(
        T=jnp.asarray(T, dtype=jnp.float32),
        B_list=[jnp.asarray(np.eye(n_states), dtype=jnp.float32)],
        theta_spec=ThetaSpec(component_sizes=(n_theta,)),
        t_factor_spec=FactorSpec(theta_components=()),
        b_factor_specs=(FactorSpec(theta_components=()),),
    )


def _mdp_kwargs(T_sa: np.ndarray) -> dict:
    """Fully observed MDP: identity observation channel, singleton θ."""
    n_states = T_sa.shape[0]
    return dict(
        T=jnp.asarray(T_sa, dtype=jnp.float32),
        B_list=[jnp.asarray(np.eye(n_states), dtype=jnp.float32)],
        theta_spec=ThetaSpec(component_sizes=(1,)),
        t_factor_spec=FactorSpec(theta_components=()),
        b_factor_specs=(FactorSpec(theta_components=()),),
    )


def _door_code_reward_table(env: DoorCodeEnv) -> np.ndarray:
    arrival = np.zeros(env.n_states)
    arrival[env.success_state_index] = 1.0
    arrival[env.failure_state_index] = -1.0
    table = np.einsum("nsta,n->sat", env.get_transition_tensor(), arrival)
    table[env.success_state_index] = 0.0
    table[env.failure_state_index] = 0.0
    return table


def _door_code_setup(K: int = 3, lock_length: int = 1, probe_noise: float = 0.2):
    env = DoorCodeEnv(K=K, lock_length=lock_length, probe_noise=probe_noise)
    kwargs = dict(
        T=jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32),
        B_list=[
            jnp.asarray(B_c, dtype=jnp.float32)
            for B_c in env.get_observation_tensors()
        ],
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
        reward_table=_door_code_reward_table(env),
        terminal_states=(env.success_state_index, env.failure_state_index),
    )
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[
        env.start_state_index
    ].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    return env, kwargs, q_state, q_static


# --- 1-step values against hand computation ---------------------------------


def test_one_step_value_matches_hand_computation():
    """Single-state Tiger decision: LISTEN 0, OPEN 0.5·10 + 0.5·(-100)."""
    kwargs = _uninformative_kwargs(n_states=1, n_actions=2, n_theta=2)
    reward_table = np.zeros((1, 2, 2))
    reward_table[0, 1, 0] = -100.0  # OPEN a tiger door
    reward_table[0, 1, 1] = 10.0    # OPEN a safe door

    solution = solve_exact(
        jnp.ones(1, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32) / 2.0,
        remaining=1,
        reward_table=reward_table,
        **kwargs,
    )

    assert solution.n_nodes == 1
    np.testing.assert_allclose(solution.action_values, [0.0, -45.0], atol=1e-6)
    assert solution.value == pytest.approx(0.0, abs=1e-6)


def test_one_step_door_code_value_matches_hand_computation():
    """From START under a uniform permutation prior, a blind OPEN on K=3
    is right for 2 of 6 permutations: 1/3 · (+1) + 2/3 · (-1) = -1/3.
    Probing is free, so the 1-step optimum is 0."""
    env, kwargs, q_state, q_static = _door_code_setup(K=3, lock_length=1)

    solution = solve_exact(q_state, q_static, remaining=1, **kwargs)

    values = solution.action_values
    np.testing.assert_allclose(values[: env.M], 0.0, atol=1e-6)
    np.testing.assert_allclose(values[env.M:], -1.0 / 3.0, atol=1e-6)
    assert solution.value == pytest.approx(0.0, abs=1e-6)


# --- The theta-posterior backup against an independent brute force ----------


def _brute_force_value(
    joint: np.ndarray,
    T_sta: np.ndarray,
    B_mats: list[np.ndarray],
    reward: np.ndarray,
    remaining: int,
) -> float:
    """Exact Bayes-adaptive value, re-derived from scratch.

    Deliberately independent of `exact_ba.py`: its own history recursion, its
    own Bayes rule (predict through T, multiply the per-channel likelihoods,
    renormalise), its own float64 arithmetic, and carried on the *joint*
    q(x, θ) rather than on a factored pair. Nothing here calls the solver.

    Args:
        joint: (n_states, n_theta) prior over (x, θ).
        T_sta: (n_states_new, n_states_old, n_theta, n_actions). A
            θ-independent T enters as `_flat_theta_T`.
        B_mats: per channel, (n_obs_c, n_states, n_theta).
        reward: (n_states, n_actions, n_theta) immediate reward of (x, a, θ).
        remaining: decisions left.
    """
    if remaining <= 0:
        return 0.0
    best = -np.inf
    for action in range(T_sta.shape[-1]):
        value = float((joint * reward[:, action, :]).sum())
        if remaining > 1:
            predicted = np.einsum("nst,st->nt", T_sta[..., action], joint)
            for obs in itertools.product(*(range(B.shape[0]) for B in B_mats)):
                posterior = predicted.copy()
                for c, o in enumerate(obs):
                    posterior = posterior * B_mats[c][o]
                prob = posterior.sum()
                if prob <= 1e-12:
                    continue
                value += prob * _brute_force_value(
                    posterior / prob, T_sta, B_mats, reward, remaining - 1
                )
        best = max(best, value)
    return float(best)


def _flat_theta_T(T_sa: np.ndarray, n_theta: int) -> np.ndarray:
    """(n_new, n_old, n_actions) → (n_new, n_old, n_theta, n_actions)."""
    n_new, n_old, n_actions = T_sa.shape
    return np.broadcast_to(
        T_sa[:, :, None, :], (n_new, n_old, n_theta, n_actions)
    )


def _theta_channel_kwargs(
    T_sa: np.ndarray, B_probe: np.ndarray, extra_channels: list[np.ndarray]
) -> dict:
    """Solver kwargs for a θ-independent T and θ-dependent probe channel."""
    n_theta = B_probe.shape[-1]
    return dict(
        T=jnp.asarray(T_sa, dtype=jnp.float32),
        B_list=[jnp.asarray(B, dtype=jnp.float32)
                for B in extra_channels + [B_probe]],
        theta_spec=ThetaSpec(component_sizes=(n_theta,)),
        t_factor_spec=FactorSpec(theta_components=()),
        b_factor_specs=tuple(
            [FactorSpec(theta_components=()) for _ in extra_channels]
            + [FactorSpec(theta_components=(0,))]
        ),
    )


def test_theta_posterior_backup_matches_independent_brute_force():
    """The one property that makes this solver Bayes-adaptive: q(θ) carried
    correctly *across* steps. One state, two θ, a noisy θ-dependent probe, so
    every step's observation reshapes q(θ) and the second and third decisions
    can only be scored right by backing up through the updated posterior.

    Checked against the brute force above (own enumeration, own Bayes rule)
    and — so the check has teeth — against the same instance with the probe
    made uninformative, i.e. the value a solver that dropped the posterior
    update would return. The optimum must strictly beat it.
    """
    # BET_0 pays +1 if θ=0, BET_1 pays +1 if θ=1; both are 0 in expectation
    # under the uniform prior, so all the value comes from learning θ.
    T_sa = np.ones((1, 1, 3))
    B_probe = np.array([[[0.8, 0.3]], [[0.2, 0.7]]])  # (n_obs, n_states, n_theta)
    reward = np.zeros((1, 3, 2))
    reward[0, 1, :] = [1.0, -1.0]
    reward[0, 2, :] = [-1.0, 1.0]

    kwargs = _theta_channel_kwargs(T_sa, B_probe, extra_channels=[])
    joint = np.full((1, 2), 0.5)

    for remaining in (2, 3):
        solution = solve_exact(
            jnp.ones(1, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32) / 2.0,
            remaining=remaining,
            reward_table=reward,
            **kwargs,
        )
        expected = _brute_force_value(
            joint, _flat_theta_T(T_sa, 2), [B_probe], reward, remaining
        )
        assert solution.value == pytest.approx(expected, abs=1e-4), (
            f"remaining={remaining}"
        )
        frozen = _brute_force_value(
            joint, _flat_theta_T(T_sa, 2), [np.full_like(B_probe, 0.5)],
            reward, remaining,
        )
        assert solution.value > frozen + 0.1, (
            f"remaining={remaining}: value {solution.value} is no better than "
            f"the no-learning value {frozen}, so the test has no teeth"
        )


def test_action_controlled_information_matches_independent_brute_force():
    """Same brute force, now with the *action* deciding what is learned.

    Two states: the probe is informative only in LAB, and MOVE is the only
    way there. So the optimum has to propagate the hyperstate through T and
    value the information it buys, not just react to observations it gets for
    free. An identity position channel pins x, which is what keeps the
    runners' factored filter exact here (see the module docstring), so the
    joint-belief brute force is the right comparison.
    """
    HOME, LAB = 0, 1
    # actions: MOVE (toggles), BET_0, BET_1 (stay put).
    T_sa = np.zeros((2, 2, 3))
    T_sa[LAB, HOME, 0] = 1.0
    T_sa[HOME, LAB, 0] = 1.0
    for action in (1, 2):
        T_sa[HOME, HOME, action] = 1.0
        T_sa[LAB, LAB, action] = 1.0

    B_position = np.eye(2)                       # (n_obs, n_states), pins x
    B_probe = np.zeros((2, 2, 2))                # (n_obs, n_states, n_theta)
    B_probe[:, HOME, :] = 0.5                    # uninformative at home
    B_probe[0, LAB, :] = [0.9, 0.2]
    B_probe[1, LAB, :] = [0.1, 0.8]

    reward = np.zeros((2, 3, 2))
    reward[:, 1, :] = [1.0, -1.0]
    reward[:, 2, :] = [-1.0, 1.0]

    kwargs = _theta_channel_kwargs(T_sa, B_probe, extra_channels=[B_position])
    joint = np.zeros((2, 2))
    joint[HOME, :] = 0.5

    for remaining in (2, 3):
        solution = solve_exact(
            jnp.zeros(2, dtype=jnp.float32).at[HOME].set(1.0),
            jnp.ones(2, dtype=jnp.float32) / 2.0,
            remaining=remaining,
            reward_table=reward,
            **kwargs,
        )
        expected = _brute_force_value(
            joint, _flat_theta_T(T_sa, 2),
            [B_position[:, :, None] * np.ones(2), B_probe],
            reward, remaining,
        )
        assert solution.value == pytest.approx(expected, abs=1e-4), (
            f"remaining={remaining}"
        )

    # Teeth: with three decisions the optimum must be MOVE-then-bet, which is
    # worth strictly more than any policy that never learns θ (value 0).
    assert solution.value > 0.1
    assert int(np.argmax(solution.action_values)) == 0


def test_theta_dependent_transition_matches_independent_brute_force():
    """The θ-dependent-T path, which is door_code's actual layout
    (`t_factor_spec=FactorSpec(theta_components=(0,))`) and the one place the
    solver re-derives the predictive contraction — `_Model.predict` aligning
    a per-θ predictive onto the full component axes. Every other builder here
    is θ-independent, and `remaining >= 2` is what makes the recursion pass
    through `predict` at all.

    θ decides where MOVE lands; an identity position channel then reads the
    arrival off, so the information is bought *through T* rather than from a
    θ-dependent observation channel. That channel also pins x, which keeps
    the runners' factored filter exact (module docstring), so the joint-belief
    brute force is the right comparison.
    """
    A, B = 0, 1
    # actions: MOVE (0), BET_0 (1), BET_1 (2). Axes (new, old, θ, action).
    T = np.zeros((2, 2, 2, 3))
    T[B, A, 0, 0] = 1.0        # MOVE: θ=0 leaves A ...
    T[A, A, 1, 0] = 1.0        # ... θ=1 does not.
    T[B, B, :, 0] = 1.0
    for action in (1, 2):      # bets never move.
        T[A, A, :, action] = 1.0
        T[B, B, :, action] = 1.0

    reward = np.zeros((2, 3, 2))
    reward[:, 1, :] = [1.0, -1.0]     # BET_0 pays iff θ=0
    reward[:, 2, :] = [-1.0, 1.0]     # BET_1 pays iff θ=1

    kwargs = dict(
        T=jnp.asarray(T, dtype=jnp.float32),
        B_list=[jnp.asarray(np.eye(2), dtype=jnp.float32)],
        theta_spec=ThetaSpec(component_sizes=(2,)),
        t_factor_spec=FactorSpec(theta_components=(0,)),
        b_factor_specs=(FactorSpec(theta_components=()),),
    )
    joint = np.zeros((2, 2))
    joint[A, :] = 0.5
    B_position = np.eye(2)[:, :, None] * np.ones(2)

    for remaining in (2, 3):
        solution = solve_exact(
            jnp.zeros(2, dtype=jnp.float32).at[A].set(1.0),
            jnp.ones(2, dtype=jnp.float32) / 2.0,
            remaining=remaining,
            reward_table=reward,
            **kwargs,
        )
        expected = _brute_force_value(joint, T, [B_position], reward, remaining)
        assert solution.value == pytest.approx(expected, abs=1e-4), (
            f"remaining={remaining}"
        )
        # Teeth: MOVE reveals θ, so the optimum is one MOVE plus an informed
        # bet per remaining decision — strictly above the 0 a solver that
        # ignored the θ-dependence of T would report.
        assert solution.value == pytest.approx(remaining - 1.0, abs=1e-4)
        assert int(np.argmax(solution.action_values)) == 0


def test_multi_component_theta_matches_independent_brute_force():
    """Factored Tiger's θ layout: `component_sizes=(2, 2)` with one channel
    per component. That takes a different `align_to_full_components` branch
    from door_code's single-component (K!,) — each channel's θ axis is placed
    at its own global component axis and broadcast over the other — and every
    builder above is single-component.

    One noisy bit per component and one bet per component, so all the value
    is in the two posteriors. The brute force carries the flat 4-valued θ
    (row-major, θ_0 major) instead of a per-component layout.
    """
    # (n_obs, n_states, 2) per channel: a noisy bit about its own component.
    B_0 = np.array([[[0.85, 0.25]], [[0.15, 0.75]]])
    B_1 = np.array([[[0.7, 0.4]], [[0.3, 0.6]]])
    T_sa = np.ones((1, 1, 3))         # one state, θ-independent
    # actions: WAIT (0), BET_0 (1), BET_1 (2). Flat θ = 2·θ_0 + θ_1.
    reward = np.zeros((1, 3, 4))
    reward[0, 1, :] = [1.0, 1.0, -1.0, -1.0]     # BET_0 pays iff θ_0 = 0
    reward[0, 2, :] = [2.0, -2.0, 2.0, -2.0]     # BET_1 pays iff θ_1 = 0

    kwargs = dict(
        T=jnp.asarray(T_sa, dtype=jnp.float32),
        B_list=[
            jnp.asarray(B_0, dtype=jnp.float32),
            jnp.asarray(B_1, dtype=jnp.float32),
        ],
        theta_spec=ThetaSpec(component_sizes=(2, 2)),
        t_factor_spec=FactorSpec(theta_components=()),
        b_factor_specs=(
            FactorSpec(theta_components=(0,)),
            FactorSpec(theta_components=(1,)),
        ),
    )
    # Same two channels flattened onto the 4-valued θ for the brute force:
    # channel 0 varies with θ_0 = flat // 2, channel 1 with θ_1 = flat % 2.
    B_mats = [np.repeat(B_0, 2, axis=2), np.tile(B_1, (1, 1, 2))]
    joint = np.full((1, 4), 0.25)

    for remaining in (2, 3):
        solution = solve_exact(
            jnp.ones(1, dtype=jnp.float32),
            jnp.full((2, 2), 0.25, dtype=jnp.float32),
            remaining=remaining,
            reward_table=reward,
            **kwargs,
        )
        expected = _brute_force_value(
            joint, _flat_theta_T(T_sa, 4), B_mats, reward, remaining
        )
        assert solution.value == pytest.approx(expected, abs=1e-4), (
            f"remaining={remaining}"
        )
        # Teeth: both bets are worth 0 under the uniform prior, so any value
        # at all comes from carrying the two component posteriors.
        frozen = _brute_force_value(
            joint, _flat_theta_T(T_sa, 4),
            [np.full_like(m, 0.5) for m in B_mats], reward, remaining,
        )
        assert solution.value > frozen + 0.1, (
            f"remaining={remaining}: value {solution.value} is no better than "
            f"the no-learning value {frozen}, so the test has no teeth"
        )


# --- Fully observed instance against plain value iteration ------------------


def test_fully_observed_matches_value_iteration():
    """With an identity observation channel and a singleton θ the problem is
    an MDP, so the expectimax must reproduce backward induction exactly."""
    n_states, n_actions = 3, 2
    rng = np.random.default_rng(7)
    T_sa = rng.random((n_states, n_states, n_actions))
    T_sa /= T_sa.sum(axis=0, keepdims=True)
    reward = rng.normal(size=(n_states, n_actions))

    kwargs = _mdp_kwargs(T_sa)
    reward_table = reward.reshape(n_states, n_actions, 1)

    values = np.zeros(n_states)
    for horizon in range(1, 5):
        q_values = reward + np.einsum("nsa,n->sa", T_sa, values)
        values = q_values.max(axis=1)
        for state in range(n_states):
            q_state = jnp.zeros(n_states, dtype=jnp.float32).at[state].set(1.0)
            solution = solve_exact(
                q_state,
                jnp.ones(1, dtype=jnp.float32),
                remaining=horizon,
                reward_table=reward_table,
                **kwargs,
            )
            assert solution.value == pytest.approx(values[state], abs=1e-4), (
                f"horizon={horizon} state={state}"
            )
            np.testing.assert_allclose(
                solution.action_values, q_values[state], atol=1e-4,
            )


# --- Memoization is value-preserving ---------------------------------------


def test_memoized_matches_uncached_recursion():
    """The belief-keyed memo must only collapse nodes, never change values."""
    _env, kwargs, q_state, q_static = _door_code_setup(K=3, lock_length=1)

    cached = solve_exact(q_state, q_static, remaining=3, memoize=True, **kwargs)
    uncached = solve_exact(q_state, q_static, remaining=3, memoize=False, **kwargs)

    assert cached.value == pytest.approx(uncached.value, abs=1e-9)
    np.testing.assert_allclose(
        cached.action_values, uncached.action_values, atol=1e-9,
    )
    # The collapse is the whole point: order-permuted probe histories share a
    # posterior, so the cached recursion must expand strictly fewer nodes.
    assert cached.n_nodes < uncached.n_nodes


# --- Cost guard -------------------------------------------------------------


def test_node_cap_raises_with_the_count():
    _env, kwargs, q_state, q_static = _door_code_setup(K=3, lock_length=1)

    with pytest.raises(NodeCapExceeded) as excinfo:
        solve_exact(q_state, q_static, remaining=4, max_nodes=5, **kwargs)

    message = str(excinfo.value)
    assert "expanded 6 nodes" in message
    assert "max_nodes=5" in message


# --- Planner protocol / runner integration ---------------------------------


def _plan_kwargs(env, kwargs) -> dict:
    return dict(
        goal=env.get_goal(),
        theta_spec=kwargs["theta_spec"],
        t_factor_spec=kwargs["t_factor_spec"],
        b_factor_specs=kwargs["b_factor_specs"],
    )


def test_planner_returns_onehot_at_an_optimal_action():
    env, kwargs, q_state, q_static = _door_code_setup(K=3, lock_length=2)
    planner = ExactBAPlanner(
        max_steps=3,
        reward_table=kwargs["reward_table"],
        terminal_states=kwargs["terminal_states"],
    )

    dist = planner.plan(
        q_state, q_static, kwargs["T"], kwargs["B_list"],
        horizon=3, **_plan_kwargs(env, kwargs),
    )

    dist = np.asarray(dist)
    assert dist.shape == (env.n_actions,)
    assert dist.sum() == pytest.approx(1.0)
    values = planner.last_action_values
    assert values[int(dist.argmax())] == pytest.approx(values.max())
    assert planner.max_nodes_expanded > 0


def test_tie_break_picks_different_members_of_one_optimal_set():
    """Ties are structural on door_code — every probe station is exactly
    optimal from the start belief — so the two tie-break rules must act
    differently while agreeing on every action *value*. That difference is
    what the two exact arms in scripts/run_exact_reference.py bracket: both
    are exactly optimal per episode, but they learn different bits of θ
    first and so post different block returns.
    """
    env, kwargs, q_state, q_static = _door_code_setup(K=3, lock_length=2)
    planners = {
        rule: ExactBAPlanner(
            max_steps=3,
            reward_table=kwargs["reward_table"],
            terminal_states=kwargs["terminal_states"],
            tie_break=rule,
        )
        for rule in ("lowest", "highest")
    }
    actions = {}
    for rule, planner in planners.items():
        dist = planner.plan(
            q_state, q_static, kwargs["T"], kwargs["B_list"],
            horizon=3, **_plan_kwargs(env, kwargs),
        )
        actions[rule] = int(np.asarray(dist).argmax())

    values = planners["lowest"].last_action_values
    np.testing.assert_array_equal(values, planners["highest"].last_action_values)
    optimal = optimal_action_set(values)
    assert len(optimal) > 1, "no ties at this belief — the test has no teeth"
    assert actions["lowest"] == int(optimal[0])
    assert actions["highest"] == int(optimal[-1])
    assert actions["lowest"] != actions["highest"]


def test_optimal_set_shrinks_monotonically_as_the_tolerance_tightens():
    """A tighter tie tolerance must yield a *subset*, never a different set.

    What makes the tolerance sweep in scripts/run_exact_reference.py readable:
    its `mean_optimal_set_size` is only a discriminator for "did tightening
    drop genuine ties?" if tightening can only ever remove members. The
    argmax itself must survive every tolerance, so the set is never empty.
    """
    values = np.array([1.0, 1.0 - 1e-9, 1.0 - 1e-7, 1.0 - 1e-5, 0.5])
    sets = [
        set(optimal_action_set(values, rel_tol=tol).tolist())
        for tol in (1e-4, 1e-6, 1e-8, 1e-10)
    ]
    for looser, tighter in zip(sets, sets[1:]):
        assert tighter <= looser
        assert 0 in tighter
    assert sets[0] > sets[-1], "no member was ever dropped — no teeth"


def test_unknown_tie_break_rejected():
    with pytest.raises(ValueError, match="tie_break"):
        ExactBAPlanner(max_steps=3, tie_break="random")


def test_horizon_must_be_the_remaining_budget():
    """The contract that would otherwise fail silently: driven with a
    canonical horizon shorter than max_steps, the runner hands over a
    constant lookahead and this becomes a receding-horizon arm still
    labelled exact. Enforced per decision, so both failure modes raise."""
    env, kwargs, q_state, q_static = _door_code_setup(K=3, lock_length=1)
    planner = ExactBAPlanner(
        max_steps=4,
        reward_table=kwargs["reward_table"],
        terminal_states=kwargs["terminal_states"],
    )
    args = (q_state, q_static, kwargs["T"], kwargs["B_list"])

    # A clipped canonical horizon below max_steps, on the first decision.
    with pytest.raises(ValueError, match="receding-horizon"):
        planner.plan(*args, horizon=3, **_plan_kwargs(env, kwargs))

    # And clip_horizon=False, which repeats the full horizon instead of
    # decrementing it: correct on decision 0, wrong on decision 1.
    planner.plan(*args, horizon=4, **_plan_kwargs(env, kwargs))
    with pytest.raises(ValueError, match="remaining budget 3"):
        planner.plan(*args, horizon=4, **_plan_kwargs(env, kwargs))


def test_reward_table_is_required():
    env, kwargs, q_state, q_static = _door_code_setup(K=3, lock_length=1)
    planner = ExactBAPlanner(max_steps=2)

    with pytest.raises(ValueError, match="env reward table"):
        planner.plan(
            q_state, q_static, kwargs["T"], kwargs["B_list"],
            horizon=2, **_plan_kwargs(env, kwargs),
        )


def test_drives_the_factored_tiger_runner():
    """The Factored Tiger runner supplies the reward table through plan()
    and filters kwargs by the planner's declared signature, so plan() must
    declare `reward_table` to receive one at all."""
    from learningpomdp.experiment.factored_tiger_runner import (
        run_outer_loop as run_tiger,
    )

    planner = ExactBAPlanner(max_steps=4)
    result = run_tiger(
        env_kwargs=dict(K=2, listen_accuracy=0.85, max_steps=4),
        planner=planner,
        n_seeds=2,
        n_episodes=2,
        horizon=4,
        max_steps=4,
        deterministic=True,
        seed=0,
        clip_horizon=True,
        show_progress=False,
        workers=1,
    )

    assert result.cumulative_reward.shape == (2, 2)
    assert planner.n_plan_calls > 0
    # Never opens a tiger door: at a uniform θ_k the exact value of OPEN is
    # 0.5 * 10 - 0.5 * 100 < 0, so the optimum listens until confident.
    assert result.n_tiger_opens.sum() == 0


def test_within_episode_decisions_are_memo_hits():
    """After the first decision of an episode the whole reachable tree is
    already expanded, so every later decision must add zero nodes. This only
    holds if the runner's belief is bit-for-bit a node of that tree — it is
    the shared-filter protocol, asserted.

    Caveat this test cannot cover: with `learn_on_failure=False` the runner
    *skips* the Bayesian update on the FAILURE transition while the solver's
    tree applied it, so on that branch the runner's belief genuinely leaves
    the tree. It is invisible here (and harmless) only because FAILURE
    terminates the episode on the same step, so no later `plan()` call in
    that episode can miss the memo, and `prepare_episode` drops the memo
    before the next one. A future env where the skipped update is followed by
    further decisions would break the invariant for real, and this assertion
    would not catch it.
    """
    max_steps = 4
    env_kwargs = dict(K=3, lock_length=2, probe_noise=0.2)
    env = DoorCodeEnv(**env_kwargs)
    per_episode: list[list[int]] = []

    class _Tracked(ExactBAPlanner):
        def prepare_episode(self, q_static, rng):
            per_episode.append([])
            super().prepare_episode(q_static, rng)

        def plan(self, *args, **kwargs):
            out = super().plan(*args, **kwargs)
            per_episode[-1].append(self._solver.n_nodes)
            return out

    planner = _Tracked(
        max_steps=max_steps,
        reward_table=_door_code_reward_table(env),
        terminal_states=(env.success_state_index, env.failure_state_index),
    )
    run_outer_loop(
        env_kwargs=env_kwargs,
        planner=planner,
        n_seeds=1,
        n_episodes=2,
        horizon=max_steps,
        max_steps=max_steps,
        deterministic=True,
        seed=0,
        learn_on_failure=False,
        clip_horizon=True,
        show_progress=False,
    )

    assert len(per_episode) == 2
    for counts in per_episode:
        assert len(counts) >= 2, f"episode too short to test: {counts}"
        # The running total is monotone, so any growth after the first
        # decision is a memo miss.
        assert set(counts) == {counts[0]}, f"memo missed mid-episode: {counts}"


def test_drives_the_door_code_runner():
    """The reference must be a drop-in for the shared-filter runner: with
    clip_horizon and horizon == max_steps, `horizon` is the remaining
    budget, which is what the solver consumes."""
    max_steps = 4
    env_kwargs = dict(K=3, lock_length=2, probe_noise=0.2)
    env = DoorCodeEnv(**env_kwargs)
    planner = ExactBAPlanner(
        max_steps=max_steps,
        reward_table=_door_code_reward_table(env),
        terminal_states=(env.success_state_index, env.failure_state_index),
    )

    result = run_outer_loop(
        env_kwargs=env_kwargs,
        planner=planner,
        n_seeds=2,
        n_episodes=2,
        horizon=max_steps,
        max_steps=max_steps,
        deterministic=True,
        seed=0,
        learn_on_failure=False,
        clip_horizon=True,
        show_progress=False,
    )

    assert result.cumulative_reward.shape == (2, 2)
    assert planner.n_plan_calls > 0
    assert planner.max_nodes_expanded > 0
