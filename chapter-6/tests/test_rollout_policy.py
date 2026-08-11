"""Tests for the configurable MCTS leaf-rollout policy.

The shipped rollout draws actions uniformly. Guez et al. 2012's BAMCP
supplies an epsilon-greedy rollout against a Q-function learned
off-policy from the agent's own experience, introduced for exactly the
case where reward requires a carefully selected action sequence. These
tests pin the three properties the ablation rests on:

- ``rollout_epsilon=1.0`` is statistically indistinguishable from
  ``"uniform"`` (the eps-greedy branch really does reduce to uniform in
  the limit, so any measured difference at eps<1 is the policy and not an
  incidental RNG or bookkeeping change);
- ``rollout_epsilon=0.0`` acts greedily in Q-hat;
- under a *flat* Q-hat the greedy branch is uniform, not pinned to action
  0. This is the property a lowest-index argmax silently violated: with
  Q-hat all-equal every greedy step took action 0, which on door_code is
  the zero-reward VISIT_PROBE_0 whose Q-hat entry therefore never moves,
  so the "competent" rollout was strictly worse than the uniform one it
  was meant to improve on;
- Q-hat is initialised optimistically at each path's own reward scale
  (the ε-greedy *rule* is scale-invariant, its initialisation is not);
- Q-hat persists across ``plan()`` calls within a block and is reset by
  ``prepare_block()`` — the scope that stops it leaking one block's
  system identity into the next.

Both the NumPy path (door_code / minigrid) and the Numba path
(rock_sample / factored_tiger) are covered, since @njit code cannot close
over Python state and needed a separately threaded buffer.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.environments.factored_tiger import FactoredTigerEnv
from learningpomdp.experiment.door_code_runner import run_block
from learningpomdp.experiment.factored_tiger_runner import (
    _build_factored_tiger_reward_table,
    run_block as run_tiger_block,
)
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.observations import (
    ModalityRegistry,
    ObservationModality,
    ThetaDependence,
)
from learningpomdp.planning._bamcp_numba import _rollout_numba
from learningpomdp.planning._boss_numba import _rollout_numba_boss
from learningpomdp.planning.bamcp_planner import BAMCPPlanner
from learningpomdp.planning.boss_planner import BOSSPlanner


N_STATES = 6
N_ACTIONS = 4


def _flat_env(n_static: int = 2):
    """Reward-free, terminal-free chain: isolates the action-selection
    rule from any reward-driven feedback."""
    T = np.zeros((N_STATES, N_STATES, n_static, N_ACTIONS), dtype=np.float64)
    for theta in range(n_static):
        for action in range(N_ACTIONS):
            for s in range(N_STATES):
                T[(s + 1 + action) % N_STATES, s, theta, action] = 1.0
    B_position = np.eye(N_STATES, dtype=np.float64)
    B_probe = np.zeros((3, N_STATES, n_static), dtype=np.float64)
    B_probe[2, :, :] = 1.0
    B_outcome = np.zeros((3, N_STATES), dtype=np.float64)
    B_outcome[0, :] = 1.0
    return T, [B_position, B_probe, B_outcome]


def _jax_args(T, B_list, n_static):
    return (
        jnp.zeros(N_STATES, dtype=jnp.float32).at[0].set(1.0),
        jnp.ones(n_static, dtype=jnp.float32) / n_static,
        jnp.asarray(T, dtype=jnp.float32),
        [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
    )


# --- Validation --------------------------------------------------------------


@pytest.mark.parametrize("cls", [BAMCPPlanner, BOSSPlanner])
def test_unknown_rollout_policy_rejected(cls) -> None:
    with pytest.raises(ValueError, match="rollout_policy"):
        cls(rollout_policy="boltzmann")


def test_default_is_uniform_and_allocates_no_q_hat() -> None:
    """The canonical path must not even allocate Q-hat, which is the
    structural guarantee that no Q-hat branch can run by default."""
    T, B_list = _flat_env()
    planner = BAMCPPlanner(n_simulations=32, depth=4, rollout_seed=0)
    assert planner.rollout_policy == "uniform"
    q_state, q_static, T_j, B_j = _jax_args(T, B_list, 2)
    planner.plan(q_state, q_static, T_j, B_j, goal=None, horizon=4)
    assert planner._q_rollout is None


# --- eps = 1.0 reduces to uniform -------------------------------------------


def _mean_dist(rollout_policy: str, epsilon: float, seeds) -> np.ndarray:
    T, B_list = _flat_env()
    q_state, q_static, T_j, B_j = _jax_args(T, B_list, 2)
    dists = []
    for s in seeds:
        planner = BAMCPPlanner(
            n_simulations=256, depth=5, rollout_seed=s,
            rollout_policy=rollout_policy, rollout_epsilon=epsilon,
        )
        dists.append(np.asarray(
            planner.plan(q_state, q_static, T_j, B_j, goal=None, horizon=5),
            dtype=np.float64,
        ))
    return np.stack(dists)


def test_eps_one_is_statistically_indistinguishable_from_uniform() -> None:
    """At eps=1.0 every rollout action is drawn uniformly, so the two
    policies describe the same stochastic process.

    They are not bit-identical — the eps branch consumes one extra RNG
    draw per rollout step, which shifts the stream — so the comparison is
    over seed-means, at 3 SE of the difference.
    """
    seeds = range(12)
    uni = _mean_dist("uniform", 0.1, seeds)
    eps1 = _mean_dist("eps_greedy_q", 1.0, seeds)
    n = len(list(seeds))
    se = np.sqrt(uni.var(axis=0, ddof=1) / n + eps1.var(axis=0, ddof=1) / n)
    tol = np.maximum(3.0 * se, 0.05)
    assert np.all(np.abs(uni.mean(axis=0) - eps1.mean(axis=0)) <= tol), (
        f"eps=1.0 differs from uniform beyond Monte-Carlo error:\n"
        f"  uniform={uni.mean(axis=0)}\n  eps=1.0={eps1.mean(axis=0)}\n"
        f"  tol={tol}"
    )


# --- eps = 0.0 is greedy in Q-hat -------------------------------------------


def test_eps_zero_rollout_is_greedy_in_q_hat(monkeypatch) -> None:
    """With eps=0 and a Q-hat that strongly prefers one action, every
    rollout step must take that action."""
    T, B_list = _flat_env(n_static=1)
    preferred = 2
    planner = BAMCPPlanner(
        n_simulations=1, depth=6, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.0,
    )
    # Seed Q-hat by hand. The magnitude is large enough that the
    # off-policy update inside the rollout (which nudges the chosen entry
    # toward gamma * max Q of the successor, i.e. slightly down under zero
    # reward) cannot flip the argmax within the rollout.
    planner._q_rollout = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    planner._q_rollout[:, preferred] = 10.0

    chosen = []
    original_step = BAMCPPlanner._step

    def recording_step(state, theta, action, *args, **kwargs):
        chosen.append(action)
        return original_step(state, theta, action, *args, **kwargs)

    monkeypatch.setattr(BAMCPPlanner, "_step", staticmethod(recording_step))

    planner._rollout(
        state=0, theta=0, depth=6,
        T=T[:, :, :, :], B_position=B_list[0], B_probe=B_list[1],
        B_outcome=B_list[2], probe_state_set=set(), n_actions=N_ACTIONS,
    )
    assert chosen == [preferred] * 6, (
        f"eps=0 rollout was not greedy in Q-hat: chose {chosen}, "
        f"expected all {preferred}"
    )


def test_eps_zero_boss_rollout_is_greedy_in_q_hat(monkeypatch) -> None:
    """Same for BOSS's rollout — its real action must follow Q-hat while
    the k_idx draw (the merged-action-space mechanism) stays uniform."""
    T, B_list = _flat_env(n_static=2)
    preferred = 1
    planner = BOSSPlanner(
        n_samples=2, n_simulations=1, depth=5, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.0,
    )
    planner._q_rollout = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    planner._q_rollout[:, preferred] = 10.0

    chosen = []
    original_step = BAMCPPlanner._step

    def recording_step(state, theta, action, *args, **kwargs):
        chosen.append(action)
        return original_step(state, theta, action, *args, **kwargs)

    monkeypatch.setattr(BAMCPPlanner, "_step", staticmethod(recording_step))

    planner._rollout(
        state=0, depth=5,
        T=T, B_position=B_list[0], B_probe=B_list[1], B_outcome=B_list[2],
        probe_state_set=set(), n_actions=N_ACTIONS, K=2, thetas=[0, 1],
    )
    assert chosen == [preferred] * 5, (
        f"BOSS eps=0 rollout was not greedy in Q-hat: chose {chosen}"
    )


# --- flat Q-hat must not collapse the greedy branch --------------------------


def test_greedy_rollout_is_uniform_under_a_flat_q_hat(monkeypatch) -> None:
    """The regression this pins: `np.argmax` returns the LOWEST index, so
    an all-equal Q-hat made every greedy step take action 0 — a
    degenerate rollout, and on door_code a strictly worse one than
    uniform. `_flat_env` is reward-free, so a zero Q-hat has zero TD error
    and stays exactly flat for the whole rollout, which isolates the
    tie-break from the learning rule.
    """
    T, B_list = _flat_env(n_static=1)
    planner = BAMCPPlanner(
        n_simulations=1, depth=4, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.0,
    )
    planner._q_rollout = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    chosen: list[int] = []
    original_step = BAMCPPlanner._step

    def recording_step(state, theta, action, *args, **kwargs):
        chosen.append(action)
        return original_step(state, theta, action, *args, **kwargs)

    monkeypatch.setattr(BAMCPPlanner, "_step", staticmethod(recording_step))
    planner._rollout(
        state=0, theta=0, depth=4000,
        T=T, B_position=B_list[0], B_probe=B_list[1], B_outcome=B_list[2],
        probe_state_set=set(), n_actions=N_ACTIONS,
    )

    assert np.all(planner._q_rollout == 0.0), (
        "Q-hat moved, so this no longer tests the flat-Q-hat tie-break"
    )
    freqs = np.bincount(chosen, minlength=N_ACTIONS) / len(chosen)
    # SE at n=4000 over 4 actions is ~0.007, so 0.05 is a ~7-sigma band.
    assert np.all(np.abs(freqs - 1.0 / N_ACTIONS) < 0.05), (
        f"greedy rollout under a flat Q-hat is not uniform: {freqs}"
    )


def test_greedy_rollout_does_not_pin_to_probe_zero_on_door_code(
    monkeypatch,
) -> None:
    """The same failure measured on the env the ablation actually runs on,
    through the real optimistic init rather than a hand-seeded Q-hat.

    door_code's action 0 is VISIT_PROBE_0: reward 0 and non-terminal, so
    with a lowest-index tie-break and zero init its Q-hat entry stays the
    argmax for ever while every failed OPEN_DOOR goes negative. Measured
    on this configuration, action 0 took 60% of all greedy rollout steps
    before the fix and 6% after (uniform would be 1/9 = 11%).
    """
    env = DoorCodeEnv(K=4, probe_noise=0.2, lock_length=2)
    T = np.asarray(env.get_transition_tensor(), dtype=np.float64)
    B_position, B_probe, B_outcome = (
        np.asarray(B, dtype=np.float64) for B in env.get_observation_tensors()
    )
    probe_state_set = {
        s for s in range(T.shape[0]) if not np.allclose(B_probe[2, s, :], 1.0)
    }
    planner = BAMCPPlanner(
        n_simulations=1, depth=8, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.1, rollout_q_lr=0.1,
    )
    planner._ensure_q_rollout(env.n_states, env.n_actions, 1.0)

    chosen: list[int] = []
    original_step = BAMCPPlanner._step

    def recording_step(state, theta, action, *args, **kwargs):
        chosen.append(action)
        return original_step(state, theta, action, *args, **kwargs)

    monkeypatch.setattr(BAMCPPlanner, "_step", staticmethod(recording_step))
    for _ in range(200):
        planner._rollout(
            state=env._START, theta=0, depth=8,
            T=T, B_position=B_position, B_probe=B_probe,
            B_outcome=B_outcome, probe_state_set=probe_state_set,
            n_actions=env.n_actions,
        )

    share_action_0 = chosen.count(0) / len(chosen)
    assert share_action_0 < 0.30, (
        f"eps-greedy rollout is pinned to VISIT_PROBE_0 "
        f"({share_action_0:.2%} of steps) — the tie-break has regressed"
    )


def test_optimistic_init_uses_the_path_s_reward_scale(monkeypatch) -> None:
    """ε-greedy is scale-invariant; its Q-hat initialisation is not.

    The legacy path's reward is {+1, -1, 0} from the outcome channel, so
    +1 is its optimistic bound. The Numba path knows the env's true scale
    and must read it from ``reward_table`` — hard-coding the legacy +1
    there would be pessimistic on factored_tiger (+10 / -100) and
    optimistic-by-orders-of-magnitude nowhere useful.
    """
    recorded: list[float] = []
    original = BAMCPPlanner._ensure_q_rollout

    def recording(self, n_states, n_actions, q_init):
        recorded.append(q_init)
        return original(self, n_states, n_actions, q_init)

    monkeypatch.setattr(BAMCPPlanner, "_ensure_q_rollout", recording)

    T, B_list = _flat_env()
    q_state, q_static, T_j, B_j = _jax_args(T, B_list, 2)
    BAMCPPlanner(
        n_simulations=8, depth=3, rollout_seed=0,
        rollout_policy="eps_greedy_q",
    ).plan(q_state, q_static, T_j, B_j, goal=None, horizon=3)
    assert recorded == [1.0]

    recorded.clear()
    _plan_numba(BAMCPPlanner(
        n_simulations=8, depth=3, rollout_seed=0,
        rollout_policy="eps_greedy_q",
    ), reward_scale=7.0)
    assert recorded == [7.0]


# --- Q-hat lifetime ---------------------------------------------------------


def test_q_hat_persists_across_plan_and_resets_on_prepare_block() -> None:
    """Q-hat must survive plan() boundaries (it is learned across a block)
    and be dropped by prepare_block() (so it cannot carry one block's
    system identity into the next)."""
    T, B_list = _flat_env()
    q_state, q_static, T_j, B_j = _jax_args(T, B_list, 2)
    planner = BAMCPPlanner(
        n_simulations=64, depth=5, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.5,
    )

    planner.plan(q_state, q_static, T_j, B_j, goal=None, horizon=5)
    assert planner._q_rollout is not None
    assert planner._q_rollout.shape == (N_STATES, N_ACTIONS)
    first_buffer = planner._q_rollout

    # Second plan() reuses the SAME buffer (persistence, not reallocation).
    planner.plan(q_state, q_static, T_j, B_j, goal=None, horizon=5)
    assert planner._q_rollout is first_buffer

    # Block boundary drops it entirely.
    planner.prepare_block(q_static, np.random.default_rng(0))
    assert planner._q_rollout is None

    # And the next plan() starts from a fresh buffer, not the old object.
    planner.plan(q_state, q_static, T_j, B_j, goal=None, horizon=5)
    assert planner._q_rollout is not None
    assert planner._q_rollout is not first_buffer


def test_prepare_episode_does_not_reset_q_hat() -> None:
    """The scope distinction is the whole point of the new hook: episodes
    within a block share theta_true, so Q-hat must cross an episode
    boundary."""
    T, B_list = _flat_env()
    q_state, q_static, T_j, B_j = _jax_args(T, B_list, 2)
    planner = BAMCPPlanner(
        n_simulations=64, depth=5, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.5,
    )
    planner.plan(q_state, q_static, T_j, B_j, goal=None, horizon=5)
    buffer = planner._q_rollout
    planner.prepare_episode(q_static, np.random.default_rng(0))
    assert planner._q_rollout is buffer


def test_runner_calls_prepare_block_once_per_block() -> None:
    """run_block must invoke the hook, or Q-hat would silently persist
    across theta_true draws in the real experiment."""
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    planner = BAMCPPlanner(
        n_simulations=16, depth=4, rollout_seed=0,
        rollout_policy="eps_greedy_q",
    )
    calls = []
    original = planner.prepare_block

    def counting_prepare_block(q_static, rng):
        calls.append(1)
        return original(q_static, rng)

    planner.prepare_block = counting_prepare_block  # type: ignore[method-assign]
    run_block(
        env, planner, theta_true=0, n_episodes=3,
        horizon=4, max_steps=4, deterministic=True, seed=0,
    )
    assert len(calls) == 1, (
        f"prepare_block should fire once per block, fired {len(calls)}x"
    )


# --- Numba path -------------------------------------------------------------


def _numba_side_inputs(n_static: int, reward_scale: float = 1.0):
    registry = ModalityRegistry(modalities=(
        ObservationModality(
            name="position", cardinality=N_STATES,
            theta_dependence=ThetaDependence.INDEPENDENT,
        ),
        ObservationModality(
            name="probe", cardinality=3,
            theta_dependence=ThetaDependence.INDEPENDENT,
        ),
        ObservationModality(
            name="outcome", cardinality=3,
            theta_dependence=ThetaDependence.INDEPENDENT,
        ),
    ))
    specs = (
        FactorSpec(theta_components=()),
        FactorSpec(theta_components=()),
        FactorSpec(theta_components=()),
    )
    reward_table = np.zeros((N_STATES, N_ACTIONS, n_static), dtype=np.float64)
    # One rewarding (state, action), so Q-hat actually moves off its
    # optimistic init. Its magnitude IS the reward scale the Numba path
    # must derive the init from.
    reward_table[1, 3, :] = reward_scale
    return registry, ThetaSpec(component_sizes=(n_static,)), specs, reward_table


def _plan_numba(
    planner: BAMCPPlanner, n_static: int = 2, reward_scale: float = 1.0,
):
    T, B_list = _flat_env(n_static)
    T_3d = np.ascontiguousarray(T[:, :, 0, :])
    B_list = [B_list[0], np.ascontiguousarray(B_list[1][:, :, 0]), B_list[2]]
    registry, theta_spec, specs, reward_table = _numba_side_inputs(
        n_static, reward_scale,
    )
    return planner.plan(
        jnp.zeros(N_STATES, dtype=jnp.float32).at[0].set(1.0),
        jnp.ones(n_static, dtype=jnp.float32) / n_static,
        jnp.asarray(T_3d, dtype=jnp.float32),
        [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
        goal=None, horizon=5,
        theta_spec=theta_spec,
        t_factor_spec=FactorSpec(theta_components=()),
        b_factor_specs=specs,
        registry=registry,
        reward_fn=lambda *a, **k: 0.0,
        terminal_fn=None,
        reward_table=reward_table,
    )


def test_numba_path_uniform_allocates_no_q_hat() -> None:
    """@njit gets a (1, 1) dummy under uniform; the planner-side buffer
    must stay None so the canonical Numba path is untouched."""
    planner = BAMCPPlanner(n_simulations=64, depth=5, rollout_seed=0)
    _plan_numba(planner)
    assert planner._q_rollout is None


def test_numba_path_eps_greedy_learns_and_persists() -> None:
    """The threaded buffer must actually be written by the @njit rollout
    (a silently-unwritten buffer would make the whole Numba arm of the
    ablation a no-op) and must survive across plan() calls."""
    planner = BAMCPPlanner(
        n_simulations=256, depth=5, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.3, rollout_q_lr=0.5,
    )
    _plan_numba(planner)
    assert planner._q_rollout is not None
    # The buffer starts at the optimistic init (reward_table.max() == 1.0
    # here), so "written" means "moved off the init", not "off zero".
    assert np.any(planner._q_rollout != 1.0), (
        "Numba rollout never wrote Q-hat — the buffer is not being "
        "threaded into _rollout_numba"
    )
    buffer = planner._q_rollout
    before = buffer.copy()
    _plan_numba(planner)
    assert planner._q_rollout is buffer, "Q-hat was reallocated across plan()"
    assert not np.array_equal(before, buffer), (
        "Q-hat did not keep learning on the second plan() call"
    )
    planner.prepare_block(
        jnp.ones(2, dtype=jnp.float32) / 2, np.random.default_rng(0),
    )
    assert planner._q_rollout is None


def test_numba_path_eps_greedy_changes_the_action_distribution() -> None:
    """End-to-end: a competent rollout must be able to move the root
    action distribution. If it cannot, the ablation cannot answer the
    question it exists to answer."""
    uniform = BAMCPPlanner(n_simulations=256, depth=5, rollout_seed=0)
    greedy = BAMCPPlanner(
        n_simulations=256, depth=5, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.05, rollout_q_lr=0.5,
    )
    dist_uniform = np.asarray(_plan_numba(uniform), dtype=np.float64)
    # Warm Q-hat over a few plan() calls, as happens within a block.
    for _ in range(3):
        dist_greedy = np.asarray(_plan_numba(greedy), dtype=np.float64)
    assert not np.allclose(dist_uniform, dist_greedy, atol=1e-3), (
        f"eps_greedy_q rollout had no effect on the root distribution: "
        f"uniform={dist_uniform}, greedy={dist_greedy}"
    )


# --- Numba path: flat Q-hat must not collapse the greedy branch --------------
#
# The @njit rollouts cannot be monkeypatched to record their actions, so
# the action is read out of the return value instead: reward is set to the
# action index, depth is 1 and gamma is 1, so the rollout's total return
# IS the action it chose. rollout_q_lr=0 freezes Q-hat flat, which is the
# case the lowest-index argmax collapsed. Unseeded — Numba keeps its own
# global RNG state — so the tolerance is a ~7-sigma band, not a tight one.


def _numba_action_env(n_actions: int = N_ACTIONS):
    """(T_eff, B_pos, B_lik_all, lik_obs_card, reward_table) where the
    reward of every (state, action) is the action index."""
    T, _ = _flat_env(n_static=1)
    T_eff = np.ascontiguousarray(T[:, :, 0, :])
    B_pos = np.eye(N_STATES, dtype=np.float64)
    B_lik_all = np.zeros((1, 1, 1, N_STATES), dtype=np.float64)
    lik_obs_card = np.zeros(1, dtype=np.int32)
    reward_table = np.zeros((N_STATES, n_actions, 1), dtype=np.float64)
    for a in range(n_actions):
        reward_table[:, a, 0] = float(a)
    return T_eff, B_pos, B_lik_all, lik_obs_card, reward_table


def _assert_uniform_over_actions(actions: np.ndarray, label: str) -> None:
    freqs = np.bincount(actions, minlength=N_ACTIONS) / actions.size
    assert np.all(np.abs(freqs - 1.0 / N_ACTIONS) < 0.05), (
        f"{label}: greedy rollout under a flat Q-hat is not uniform: {freqs}"
    )


def test_numba_greedy_rollout_is_uniform_under_a_flat_q_hat() -> None:
    T_eff, B_pos, B_lik_all, lik_obs_card, reward_table = _numba_action_env()
    terminal_table = np.zeros((N_STATES, N_ACTIONS, 1), dtype=np.bool_)
    q_rollout = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    actions = np.array([
        _rollout_numba(
            0, 0, 1,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, False,
            1.0, N_ACTIONS, 0,
            q_rollout, True, 0.0, 0.0,
        )
        for _ in range(4000)
    ], dtype=np.int64)
    assert np.all(q_rollout == 0.0), "rollout_q_lr=0 must freeze Q-hat"
    _assert_uniform_over_actions(actions, "BAMCP numba")


def test_numba_boss_greedy_rollout_is_uniform_under_a_flat_q_hat() -> None:
    T_eff, B_pos, B_lik_all, lik_obs_card, reward_table = _numba_action_env()
    terminal_table = np.zeros((N_STATES, N_ACTIONS, 1), dtype=np.bool_)
    q_rollout = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    thetas = np.zeros(1, dtype=np.int32)
    actions = np.array([
        _rollout_numba_boss(
            0, 1,
            thetas, 1,
            T_eff, B_pos, B_lik_all, lik_obs_card,
            reward_table, terminal_table, False,
            1.0, N_ACTIONS, 0,
            q_rollout, True, 0.0, 0.0,
        )
        for _ in range(4000)
    ], dtype=np.int64)
    assert np.all(q_rollout == 0.0), "rollout_q_lr=0 must freeze Q-hat"
    _assert_uniform_over_actions(actions, "BOSS numba")


def test_runner_reaches_the_numba_eps_greedy_path() -> None:
    """The Numba half of the rollout work must be reachable from a runner,
    not only from unit tests.

    ``factored_tiger``'s runner passes ``reward_fn``, so BAMCP dispatches
    to ``_plan_numba``; the recorded optimistic init proves which branch
    ran, since only the Numba branch reads the scale from
    ``reward_table`` (+10 here) rather than assuming the legacy +1.
    """
    env = FactoredTigerEnv(K=2, listen_accuracy=1.0, max_steps=4)
    expected_init = float(_build_factored_tiger_reward_table(env).max())
    planner = BAMCPPlanner(
        n_simulations=32, depth=3, rollout_seed=0,
        rollout_policy="eps_greedy_q", rollout_epsilon=0.3, rollout_q_lr=0.5,
    )
    recorded: list[float] = []
    original = BAMCPPlanner._ensure_q_rollout

    def recording(self, n_states, n_actions, q_init):
        recorded.append(q_init)
        return original(self, n_states, n_actions, q_init)

    planner._ensure_q_rollout = recording.__get__(planner, BAMCPPlanner)
    block = run_tiger_block(
        env, planner, block_theta_true=3,  # (1, 1) — both doors safe
        n_episodes=2, horizon=3, max_steps=4, deterministic=True, seed=0,
    )
    assert len(block.episodes) == 2
    assert recorded and all(v == expected_init for v in recorded), (
        f"optimistic init {recorded} != reward_table.max() "
        f"{expected_init} — "
        f"the legacy path ran, or the scale is hard-coded"
    )
    assert planner._q_rollout is not None
    assert planner._q_rollout.shape == (env.n_states, env.n_actions)
    assert np.any(planner._q_rollout != expected_init), (
        "Q-hat never moved off its init across a whole block of episodes"
    )
