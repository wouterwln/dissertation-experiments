"""Tests for BAMCPPlanner (and its literal-POMCP ablation flag)."""

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_block, run_outer_loop
from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.observations import (
    ModalityRegistry,
    ObservationModality,
    ThetaDependence,
)
from learningpomdp.planning.bamcp_planner import BAMCPPlanner


def test_plan_with_sharp_theta_picks_correct_door_K3() -> None:
    """When q(theta) is sharp, BAMCP's tree search should converge to
    OPEN_DOOR_pi(0) given enough simulations."""
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1)
    bamcp = BAMCPPlanner(n_simulations=512, depth=4, rollout_seed=0)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask

    # Test for two different sharp thetas.
    for theta_idx in [0, 4]:
        perm = env.theta_permutation if env.theta == theta_idx else None
        env.reset(theta=theta_idx)
        correct = env.correct_door
        q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
        q_static = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_idx].set(1.0)
        dist = np.asarray(bamcp.plan(
            q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
        ))
        # The argmax should be OPEN_DOOR_correct = M + correct_door.
        best_action = int(dist.argmax())
        assert best_action == env.M + correct, (
            f"theta={theta_idx}: expected OPEN_DOOR_{correct} "
            f"(action {env.M + correct}), got {best_action}; dist={dist}"
        )


def test_run_block_with_bamcp() -> None:
    env = DoorCodeEnv(K=3, probe_noise=0.0)
    bamcp = BAMCPPlanner(n_simulations=64, depth=4, rollout_seed=0)
    block = run_block(
        env, bamcp, theta_true=0, n_episodes=4,
        horizon=4, max_steps=4, deterministic=True, seed=0,
    )
    assert len(block.episodes) == 4


def test_run_outer_loop_with_bamcp() -> None:
    bamcp = BAMCPPlanner(n_simulations=64, depth=4, rollout_seed=0)
    result = run_outer_loop(
        env_kwargs=dict(K=3, probe_noise=0.0),
        planner=bamcp, n_seeds=2, n_episodes=3,
        horizon=4, max_steps=4, deterministic=True, seed=0,
        show_progress=False,
    )
    assert result.success.shape == (2, 3)


def test_resample_flag_changes_behaviour_under_diffuse_prior() -> None:
    """Literal-POMCP (resample_theta_per_sim=False) and BAMCP
    (=True) must produce measurably different visit-count distributions
    under a diffuse prior with non-trivial probe noise — otherwise the
    flag is a no-op. With seed-matched RNGs the per-sim resample draws
    K!*n_simulations theta values, while the literal variant draws 1
    plus per-step probe outcomes; the resulting trees diverge.
    """
    env = DoorCodeEnv(K=3, probe_noise=0.2)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[env._START].set(1.0)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    bamcp = BAMCPPlanner(
        n_simulations=128, depth=4, rollout_seed=0,
        resample_theta_per_sim=True,
    )
    literal = BAMCPPlanner(
        n_simulations=128, depth=4, rollout_seed=0,
        resample_theta_per_sim=False,
    )

    bamcp_dist = np.asarray(bamcp.plan(
        q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
    ))
    literal_dist = np.asarray(literal.plan(
        q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
    ))
    # Different sampling regimes -> different action visit distributions.
    assert not np.allclose(bamcp_dist, literal_dist, atol=1e-3), (
        f"resample flag had no effect: bamcp={bamcp_dist}, "
        f"literal={literal_dist}"
    )


def test_step_samples_stochastic_transitions() -> None:
    """When T(.|s,theta,a) is stochastic, _step must sample (not argmax)
    or stochastic-T envs (e.g. minigrid with slip_prob>0) collapse to a
    deterministic-T proxy and the rollout under-explores the slip mass.
    """
    n_states = 3
    T = np.zeros((n_states, n_states, 1, 1), dtype=np.float64)
    T[1, 0, 0, 0] = 0.7
    T[2, 0, 0, 0] = 0.3
    # Trivial probe / outcome (no signal): NEUTRAL emit at every state.
    B_position = np.eye(n_states, dtype=np.float64)
    B_probe = np.zeros((3, n_states, 1), dtype=np.float64)
    B_probe[2, :, :] = 1.0
    B_outcome = np.zeros((3, n_states), dtype=np.float64)
    B_outcome[0, :] = 1.0

    rng = np.random.default_rng(0)
    counts = {1: 0, 2: 0}
    for _ in range(2000):
        nxt, _, _ = BAMCPPlanner._step(
            state=0, theta=0, action=0,
            T=T, B_position=B_position, B_probe=B_probe, B_outcome=B_outcome,
            probe_state_set=set(), rng=rng,
        )
        counts[nxt] = counts.get(nxt, 0) + 1
    # Expected ~1400 / ~600 for 2000 draws at 0.7/0.3 split. Under argmax
    # we'd see 2000/0.
    assert 1200 < counts[1] < 1600, f"state-1 count off: {counts}"
    assert 400 < counts[2] < 800, f"state-2 count off: {counts}"


def test_tree_keys_distinguish_successors_under_identity_B_position() -> None:
    """With identity B_position, view_idx == next_state, so distinct
    successors of a stochastic transition give distinct obs_idx — siblings
    correctly build separate tree branches.
    """
    n_states = 3
    T = np.zeros((n_states, n_states, 1, 1), dtype=np.float64)
    T[1, 0, 0, 0] = 0.7
    T[2, 0, 0, 0] = 0.3
    B_position = np.eye(n_states, dtype=np.float64)
    B_probe = np.zeros((3, n_states, 1), dtype=np.float64)
    B_probe[2, :, :] = 1.0
    B_outcome = np.zeros((3, n_states), dtype=np.float64)
    B_outcome[0, :] = 1.0

    rng = np.random.default_rng(0)
    seen_obs_idx_per_next_state: dict[int, set[int]] = {1: set(), 2: set()}
    for _ in range(200):
        nxt, obs_idx, _ = BAMCPPlanner._step(
            state=0, theta=0, action=0,
            T=T, B_position=B_position, B_probe=B_probe, B_outcome=B_outcome,
            probe_state_set=set(), rng=rng,
        )
        seen_obs_idx_per_next_state[nxt].add(obs_idx)

    obs_for_1 = seen_obs_idx_per_next_state[1]
    obs_for_2 = seen_obs_idx_per_next_state[2]
    assert obs_for_1.isdisjoint(obs_for_2), (
        f"obs_idx collides across distinct successors — tree would "
        f"merge stochastic siblings: state1={obs_for_1}, state2={obs_for_2}"
    )


def test_tree_keys_alias_states_under_many_to_one_B_position() -> None:
    """With a many-to-one B_position (the minigrid case), latent states
    that share a view must produce the same obs_idx — otherwise the tree
    branches on unobservable state and leaks privileged localization.
    """
    n_states = 4
    n_views = 2
    # Two equally likely successors (states 1, 2) that map to the SAME view.
    T = np.zeros((n_states, n_states, 1, 1), dtype=np.float64)
    T[1, 0, 0, 0] = 0.5
    T[2, 0, 0, 0] = 0.5
    # B_position: states 0,1 -> view 0; states 2,3 -> view 1. So state 1
    # and state 2 emit different views; state 0 and state 1 alias.
    B_position = np.zeros((n_views, n_states), dtype=np.float64)
    B_position[0, [0, 1]] = 1.0
    B_position[1, [2, 3]] = 1.0
    B_probe = np.zeros((3, n_states, 1), dtype=np.float64)
    B_probe[2, :, :] = 1.0
    B_outcome = np.zeros((3, n_states), dtype=np.float64)
    B_outcome[0, :] = 1.0

    rng = np.random.default_rng(0)
    obs_idx_per_view: dict[int, set[int]] = {0: set(), 1: set()}
    for _ in range(200):
        nxt, obs_idx, _ = BAMCPPlanner._step(
            state=0, theta=0, action=0,
            T=T, B_position=B_position, B_probe=B_probe, B_outcome=B_outcome,
            probe_state_set=set(), rng=rng,
        )
        view = int(np.argmax(B_position[:, nxt]))
        obs_idx_per_view[view].add(obs_idx)

    # All same-view samples must share one obs_idx (no latent leak).
    assert len(obs_idx_per_view[0]) == 1, (
        f"obs_idx leaks latent state for view 0: {obs_idx_per_view[0]}"
    )
    assert len(obs_idx_per_view[1]) == 1, (
        f"obs_idx leaks latent state for view 1: {obs_idx_per_view[1]}"
    )
    # And distinct views must give distinct obs_idx.
    assert obs_idx_per_view[0].isdisjoint(obs_idx_per_view[1])


def test_root_state_is_sampled_from_q_state() -> None:
    """plan() must sample the root state from q_state (not argmax). With
    a uniform q_state over two states, two different RNG seeds must give
    different action distributions — otherwise the planner is collapsing
    to MAP and silently picking one aliased mode.
    """
    env = DoorCodeEnv(K=3, probe_noise=0.0, lock_length=1)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B, dtype=jnp.float32) for B in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask
    # Diffuse q_state over the two probe states (not one-hot). Different
    # roots produce different probe outcomes -> different visit counts.
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32)
    q_state = q_state.at[1].set(0.5).at[2].set(0.5)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    bamcp_a = BAMCPPlanner(n_simulations=128, depth=4, rollout_seed=0)
    bamcp_b = BAMCPPlanner(n_simulations=128, depth=4, rollout_seed=42)
    dist_a = np.asarray(bamcp_a.plan(
        q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
    ))
    dist_b = np.asarray(bamcp_b.plan(
        q_state, q_static, T, B_list, goal, horizon=4, theta_mask=theta_mask,
    ))
    assert not np.allclose(dist_a, dist_b, atol=1e-3), (
        f"different RNGs gave identical visit dists -> root_state likely "
        f"collapsed to argmax: a={dist_a}, b={dist_b}"
    )


# --- Depth semantics ---------------------------------------------------------
#
# `depth` bounds the TOTAL simulated lookahead of one simulation — tree
# descent PLUS leaf rollout — and the effective bound is
# min(depth, max(horizon, 1)). Every other test in this file uses
# depth == horizon, which makes `min(depth, horizon)`, depth-only and
# horizon-only indistinguishable and would let an off-by-one in the
# `depth - 1` recursion pass the whole suite. These tests separate the two
# knobs and pin the step budget exactly.


def _non_terminating_env(n_states: int, n_actions: int, n_static: int = 1):
    """A chain with no SUCCESS/FAILURE state, so nothing short-circuits.

    Early termination is the one thing that could legitimately make a
    simulation shorter than `max_depth`, so it must be absent for the
    step count to be a clean assertion about the depth budget.
    """
    T = np.zeros((n_states, n_states, n_static, n_actions), dtype=np.float64)
    for theta in range(n_static):
        for action in range(n_actions):
            for s in range(n_states):
                # Advance along the chain, absorbing at the end.
                T[min(s + 1, n_states - 1), s, theta, action] = 1.0
    B_position = np.eye(n_states, dtype=np.float64)
    B_probe = np.zeros((3, n_states, n_static), dtype=np.float64)
    B_probe[2, :, :] = 1.0
    # outcome 0 == NEUTRAL everywhere: no terminal, no reward.
    B_outcome = np.zeros((3, n_states), dtype=np.float64)
    B_outcome[0, :] = 1.0
    return T, [B_position, B_probe, B_outcome]


@pytest.mark.parametrize(
    "depth, horizon, expected_max_depth",
    [
        (3, 8, 3),   # depth < horizon -> depth binds
        (8, 3, 3),   # depth > horizon -> horizon binds (clip_horizon case)
        (4, 4, 4),   # equal, the only case the old suite covered
    ],
)
def test_steps_per_simulation_equals_max_depth(
    monkeypatch, depth, horizon, expected_max_depth,
) -> None:
    """Total env steps per simulation == min(depth, horizon), exactly.

    n_actions=1 forces every simulation down the same branch, so after the
    first few the descent is deep rather than shallow — which is what
    exercises the `depth - 1` recursion and its `depth <= 0` guard, not
    just the root-level rollout.
    """
    n_simulations = 6
    T, B_list = _non_terminating_env(n_states=12, n_actions=1)

    step_calls = []
    original_step = BAMCPPlanner._step

    def counting_step(*args, **kwargs):
        step_calls.append(1)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(BAMCPPlanner, "_step", staticmethod(counting_step))

    planner = BAMCPPlanner(
        n_simulations=n_simulations, depth=depth, rollout_seed=0,
    )
    q_state = jnp.zeros(12, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(1, dtype=jnp.float32)
    planner.plan(
        q_state, q_static,
        jnp.asarray(T, dtype=jnp.float32),
        [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
        goal=None, horizon=horizon,
    )

    assert len(step_calls) == n_simulations * expected_max_depth, (
        f"depth={depth}, horizon={horizon}: expected "
        f"{n_simulations} x {expected_max_depth} = "
        f"{n_simulations * expected_max_depth} env steps, "
        f"got {len(step_calls)}"
    )


@pytest.mark.parametrize(
    "depth, horizon, expected_max_depth",
    [(3, 8, 3), (8, 3, 3), (4, 4, 4)],
)
def test_numba_path_root_value_equals_max_depth(
    depth, horizon, expected_max_depth,
) -> None:
    """Same assertion for the Numba path, read off the returned Q.

    `_step` cannot be monkeypatched inside an @njit function, so instead
    every (state, action) pays reward exactly 1 and gamma is exactly 1.
    The backed-up root value is then the undiscounted sum over the whole
    simulation, i.e. the number of env steps it took — so Q(a) at the root
    must equal max_depth on the nose. This is what a depth off-by-one in
    either the descent or the `max_depth - d - 1` rollout budget would
    break.
    """
    n_states, n_actions, n_static = 12, 1, 1
    T, B_list = _non_terminating_env(n_states, n_actions, n_static)
    # The Numba path takes a θ-independent T with the θ axis dropped, and
    # θ-independent B channels without a θ tail. Same dynamics as the
    # legacy 4-D layout above, re-laid-out.
    T_3d = np.ascontiguousarray(T[:, :, 0, :])
    B_list = [B_list[0], np.ascontiguousarray(B_list[1][:, :, 0]), B_list[2]]

    registry = ModalityRegistry(modalities=(
        ObservationModality(
            name="position", cardinality=n_states,
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
    reward_table = np.ones((n_states, n_actions, n_static), dtype=np.float64)

    planner = BAMCPPlanner(
        n_simulations=6, depth=depth, gamma=1.0, rollout_seed=0,
    )
    q_values, _ = planner.plan(
        jnp.zeros(n_states, dtype=jnp.float32).at[0].set(1.0),
        jnp.ones(n_static, dtype=jnp.float32),
        jnp.asarray(T_3d, dtype=jnp.float32),
        [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
        goal=None, horizon=horizon,
        theta_spec=ThetaSpec(component_sizes=(n_static,)),
        t_factor_spec=FactorSpec(theta_components=()),
        b_factor_specs=(
            FactorSpec(theta_components=()),
            FactorSpec(theta_components=()),
            FactorSpec(theta_components=()),
        ),
        registry=registry,
        reward_fn=lambda *a, **k: 1.0,
        terminal_fn=None,
        reward_table=reward_table,
        return_q_values=True,
    )
    q = float(np.asarray(q_values)[0])
    assert q == pytest.approx(float(expected_max_depth), abs=1e-4), (
        f"depth={depth}, horizon={horizon}: root Q should equal the step "
        f"budget {expected_max_depth}, got {q}"
    )
