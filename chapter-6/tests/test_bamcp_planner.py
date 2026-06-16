"""Tests for BAMCPPlanner (and its literal-POMCP ablation flag)."""

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.experiment.door_code_runner import run_block, run_outer_loop
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
