"""M2 gate: single-episode AIF planning on the door-code env.

The AIF planner already factorizes x and theta via the 4D tensor
convention; no augmented-state utilities are needed. These tests check
the behavioral gate from M2 of door_code_system_id.md:

- With q(theta) sharp on the true theta, AIF opens the correct door
  directly (no probe).
- With q(theta) uniform, AIF visits a probe before opening any door.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import (
    DoorCodeEnv,
    enumerate_permutations,
    state_indices,
)
from learningpomdp.planning.aif_planner import AIFPlanner


def _planner_inputs(env: DoorCodeEnv):
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [jnp.asarray(B_c, dtype=jnp.float32) for B_c in env.get_observation_tensors()]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    theta_mask = env.get_modality_registry().theta_dependent_mask
    return T, B_list, goal, theta_mask


def _spec_kwargs(env: DoorCodeEnv) -> dict:
    return dict(
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
    )


def _q_state_at_start(env: DoorCodeEnv) -> jnp.ndarray:
    layout = state_indices(env.M)
    q_state = jnp.zeros(env.n_states, dtype=jnp.float32).at[layout["START"]].set(1.0)
    return q_state


@pytest.mark.parametrize("K", [3, 4])
def test_aif_picks_correct_door_under_sharp_prior(K: int) -> None:
    """If q(theta) is concentrated on the true theta, AIF should open the
    correct door from START — no probing needed."""
    env = DoorCodeEnv(K=K, probe_noise=0.0)
    T, B_list, goal, theta_mask = _planner_inputs(env)
    planner = AIFPlanner()
    horizon = 3

    perms = enumerate_permutations(K)

    for theta_idx in range(env.n_static):
        correct_door = perms[theta_idx][0]
        correct_action = env.M + correct_door

        q_state = _q_state_at_start(env)
        q_static = jnp.zeros(env.n_static, dtype=jnp.float32).at[theta_idx].set(1.0)

        action_dist = planner.plan(
            q_state, q_static, T, B_list, goal, horizon,
            theta_mask=theta_mask, **_spec_kwargs(env),
        )
        action_dist = np.asarray(action_dist)

        best = int(action_dist.argmax())
        assert best == correct_action, (
            f"K={K}, theta={theta_idx} (perm={perms[theta_idx]}): expected "
            f"OPEN_DOOR_{correct_door} (action {correct_action}), "
            f"got action {best}"
        )


@pytest.mark.parametrize("K", [3, 4])
def test_aif_prefers_probe_under_uniform_prior(K: int) -> None:
    """With uniform q(theta), AIF should put more mass on probe actions
    than door actions — committing to any door is 1/K-success, but probing
    first lets the agent reduce theta uncertainty and improve the next
    door commit's expected reward."""
    env = DoorCodeEnv(K=K, probe_noise=0.0)
    T, B_list, goal, theta_mask = _planner_inputs(env)
    planner = AIFPlanner()
    horizon = 3

    q_state = _q_state_at_start(env)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    action_dist = planner.plan(
        q_state, q_static, T, B_list, goal, horizon,
        theta_mask=theta_mask, **_spec_kwargs(env),
    )
    action_dist = np.asarray(action_dist)

    probe_mass = action_dist[:env.M].sum()
    door_mass = action_dist[env.M:].sum()

    assert probe_mass > door_mass, (
        f"K={K}: expected probe_mass > door_mass under uniform q(theta); "
        f"got probe_mass={probe_mass:.4f}, door_mass={door_mass:.4f}"
    )

    best = int(action_dist.argmax())
    assert best < env.M, (
        f"K={K}: expected argmax action to be a probe (< {env.M}); got {best}"
    )


@pytest.mark.parametrize("K,probe_noise", [(3, 0.0), (3, 0.2), (4, 0.0)])
def test_aif_still_prefers_probe_with_noise(K: int, probe_noise: float) -> None:
    """Probes remain attractive even when noisy — the EFE accounts for
    expected information gain under the noisy likelihood."""
    env = DoorCodeEnv(K=K, probe_noise=probe_noise)
    T, B_list, goal, theta_mask = _planner_inputs(env)
    planner = AIFPlanner()
    horizon = 3

    q_state = _q_state_at_start(env)
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static

    action_dist = np.asarray(planner.plan(
        q_state, q_static, T, B_list, goal, horizon,
        theta_mask=theta_mask, **_spec_kwargs(env),
    ))

    probe_mass = action_dist[:env.M].sum()
    door_mass = action_dist[env.M:].sum()
    assert probe_mass > door_mass, (
        f"K={K}, noise={probe_noise}: probe_mass={probe_mass:.4f} "
        f"<= door_mass={door_mass:.4f}"
    )
