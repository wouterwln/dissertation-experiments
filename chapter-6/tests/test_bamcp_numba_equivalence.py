"""Statistical equivalence of the NumPy and Numba BAMCP paths.

``_bamcp_numba.py``'s module docstring has always claimed this file
exists; it did not, so the two implementations of the same algorithm had
no guard tying them together. That matters most when a change lands in
both at once (the rollout-policy work is exactly such a change).

The two paths cannot be bit-identical — the NumPy path draws from a
``np.random.Generator`` while the Numba path uses the global
``np.random`` state, and the tree is stored as linked ``_Node`` objects
vs parallel dense arrays. What must agree is the *distribution over root
actions*: both are estimating the same Q(a) by the same UCB1 tree search
over the same generative model, so at a large enough simulation budget
their visit-count distributions must concentrate on the same action and
their per-action means must agree within Monte-Carlo error.

The shared config is a hand-built 4-state chain where one action leads
to reward and the others do not, so the correct answer is unambiguous
and both paths can be driven through their own entry points
(``_plan_legacy`` via ``reward_fn=None``, ``_plan_numba`` via a supplied
``reward_fn``/``reward_table``).

Both rollout policies are covered. ``eps_greedy_q`` is the reason this
file exists — it is implemented twice, once per path, including a
tie-break and an optimistic initialisation that have to agree — so
testing only the default would leave the duplicated half unguarded. The
two paths' optimistic inits coincide here by construction: the legacy
path uses +1 (its outcome-channel reward scale) and ``reward_table``'s
max is +1.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.factors import FactorSpec, ThetaSpec
from learningpomdp.observations import (
    ModalityRegistry,
    ObservationModality,
    ThetaDependence,
)
from learningpomdp.planning.bamcp_planner import BAMCPPlanner


N_STATES = 4
N_ACTIONS = 3
N_STATIC = 2
# Action 0 from START(0) reaches the rewarding state 1; actions 1 and 2
# reach the neutral states 2 and 3. All of 1..3 self-loop.
GOOD_ACTION = 0
GOOD_STATE = 1


def _shared_env():
    """Build the (T_4d, B_list) pair the legacy path consumes.

    The two paths take different T *layouts* by design: the legacy path
    indexes ``T[:, s, theta, a]`` (4-D, as door_code emits) while the
    Numba path indexes ``T_eff[:, s, a]`` (3-D, as RockSample's
    θ-independent T emits). T here is θ-independent in content, so
    ``_t_3d()`` is an exact re-layout of the same dynamics rather than a
    different model — which is what makes the two paths comparable.
    """
    T = np.zeros((N_STATES, N_STATES, N_STATIC, N_ACTIONS), dtype=np.float64)
    for theta in range(N_STATIC):
        for action in range(N_ACTIONS):
            # From START, action a -> state a+1.
            T[action + 1, 0, theta, action] = 1.0
            # Absorbing self-loops elsewhere.
            for s in range(1, N_STATES):
                T[s, s, theta, action] = 1.0

    # Channel 0 = position: identity, so view == state.
    B_position = np.eye(N_STATES, dtype=np.float64)
    # Channel 1 = probe: θ-conditional but uninformative here (NEUTRAL
    # everywhere) so the two paths' *observation* streams cannot diverge
    # on content, only on RNG consumption.
    B_probe = np.zeros((3, N_STATES, N_STATIC), dtype=np.float64)
    B_probe[2, :, :] = 1.0
    # Channel 2 = outcome: SUCCESS at GOOD_STATE, NEUTRAL elsewhere. The
    # legacy path derives its reward from this channel.
    B_outcome = np.zeros((3, N_STATES), dtype=np.float64)
    B_outcome[0, :] = 1.0
    B_outcome[0, GOOD_STATE] = 0.0
    B_outcome[1, GOOD_STATE] = 1.0
    return T, [B_position, B_probe, B_outcome]


def _t_3d(T_4d: np.ndarray) -> np.ndarray:
    """Drop the θ axis: T is θ-independent, so every slice is identical."""
    assert np.allclose(T_4d[:, :, 0, :], T_4d[:, :, 1, :])
    return np.ascontiguousarray(T_4d[:, :, 0, :])


def _numba_side_inputs():
    """registry / specs / reward table matching ``_shared_env``.

    The Numba path reads reward from ``reward_table[state, action, θ]``,
    keyed on the state the agent is *leaving*. So reward +1 is attached
    to (START, GOOD_ACTION), which is the transition the legacy path
    scores +1 via the outcome channel at the state it arrives in.
    """
    registry = ModalityRegistry(modalities=(
        ObservationModality(
            name="position", cardinality=N_STATES,
            theta_dependence=ThetaDependence.INDEPENDENT,
        ),
        ObservationModality(
            name="probe", cardinality=3,
            theta_dependence=ThetaDependence.CONDITIONAL,
            informative_states=frozenset({GOOD_STATE}),
        ),
        ObservationModality(
            name="outcome", cardinality=3,
            theta_dependence=ThetaDependence.INDEPENDENT,
        ),
    ))
    theta_spec = ThetaSpec(component_sizes=(N_STATIC,))
    t_factor_spec = FactorSpec(theta_components=())
    b_factor_specs = (
        FactorSpec(theta_components=()),
        FactorSpec(theta_components=(0,)),
        FactorSpec(theta_components=()),
    )
    reward_table = np.zeros((N_STATES, N_ACTIONS, N_STATIC), dtype=np.float64)
    reward_table[0, GOOD_ACTION, :] = 1.0
    return registry, theta_spec, t_factor_spec, b_factor_specs, reward_table


def _root_beliefs():
    q_state = jnp.zeros(N_STATES, dtype=jnp.float32).at[0].set(1.0)
    q_static = jnp.ones(N_STATIC, dtype=jnp.float32) / N_STATIC
    return q_state, q_static


def _plan_numpy(
    seed: int, n_simulations: int, depth: int,
    rollout_policy: str = "uniform",
) -> np.ndarray:
    T, B_list = _shared_env()
    q_state, q_static = _root_beliefs()
    planner = BAMCPPlanner(
        n_simulations=n_simulations, depth=depth, rollout_seed=seed,
        rollout_policy=rollout_policy,
    )
    # reward_fn=None selects the legacy NumPy path.
    return np.asarray(planner.plan(
        jnp.asarray(q_state), jnp.asarray(q_static),
        jnp.asarray(T, dtype=jnp.float32),
        [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
        goal=None, horizon=depth,
    ), dtype=np.float64)


def _plan_numba(
    seed: int, n_simulations: int, depth: int,
    rollout_policy: str = "uniform",
) -> np.ndarray:
    T, B_list = _shared_env()
    q_state, q_static = _root_beliefs()
    registry, theta_spec, t_spec, b_specs, reward_table = _numba_side_inputs()
    planner = BAMCPPlanner(
        n_simulations=n_simulations, depth=depth, rollout_seed=seed,
        rollout_policy=rollout_policy,
    )
    # A non-None reward_fn selects the Numba path; reward_table short-
    # circuits the slow Python fallback that would otherwise call it.
    return np.asarray(planner.plan(
        jnp.asarray(q_state), jnp.asarray(q_static),
        jnp.asarray(_t_3d(T), dtype=jnp.float32),
        [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
        goal=None, horizon=depth,
        theta_spec=theta_spec,
        t_factor_spec=t_spec,
        b_factor_specs=b_specs,
        registry=registry,
        reward_fn=lambda *a, **k: 0.0,
        terminal_fn=None,
        reward_table=reward_table,
    ), dtype=np.float64)


@pytest.mark.parametrize("rollout_policy", ["uniform", "eps_greedy_q"])
def test_both_paths_identify_the_rewarding_action(rollout_policy) -> None:
    """The weakest possible agreement, and the one that must never break:
    both implementations must prefer the same root action."""
    dist_np = _plan_numpy(0, 512, 3, rollout_policy)
    dist_nb = _plan_numba(0, 512, 3, rollout_policy)
    assert int(dist_np.argmax()) == GOOD_ACTION, f"NumPy path: {dist_np}"
    assert int(dist_nb.argmax()) == GOOD_ACTION, f"Numba path: {dist_nb}"


@pytest.mark.parametrize("rollout_policy", ["uniform", "eps_greedy_q"])
def test_root_action_distributions_agree_within_monte_carlo_error(
    rollout_policy,
) -> None:
    """Per-action visit shares must match across paths.

    Averaging over several rollout seeds kills the per-seed noise that
    the two RNG regimes cannot share. The tolerance is deliberately loose
    enough to pass on RNG divergence alone but far too tight to survive a
    real semantic divergence (e.g. an off-by-one in the depth budget,
    which shifts visit mass between the rewarding and neutral branches).
    """
    seeds = range(8)
    n_simulations = 512
    depth = 3
    np_dists = np.stack([
        _plan_numpy(s, n_simulations, depth, rollout_policy) for s in seeds
    ])
    nb_dists = np.stack([
        _plan_numba(s, n_simulations, depth, rollout_policy) for s in seeds
    ])
    mean_np = np_dists.mean(axis=0)
    mean_nb = nb_dists.mean(axis=0)
    # Pooled standard error of the difference of two seed-means.
    n = len(list(seeds))
    se = np.sqrt(
        np_dists.var(axis=0, ddof=1) / n + nb_dists.var(axis=0, ddof=1) / n
    )
    # 3 SE, floored so exactly-deterministic components (se == 0) still
    # get a small absolute tolerance rather than demanding bit-equality.
    tol = np.maximum(3.0 * se, 0.05)
    assert np.all(np.abs(mean_np - mean_nb) <= tol), (
        f"root action distributions diverge beyond Monte-Carlo error:\n"
        f"  numpy={mean_np}\n  numba={mean_nb}\n  tol={tol}"
    )


def test_both_paths_agree_on_root_q_values() -> None:
    """``return_q_values=True`` (the flag IDS relies on) must agree too —
    the visit distribution can hide a Q-scale disagreement because it is
    driven by UCB *ranking*, not by the Q magnitudes."""
    seeds = range(8)
    q_np, q_nb = [], []
    for s in seeds:
        T, B_list = _shared_env()
        q_state, q_static = _root_beliefs()
        registry, theta_spec, t_spec, b_specs, reward_table = _numba_side_inputs()
        planner_np = BAMCPPlanner(n_simulations=512, depth=3, rollout_seed=s)
        vals_np, _ = planner_np.plan(
            jnp.asarray(q_state), jnp.asarray(q_static),
            jnp.asarray(T, dtype=jnp.float32),
            [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
            goal=None, horizon=3, return_q_values=True,
        )
        planner_nb = BAMCPPlanner(n_simulations=512, depth=3, rollout_seed=s)
        vals_nb, _ = planner_nb.plan(
            jnp.asarray(q_state), jnp.asarray(q_static),
            jnp.asarray(_t_3d(T), dtype=jnp.float32),
            [jnp.asarray(B, dtype=jnp.float32) for B in B_list],
            goal=None, horizon=3,
            theta_spec=theta_spec, t_factor_spec=t_spec,
            b_factor_specs=b_specs, registry=registry,
            reward_fn=lambda *a, **k: 0.0, terminal_fn=None,
            reward_table=reward_table, return_q_values=True,
        )
        q_np.append(np.asarray(vals_np, dtype=np.float64))
        q_nb.append(np.asarray(vals_nb, dtype=np.float64))
    mean_np = np.stack(q_np).mean(axis=0)
    mean_nb = np.stack(q_nb).mean(axis=0)
    assert np.allclose(mean_np, mean_nb, atol=0.1), (
        f"root Q values diverge: numpy={mean_np}, numba={mean_nb}"
    )
