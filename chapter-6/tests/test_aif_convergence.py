"""Convergence diagnostics for the AIF-MP damped fixed-point iteration.

The planner runs a fixed `n_iterations` at fixed damping with no stopping
criterion, and nothing was reported about whether that reaches a fixed
point. `AIFPlanner.plan(..., return_residuals=True)` exposes the
per-iteration channel residual (`bp_core.channel_tv_residual`) so the
question is answerable.

The load-bearing test here is the bit-identity one: the diagnostic must
be free, i.e. the default `return_residuals=False` path must produce
exactly the bits the planner produced before the instrumentation existed.

The measured behaviour: the residual falls by two to three orders of
magnitude over the iteration budget, non-monotonically — it is small but
still shrinking at the canonical 70. So the tests assert *decay by orders
of magnitude*, and deliberately assert neither monotonicity (false) nor a
lower bound on the tail (which would pin an artifact of the measure
rather than a property of the iteration).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from learningpomdp.environments.door_code import DoorCodeEnv, state_indices
from learningpomdp.planning.aif_planner import AIFPlanner
from learningpomdp.planning.bp_core import channel_tv_residual
from learningpomdp.planning.messages import LOG_ZERO

# The paper's canonical message-passing cell (params.yaml planner.{iterations,
# damping}); every case here uses it unless it is the axis under test.
CANONICAL_ITERATIONS = 70
CANONICAL_DAMPING = 0.9


def _instance(K: int = 3, lock_length: int = 2):
    """A small door_code instance plus everything plan() needs."""
    env = DoorCodeEnv(K=K, lock_length=lock_length, probe_noise=0.2)
    T = jnp.asarray(env.get_transition_tensor(), dtype=jnp.float32)
    B_list = [
        jnp.asarray(B_c, dtype=jnp.float32)
        for B_c in env.get_observation_tensors()
    ]
    goal = jnp.asarray(env.get_goal(), dtype=jnp.float32)
    layout = state_indices(env.M, lock_length)
    q_state = (
        jnp.zeros(env.n_states, dtype=jnp.float32).at[layout["START"]].set(1.0)
    )
    q_static = jnp.ones(env.n_static, dtype=jnp.float32) / env.n_static
    kwargs = dict(
        theta_spec=env.theta_spec,
        t_factor_spec=env.t_factor_spec,
        b_factor_specs=env.b_factor_specs,
        theta_mask=env.get_modality_registry().theta_dependent_mask,
    )
    return env, (q_state, q_static, T, B_list, goal), kwargs


@pytest.mark.parametrize("K,lock_length,horizon", [(3, 2, 3), (4, 2, 4)])
def test_return_residuals_false_is_bit_identical(K, lock_length, horizon):
    """The canonical path is numerically untouched by the instrumentation.

    Bit-exact, not approximate: `return_residuals=False` must trace the
    same program, so any drift here means the diagnostic leaked into the
    planning arithmetic.
    """
    _, args, kwargs = _instance(K, lock_length)
    planner = AIFPlanner(
        n_iterations=CANONICAL_ITERATIONS, damping=CANONICAL_DAMPING,
    )

    plain = planner.plan(*args, horizon, **kwargs)
    instrumented, _ = planner.plan(
        *args, horizon, return_residuals=True, **kwargs,
    )

    assert not isinstance(plain, tuple), (
        "return_residuals=False must return a bare action distribution"
    )
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(instrumented))


def test_residual_trace_shape_and_domain():
    """One residual per iteration, each a total variation distance.

    The [0, 1] bound is the useful half: it is what catches the
    degenerate-conditional failure mode, where a structurally empty slice
    normalizes to all-ones and inflates the residual past 1.
    """
    _, args, kwargs = _instance()
    for n_iterations in (5, 20, CANONICAL_ITERATIONS):
        planner = AIFPlanner(
            n_iterations=n_iterations, damping=CANONICAL_DAMPING,
        )
        _, residuals = planner.plan(
            *args, 3, return_residuals=True, **kwargs,
        )
        residuals = np.asarray(residuals)
        assert residuals.shape == (n_iterations,)
        assert np.all(np.isfinite(residuals))
        assert np.all((residuals >= 0.0) & (residuals <= 1.0))


def test_residual_trace_decays_by_orders_of_magnitude():
    """The residual shrinks from O(1) to O(1e-3) over the iteration budget.

    Asserted as decay across decades rather than step-by-step, because the
    decay is not monotone: it slows to a near-shelf in the middle of the
    trace before resuming. The trace is small but still shrinking at the
    canonical 70, so this pins "converging", not "converged" — and no
    lower bound on the tail is asserted.
    """
    _, args, kwargs = _instance()
    planner = AIFPlanner(
        n_iterations=140, damping=CANONICAL_DAMPING,
    )
    _, residuals = planner.plan(*args, 3, return_residuals=True, **kwargs)
    residuals = np.asarray(residuals)

    # Starts far from the fixed point: the channels are initialized uniform.
    assert residuals[0] > 0.1

    # Down by well over an order of magnitude by the canonical iteration
    # count, and lower again by the end of the budget.
    assert residuals[CANONICAL_ITERATIONS - 1] < residuals[0] / 50
    assert residuals[-1] < residuals[CANONICAL_ITERATIONS - 1]

    # Decreasing, not oscillating around a level: every decade of the trace
    # sits below the one before it.
    decades = [residuals[:14], residuals[14:70], residuals[70:]]
    assert [d.max() for d in decades] == sorted(
        (d.max() for d in decades), reverse=True
    )


def test_argmax_action_is_stable_across_iteration_count():
    """The *decision* is insensitive to the iteration count, which is why
    the un-justified fixed 70 has gone unnoticed.

    The action *distribution* is not: it sharpens monotonically onto that
    same argmax as the budget grows. So this is a statement about argmax
    control only, which is what the runners use.
    """
    _, args, kwargs = _instance()
    dists = {
        n: np.asarray(
            AIFPlanner(n_iterations=n, damping=CANONICAL_DAMPING).plan(
                *args, 3, **kwargs,
            )
        )
        for n in (10, 20, 40, CANONICAL_ITERATIONS, 140)
    }
    argmaxes = {n: int(d.argmax()) for n, d in dists.items()}
    assert len(set(argmaxes.values())) == 1, (
        f"argmax action changed with iteration count: {argmaxes}"
    )
    peaks = [dists[n].max() for n in (10, 20, 40, CANONICAL_ITERATIONS, 140)]
    assert peaks == sorted(peaks), f"peak mass not monotone: {peaks}"


def test_channel_tv_residual_ignores_probability_negligible_log_swings():
    """The reason the measure lives in probability space.

    Entry 0 moves by 44 in log space while moving the distribution by
    ~1e-11. A log-space `max |Δ log r|` would report 44 and drown out
    every entry that carries mass; the TV reports ~0.
    """
    log_old = jnp.array([[-69.0, 0.0]])
    log_new = jnp.array([[-25.0, 0.0]])
    assert float(channel_tv_residual(log_old, log_new, cond_axis=1)) < 1e-9


def test_channel_tv_residual_ignores_structural_zeros_for_free():
    """exp(LOG_ZERO) is exactly 0, so structural zeros need no mask."""
    log_old = jnp.array([[LOG_ZERO, -jnp.log(2.0), -jnp.log(2.0)]])
    log_new = jnp.array([[LOG_ZERO, -jnp.log(4.0), jnp.log(0.75)]])
    # Only entries 1 and 2 move: |0.5-0.25| + |0.5-0.75| = 0.5, halved.
    assert float(
        channel_tv_residual(log_old, log_new, cond_axis=1)
    ) == pytest.approx(0.25, abs=1e-6)


def test_channel_tv_residual_drops_degenerate_conditionals():
    """A structurally empty slice normalizes to all-ones (0/0), which is not
    a distribution and must not be reported as a residual of (n-1)/2."""
    log_old = jnp.array([
        [LOG_ZERO, LOG_ZERO, LOG_ZERO],       # empty -> all-zero mass
        [-jnp.log(3.0)] * 3,                  # a real conditional
    ])
    log_new = jnp.array([
        [0.0, 0.0, 0.0],                      # the 0/0 all-ones artifact
        [-jnp.log(2.0), -jnp.log(2.0), LOG_ZERO],
    ])
    # Slice 0 is dropped; slice 1 gives 0.5*(1/6 + 1/6 + 1/3) = 1/3.
    assert float(
        channel_tv_residual(log_old, log_new, cond_axis=1)
    ) == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_channel_tv_residual_is_zero_at_a_fixed_point():
    """Zero iff nothing moved — and nonzero as soon as something does, so
    the zero above is not the all-slices-dropped answer in disguise."""
    log_r = jnp.log(jnp.array([[0.3, 0.7], [0.9, 0.1]]))
    assert float(channel_tv_residual(log_r, log_r, cond_axis=1)) == 0.0

    log_moved = jnp.log(jnp.array([[0.3, 0.7], [0.8, 0.2]]))
    assert float(
        channel_tv_residual(log_r, log_moved, cond_axis=1)
    ) == pytest.approx(0.1, abs=1e-6)
