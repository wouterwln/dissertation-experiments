"""Hardware-independent per-decision cost ceiling for AIF-MP.

This replaces cross-hardware wall-clock as the basis for the compute
comparison, so it has to be arithmetic anyone can check by hand.
"""

from learningpomdp.environments.door_code import DoorCodeEnv
from learningpomdp.planning.cost_model import aif_mp_contraction_count


def test_aif_contraction_count_matches_hand_computation():
    """Hand-computed on a deliberately tiny, all-distinct-dimensions config.

    dynamics: 6 contractions x horizon 3 x |x|^2 (16) x |u| 5   = 1440
    obs ch 0: theta-independent -> 2 x n_obs 2 x |x| 4 x 1       =   16
    obs ch 1: theta-dependent   -> 3 x n_obs 3 x |x| 4 x |th| 7  =  252
    per iteration                                                = 1708
    x 2 iterations                                               = 3416
    """
    assert aif_mp_contraction_count(
        n_iterations=2,
        horizon=3,
        n_states=4,
        n_actions=5,
        obs_channels=[(2, 1), (3, 7)],
    ) == 3416


def test_aif_contraction_count_is_linear_in_iterations():
    """Iterations are a pure multiplier — both loops run n_iterations times,
    so the sensitivity sweep's cost axis is exactly proportional to it."""
    dims = dict(horizon=4, n_states=6, n_actions=3, obs_channels=[(2, 5)])
    one = aif_mp_contraction_count(n_iterations=1, **dims)
    assert aif_mp_contraction_count(n_iterations=70, **dims) == 70 * one


def test_aif_contraction_count_on_canonical_door_code():
    """Pins the ceiling against the real canonical door_code dimensions, so
    a change in env shape or in the counted contractions is visible here.

    Evaluated at the nominal `runner.horizon` of 8, so it is a per-decision
    ceiling: under `runner.clip_horizon` the horizon plan() receives near
    the end of an episode is shorter and the realized count is lower.
    """
    env = DoorCodeEnv(K=6, lock_length=4, probe_noise=0.2)
    obs_channels = [
        (B.shape[0], int(B.shape[2]) if B.ndim == 3 else 1)
        for B in env.get_observation_tensors()
    ]
    assert (env.n_states, env.n_actions) == (16, 16)
    assert obs_channels == [(16, 1), (3, 720), (3, 1)]

    count = aif_mp_contraction_count(
        n_iterations=70, horizon=8,
        n_states=env.n_states, n_actions=env.n_actions,
        obs_channels=obs_channels,
    )
    # 70 x (6*8*16^2*16 + 2*16*16 + 3*3*16*720 + 2*3*16) = 70 x 300_896
    assert count == 21_062_720

    # And it really is a ceiling in the horizon: the clipped-horizon
    # decisions near the end of an episode cost strictly less.
    clipped = aif_mp_contraction_count(
        n_iterations=70, horizon=1,
        n_states=env.n_states, n_actions=env.n_actions,
        obs_channels=obs_channels,
    )
    assert clipped < count


def test_theta_only_enters_through_the_obs_channels():
    """The dynamics loop has no theta axis (T is marginalized over the prior
    before the loop), so growing |theta| must leave the dynamics term alone."""
    base = dict(n_iterations=1, horizon=2, n_states=3, n_actions=4)
    dyn_only = aif_mp_contraction_count(**base, obs_channels=[])
    small = aif_mp_contraction_count(**base, obs_channels=[(2, 10)])
    large = aif_mp_contraction_count(**base, obs_channels=[(2, 100)])
    assert dyn_only == 6 * 2 * 9 * 4
    assert (large - dyn_only) == 10 * (small - dyn_only)


def test_theta_independent_channel_is_cheaper_than_theta_dependent():
    base = dict(n_iterations=1, horizon=1, n_states=2, n_actions=1)
    indep = aif_mp_contraction_count(**base, obs_channels=[(4, 1)])
    dep = aif_mp_contraction_count(**base, obs_channels=[(4, 2)])
    assert dep > indep


def test_count_is_an_exact_integer_at_paper_grade_budgets():
    """An element tally must come back as an exact Python int, not a float
    that happens to print like one: the reported figure is quoted to the
    digit, and unbounded ints are what keep it that way as budgets grow."""
    n = aif_mp_contraction_count(
        n_iterations=10_000, horizon=25, n_states=512, n_actions=7,
        obs_channels=[(3, 4096)],
    )
    assert isinstance(n, int)
    assert n == 10_000 * (6 * 25 * 512**2 * 7 + 3 * 3 * 512 * 4096)
    assert n == 2_941_255_680_000
