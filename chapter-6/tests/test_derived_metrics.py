"""Unit tests for the derived-metrics helpers shared by all runners.

`build_derived_metrics` composes `calibration_curves` (per-episode Brier
and NLL across seeds) and `episodes_to_identify` (first-crossing scalar
per seed). The edge cases worth pinning down:

  - all seeds cross at episode 0 (immediate identification)
  - no seeds cross within the budget (median/p25/p75 = None)
  - mixed crossing (median/IQR computed over crossing seeds only)
  - p_true exactly 0 (NLL must clip, not blow up)
"""

import numpy as np

from learningpomdp.experiment.door_code_runner import (
    build_derived_metrics,
    calibration_curves,
    episodes_to_identify,
)


def test_episodes_to_identify_all_cross_immediately() -> None:
    # Every seed already at 1.0 on episode 0 → median = p25 = p75 = 0,
    # fraction = 1.0.
    posterior = np.ones((4, 5))
    out = episodes_to_identify(posterior)
    assert out["median"] == 0.0
    assert out["p25"] == 0.0
    assert out["p75"] == 0.0
    assert out["fraction_identified"] == 1.0
    assert out["per_seed"] == [0, 0, 0, 0]
    assert out["n_episodes_budget"] == 5


def test_episodes_to_identify_no_seed_crosses() -> None:
    # Posterior never reaches the 0.9 threshold; medians stay None.
    posterior = np.full((3, 5), 0.5)
    out = episodes_to_identify(posterior)
    assert out["median"] is None
    assert out["p25"] is None
    assert out["p75"] is None
    assert out["fraction_identified"] == 0.0
    assert out["per_seed"] == [None, None, None]


def test_episodes_to_identify_mixed_crossing() -> None:
    # Seed 0 crosses at ep 1, seed 1 at ep 3, seed 2 never. Median over
    # the two crossing seeds is mean(1, 3) = 2.0; IQR matches.
    posterior = np.array(
        [
            [0.5, 0.95, 0.95, 0.95],
            [0.5, 0.5, 0.5, 0.95],
            [0.1, 0.1, 0.1, 0.1],
        ]
    )
    out = episodes_to_identify(posterior)
    assert out["per_seed"] == [1, 3, None]
    assert out["median"] == 2.0
    assert out["fraction_identified"] == 2 / 3


def test_episodes_to_identify_respects_threshold() -> None:
    posterior = np.array([[0.5, 0.7, 0.85]])
    out = episodes_to_identify(posterior, threshold=0.8)
    assert out["per_seed"] == [2]
    assert out["median"] == 2.0


def test_calibration_curves_zero_mass_nll_clipped() -> None:
    # p_true = 0 → unclipped NLL is +inf; helper must clip and return
    # a finite value.
    posterior = np.zeros((2, 3))
    out = calibration_curves(posterior)
    nll_mean = np.asarray(out["nll"]["mean"])
    assert np.all(np.isfinite(nll_mean))
    # Brier with p=0 is (1 - 0)^2 = 1.0 per (seed, episode); mean = 1.
    brier_mean = np.asarray(out["brier"]["mean"])
    assert np.allclose(brier_mean, 1.0)


def test_calibration_curves_perfect_identification() -> None:
    # p_true = 1 → Brier = 0 and NLL = 0 across the board.
    posterior = np.ones((2, 3))
    out = calibration_curves(posterior)
    assert np.allclose(out["brier"]["mean"], 0.0)
    assert np.allclose(out["nll"]["mean"], 0.0)


def test_build_derived_metrics_bundles_both() -> None:
    posterior = np.array([[0.3, 0.95], [0.4, 0.5]])
    out = build_derived_metrics(posterior)
    assert set(out.keys()) == {"brier", "nll", "episodes_to_identify"}
    assert out["episodes_to_identify"]["fraction_identified"] == 0.5
