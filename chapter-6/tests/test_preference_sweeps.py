"""Numerical guards on the goal-prior preference sweep grids.

Three things are pinned here.

**The clamp floor.** Every swept cell in ``door_code_preference``,
``rock_sample_preference`` and ``factored_tiger_preference`` must produce a
goal prior the planners can actually read: they take ``log C`` through
``messages.safe_log``, which clamps at ``log(max(C, SAFE_LOG_FLOOR))``, so
a cell whose smallest entry sits at or under that floor silently flattens
the preference over the states beneath it — e.g. Factored Tiger at K=6 with
bad_logit=20 cannot distinguish 4 tigers opened from 6.

**The reward -> log-C transformation.** Both ``create_goal`` docstrings
claim that ``E[Δ log C]`` for the commit action is affine in the belief with
a zero crossing at ``bad_logit / (good_logit + bad_logit)``. That claim is
the whole answer to the reviewer's second ask, so it is checked numerically
rather than left as prose — it holds only because the per-θ softmax
normaliser cancels in the state-to-state difference.

**The heatmap row order.** Both heatmaps sort the penalty axis so that
magnitude grows downwards, but the two sorts run in opposite directions
(door_code's ``failure_logit`` is negative, Tiger's ``bad_logit`` is a
positive magnitude). A row order that moved without its tick labels would
mislabel every row — a data error, not a cosmetic one — so the rendered
figures are checked cell-by-cell against their own axis labels.

The grids are read from ``params.yaml`` on purpose: adding a value to a
sweep axis has to fail here rather than after the GPU hours are spent.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from learningpomdp.environments.base import SAFE_LOG_FLOOR
from learningpomdp.environments.door_code import (
    create_goal as door_code_goal,
    default_partitions,
)
from learningpomdp.environments.factored_tiger import (
    create_goal as factored_tiger_goal,
    n_events_for,
)
from learningpomdp.environments.rock_sample import (
    create_goal as rock_sample_goal,
    encode_state,
    sample_layout,
)

# Headroom demanded on top of the clamp floor: two decades, so no cell can
# creep onto the floor via a K bump without anyone noticing. The tightest
# swept cell — Factored Tiger (1, 10) at K=6, min entry 6.7e-28 — clears
# the resulting 1e-28 threshold by only ~6.7x, which is exactly why
# bad_logit stops at 10.
FLOOR_MARGIN = 100.0

PARAMS = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "params.yaml").read_text()
)


def _cells(block: str) -> list[tuple[float, float]]:
    """Every swept (good_logit, bad_logit) pair, either shape of sweep:
    an enumerated cross (``cells``) or a full grid (``*_logits`` axes)."""
    grid = PARAMS[block]
    if "cells" in grid:
        return [(c["good"], c["bad"]) for c in grid["cells"].values()]
    return list(itertools.product(grid["good_logits"], grid["bad_logits"]))


def _assert_clears_floor(goal: np.ndarray, label: str) -> None:
    # float32 is what the runners cast the goal to before planning.
    min_entry = float(goal.astype(np.float32).min())
    assert min_entry > FLOOR_MARGIN * SAFE_LOG_FLOOR, (
        f"{label}: min goal entry {min_entry:.3e} does not clear the "
        f"float32 safe_log floor {SAFE_LOG_FLOOR:.0e} by {FLOOR_MARGIN:g}x"
    )


@pytest.mark.parametrize("good,bad", _cells("factored_tiger_preference"))
def test_factored_tiger_preference_cell_clears_safe_log_floor(good, bad):
    goal = factored_tiger_goal(
        PARAMS["factored_tiger"]["env"]["K"],
        good_logit=float(good),
        bad_logit=float(bad),
    )
    _assert_clears_floor(goal, f"factored_tiger (good={good}, bad={bad})")


@pytest.mark.parametrize("good,bad", _cells("rock_sample_preference"))
def test_rock_sample_preference_cell_clears_safe_log_floor(good, bad):
    env_params = PARAMS["rock_sample"]["env"]
    layout = sample_layout(
        grid_size=env_params["grid_size"],
        K=env_params["K"],
        seed=env_params["layout_seed"],
    )
    goal = rock_sample_goal(
        layout, good_logit=float(good), bad_logit=float(bad),
    )
    _assert_clears_floor(goal, f"rock_sample (good={good}, bad={bad})")


@pytest.mark.parametrize("succ,fail", list(itertools.product(
    PARAMS["door_code_preference"]["success_logits"],
    PARAMS["door_code_preference"]["failure_logits"],
)))
def test_door_code_preference_cell_clears_safe_log_floor(succ, fail):
    # door_code's logit span is success - failure and does not grow with K,
    # so this grid clears the floor by ~20 decades. Pinned anyway so the
    # three sweeps fail the same way if a future grid widens the span.
    K = PARAMS["env"]["K"]
    goal = door_code_goal(
        K, len(default_partitions(K)),
        success_logit=float(succ), failure_logit=float(fail),
        L=PARAMS["env"]["lock_length"],
    )
    _assert_clears_floor(goal, f"door_code (succ={succ}, fail={fail})")


def test_factored_tiger_goal_rejects_truncating_logits():
    # The regression this pins: (2, 20) at K=6 put 2002 / 53248 entries
    # under the clamp, 91 of them exactly 0.0 in float32.
    with pytest.raises(ValueError, match="safe_log floor"):
        factored_tiger_goal(6, good_logit=2.0, bad_logit=20.0)


# --- The reward -> log-C transformation -------------------------------------

def _expected_delta_log_c(
    log_c: np.ndarray,
    commit: int,
    hold: int,
    component_sizes: tuple[int, ...],
    component: int,
    belief: float,
) -> float:
    """``E_θ[log C(commit) - log C(hold)]`` under a product belief.

    ``commit`` and ``hold`` are two states differing only in the mask bit of
    ``component`` — the door opened / the rock sampled, or not. The belief is
    Bernoulli(``belief``) on that component and uniform over every other one,
    matching the single-component expectation the docstrings state.
    """
    delta = log_c[commit] - log_c[hold]
    n_static = int(np.prod(component_sizes))
    # create_goal flattens θ with np.unravel_index over component_sizes, so
    # read the component back the same way rather than assuming a bit order.
    bits = np.array([
        np.unravel_index(t, component_sizes)[component] for t in range(n_static)
    ])
    weights = np.where(bits == 1, belief, 1.0 - belief)
    return float(delta @ weights / weights.sum())


def _assert_affine_with_ratio_threshold(
    log_c: np.ndarray,
    commit: int,
    hold: int,
    component_sizes: tuple[int, ...],
    good: float,
    bad: float,
) -> None:
    delta = lambda p: _expected_delta_log_c(  # noqa: E731
        log_c, commit, hold, component_sizes, 0, p,
    )
    beliefs = np.linspace(0.0, 1.0, 11)
    values = np.array([delta(p) for p in beliefs])
    # Affine in the belief: exactly the docstring's p·good - (1-p)·bad.
    np.testing.assert_allclose(
        values, beliefs * good - (1 - beliefs) * bad, atol=1e-10,
    )
    # ...hence one zero crossing, at the logit ratio. This is the location
    # the env reward's own threshold has to match for the two objectives to
    # induce the same commit rule.
    threshold = bad / (good + bad)
    assert delta(threshold) == pytest.approx(0.0, abs=1e-10)
    assert delta(threshold - 0.05) < 0.0
    assert delta(threshold + 0.05) > 0.0


@pytest.mark.parametrize("good,bad", [(2.0, 4.0), (1.0, 10.0), (4.0, 2.0)])
def test_factored_tiger_delta_log_c_is_affine_with_ratio_threshold(good, bad):
    """OPEN_0's expected Δ log C: affine in p, zero at bad/(good+bad)."""
    K = 4
    log_c = np.log(factored_tiger_goal(K, good_logit=good, bad_logit=bad))
    n_events = n_events_for(K)
    # Same event index; masks differ only in door 0's bit.
    _assert_affine_with_ratio_threshold(
        log_c, commit=1 * n_events, hold=0 * n_events,
        component_sizes=(2,) * K, good=good, bad=bad,
    )


@pytest.mark.parametrize("good,bad", [(2.0, 4.0), (2.0, 2.0), (4.0, 4.0)])
def test_rock_sample_delta_log_c_is_affine_with_ratio_threshold(good, bad):
    """SAMPLE_0's expected Δ log C: affine in q, zero at bad/(good+bad)."""
    layout = sample_layout(grid_size=5, K=3, seed=0)
    log_c = np.log(rock_sample_goal(layout, good_logit=good, bad_logit=bad))
    K, n_events = layout.K, layout.n_events
    # θ is 2K components: K rock qualities, then K sensor reliabilities, so
    # component 0 is rock 0's quality.
    _assert_affine_with_ratio_threshold(
        log_c,
        commit=encode_state(0, 1, 0, K, n_events),
        hold=encode_state(0, 0, 0, K, n_events),
        component_sizes=(2,) * (2 * K), good=good, bad=bad,
    )


# --- The heatmap row order --------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _fake_summary(value: float, metric_keys: tuple[str, ...], env: dict) -> dict:
    """Runner-shaped JSON whose every per-episode metric is ``value``.

    Small enough to fabricate per cell, and it makes a rendered cell's own
    number identify the cell it was read from.
    """
    arr = [[value, value], [value, value]]
    return {
        "config": {"env": env, "n_seeds": 2, "n_episodes": 2},
        "n_seeds": 2,
        "per_episode": {key: arr for key in metric_keys},
        "derived": {
            "episodes_to_identify": {"median": 1.0, "fraction_identified": 1.0},
        },
    }


def _render(module, monkeypatch, argv: list[str]):
    """Run a plot script's ``main`` and return the figure it drew."""
    import matplotlib.pyplot as plt

    drawn: list = []
    monkeypatch.setattr(plt, "close", drawn.append)  # main() closes the figure
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return drawn[0]


def _assert_rows_match_their_labels(fig) -> None:
    """Row i's cells were written with row i's penalty logit; the y tick
    label on that row has to read back the same number."""
    ax = next(a for a in fig.axes if "return" in a.get_title())
    grid = np.asarray(ax.images[0].get_array(), dtype=float)
    labels = [float(t.get_text()) for t in ax.get_yticklabels()]
    assert len(labels) == grid.shape[0]
    for i, label in enumerate(labels):
        assert grid[i, 0] == pytest.approx(label), (
            f"row {i} is labelled {label:g} but holds cells from "
            f"{grid[i, 0]:g}"
        )
    # House convention, both heatmaps: penalty magnitude grows downwards.
    magnitudes = [abs(v) for v in labels]
    assert magnitudes == sorted(magnitudes)


def test_factored_tiger_heatmap_rows_match_their_bad_logit_labels(
    tmp_path, monkeypatch,
):
    import plot_factored_tiger_preference as tiger_plot

    keys = ("reward_normalised", "n_listens", "n_tiger_opens",
            "posterior_mass_on_persistent_theta")
    env = {"K": 4, "listen_accuracy": 0.85, "open_safe_reward": 10,
           "open_tiger_penalty": 100, "listen_cost": 0.0}
    sweep = PARAMS["factored_tiger_preference"]
    cells = tmp_path / "cells"
    for good, bad in itertools.product(sweep["good_logits"], sweep["bad_logits"]):
        cell = cells / f"good{good}_bad{bad}"
        cell.mkdir(parents=True)
        (cell / "aif.json").write_text(
            json.dumps(_fake_summary(float(bad), keys, env)),
        )
    references = tmp_path / "references"
    references.mkdir()
    for planner in tiger_plot._REFERENCE_PLANNERS:
        (references / f"{planner}.json").write_text(
            json.dumps(_fake_summary(0.0, keys, env)),
        )
    fig = _render(tiger_plot, monkeypatch, [
        "plot_factored_tiger_preference.py",
        "--input-dir", str(cells),
        "--reference-dir", str(references),
        "--output", str(tmp_path / "figure.pdf"),
    ])
    _assert_rows_match_their_labels(fig)


def test_door_code_heatmap_rows_match_their_failure_logit_labels(
    tmp_path, monkeypatch,
):
    import plot_door_code_preference as door_code_plot

    keys = ("terminal_reward", "probe_visits", "posterior_mass_on_true_theta")
    env = {"K": 6, "lock_length": 4, "probe_noise": 0.2}
    sweep = PARAMS["door_code_preference"]
    cells = tmp_path / "cells"
    for succ, fail in itertools.product(
        sweep["success_logits"], sweep["failure_logits"],
    ):
        cell = cells / f"succ{succ}_fail{fail}"
        cell.mkdir(parents=True)
        (cell / "aif.json").write_text(
            json.dumps(_fake_summary(float(fail), keys, env)),
        )
    references = tmp_path / "references"
    references.mkdir()
    for planner in door_code_plot._REFERENCE_PLANNERS:
        (references / f"{planner}.json").write_text(
            json.dumps(_fake_summary(0.0, keys, env)),
        )
    fig = _render(door_code_plot, monkeypatch, [
        "plot_door_code_preference.py",
        "--input-dir", str(cells),
        "--reference-dir", str(references),
        "--output", str(tmp_path / "figure.pdf"),
    ])
    _assert_rows_match_their_labels(fig)
