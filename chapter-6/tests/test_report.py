"""Window metrics, censored identification, and the table emitter.

Three groups:

  * `learningpomdp/experiment/report.py` window metrics on synthetic
    arrays with hand-computed answers, including the NaN handling that
    `reward_normalised` forces.
  * the right-censored identification fields added to
    `episodes_to_identify` — edge cases mirroring
    `tests/test_derived_metrics.py` (all cross / none cross / mixed /
    threshold respected), plus the guarantee that no pre-existing field
    changed.
  * `scripts/make_tables.py`'s formatters, then the emitter end-to-end on
    a synthetic `output/` tree.
  * a golden check that the emitted numbers still match the values
    published in the paper. This is the drift detector: the published
    tables were hand-typed, so nothing else in the repo would notice them
    going stale. It runs against the committed slices under
    `tests/golden/` always, and additionally against the canonical
    `output/` wherever that exists — see `GOLDEN_SOURCES`.
"""

import functools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from learningpomdp.experiment.door_code_runner import episodes_to_identify
from learningpomdp.experiment.report import (
    final_episode_mean_ci,
    mean_ci,
    paired_window_difference,
    per_decision_planning_time,
    theta_pairing,
    window_mean_ci,
    window_overlap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# scripts/ is not a package; put it on the path so the golden test can
# call the emitter's row builder directly instead of parsing LaTeX back.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import make_tables  # noqa: E402


# ---------------------------------------------------------------- window metrics


def test_mean_ci_hand_computed() -> None:
    # mean 3, sample std 1 over 5 points -> SEM = 1/sqrt(5).
    mean, ci, n = mean_ci([2.0, 2.0, 3.0, 4.0, 4.0])
    assert mean == pytest.approx(3.0)
    assert ci == pytest.approx(1.96 * 1.0 / math.sqrt(5))
    assert n == 5


def test_mean_ci_single_finite_value_has_no_width() -> None:
    mean, ci, n = mean_ci([7.0, np.nan, np.nan])
    assert mean == pytest.approx(7.0)
    assert ci == 0.0
    assert n == 1


def test_mean_ci_all_nan_is_nan() -> None:
    mean, ci, n = mean_ci([np.nan, np.nan])
    assert math.isnan(mean) and math.isnan(ci)
    assert n == 0


def test_window_mean_ci_picks_the_right_episodes() -> None:
    # 3 seeds x 4 episodes. Early window k=2 -> per-seed means
    # (0, 10, 20); final window k=2 -> (1, 11, 21).
    arr = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [10.0, 10.0, 11.0, 11.0],
        [20.0, 20.0, 21.0, 21.0],
    ])
    early, early_ci, early_n = window_mean_ci(arr, 2, "early")
    final, final_ci, final_n = window_mean_ci(arr, 2, "final")
    assert early == pytest.approx(10.0)
    assert final == pytest.approx(11.0)
    # Same spread in both windows: std([0,10,20], ddof=1) = 10.
    expected = 1.96 * 10.0 / math.sqrt(3)
    assert early_ci == pytest.approx(expected)
    assert final_ci == pytest.approx(expected)
    assert early_n == final_n == 3


def test_window_mean_ci_is_per_seed_not_pooled() -> None:
    # Two blocks, flat at 0 and at 2. Both conventions give mean 1, but
    # pooling the four cells as if they were independent shrinks the
    # interval (n = 4, sample std 1.1547 -> 1.13) relative to aggregating
    # the two block-level means (n = 2, sample std sqrt(2) -> 1.96).
    arr = np.array([[0.0, 0.0], [2.0, 2.0]])
    mean, ci, n = window_mean_ci(arr, 2, "early")
    assert mean == pytest.approx(1.0)
    assert ci == pytest.approx(1.96 * math.sqrt(2) / math.sqrt(2))
    assert n == 2
    pooled = mean_ci(arr.reshape(-1))
    assert pooled[0] == pytest.approx(1.0)
    assert pooled[1] < ci
    assert pooled[2] == 4


def test_window_mean_ci_skips_nan_cells_but_keeps_the_block() -> None:
    # reward_normalised is NaN when no positive reward was obtainable.
    # Seed 0's window is [4, NaN] -> block mean 4; seed 1 is all NaN and
    # drops out entirely, leaving a single finite block. The reported n is
    # 1, not the 2 blocks in the run: a caption quoting the header count
    # next to this interval would be quoting the wrong n.
    arr = np.array([[4.0, np.nan], [np.nan, np.nan]])
    mean, ci, n = window_mean_ci(arr, 2, "early")
    assert mean == pytest.approx(4.0)
    assert ci == 0.0
    assert n == 1


def test_window_mean_ci_rejects_bad_side() -> None:
    with pytest.raises(ValueError):
        window_mean_ci(np.zeros((2, 2)), 1, "middle")


@pytest.mark.parametrize("side", ["early", "final"])
@pytest.mark.parametrize("k", [0, -1])
def test_window_rejects_non_positive_lengths(side: str, k: int) -> None:
    # `arr[:, -0:]` is the whole array, so a k of 0 on the final side used
    # to widen the window to the entire run instead of failing.
    with pytest.raises(ValueError, match="positive"):
        window_mean_ci(np.zeros((2, 4)), k, side)


@pytest.mark.parametrize("side", ["early", "final"])
def test_window_rejects_a_length_past_the_end(side: str) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        window_mean_ci(np.zeros((2, 4)), 5, side)


def test_window_overlap_counts_shared_episodes() -> None:
    assert window_overlap(25, 5, 5) == 0     # door_code: disjoint
    assert window_overlap(10, 5, 3) == 0     # minigrid / rock_sample
    assert window_overlap(5, 2, 2) == 0      # factored_tiger, corrected
    assert window_overlap(8, 5, 3) == 0      # exactly adjacent
    assert window_overlap(5, 3, 3) == 1      # factored_tiger, before (3, 3)
    assert window_overlap(4, 4, 4) == 4      # windows fully coincide


def test_window_overlap_rejects_non_positive_lengths() -> None:
    with pytest.raises(ValueError, match="positive"):
        window_overlap(10, 0, 3)


def test_final_episode_mean_ci_uses_the_last_column() -> None:
    arr = np.array([[0.0, 0.5], [0.0, 0.7]])
    mean, _, n = final_episode_mean_ci(arr)
    assert mean == pytest.approx(0.6)
    assert n == 2


def test_per_decision_planning_time_divides_by_steps() -> None:
    # planning_time_s is an episode total: 10 s over 5 steps and 20 s
    # over 5 steps -> 30/10 = 3 s per decision.
    value = per_decision_planning_time([[10.0, 20.0]], [[5, 5]])
    assert value == pytest.approx(3.0)


def test_per_decision_planning_time_ignores_zero_step_episodes() -> None:
    value = per_decision_planning_time([[10.0, 99.0]], [[5, 0]])
    assert value == pytest.approx(2.0)


# ---------------------------------------------------------------- pairing


def test_theta_pairing_identical_sequences() -> None:
    out = theta_pairing([1, 2, 3], [1, 2, 3])
    assert out == {"n_compared": 3, "n_shared_theta": 3, "paired": True}


def test_theta_pairing_partial_overlap_is_unpaired() -> None:
    out = theta_pairing([1, 2, 3], [1, 9, 3])
    assert out["n_shared_theta"] == 2
    assert out["paired"] is False


def test_paired_window_difference_hand_computed() -> None:
    a = np.array([[3.0, 3.0], [5.0, 5.0]])
    b = np.array([[1.0, 1.0], [2.0, 2.0]])
    mean, ci, n = paired_window_difference(a, b, 2, "early")
    # Per-seed differences 2 and 3 -> mean 2.5, sample std 1/sqrt(2).
    assert mean == pytest.approx(2.5)
    assert ci == pytest.approx(1.96 * (1.0 / math.sqrt(2)) / math.sqrt(2))
    assert n == 2


def test_paired_window_difference_rejects_mismatched_seed_counts() -> None:
    with pytest.raises(ValueError):
        paired_window_difference(np.zeros((2, 2)), np.zeros((3, 2)), 1, "early")


# ------------------------------------------------- censored identification


def test_censored_fields_when_every_seed_crosses() -> None:
    out = episodes_to_identify(np.ones((4, 5)))
    assert out["fraction_identified_by_episode"] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert out["censored_median"] == 0.0
    assert out["censored_p25"] == 0.0
    assert out["censored_p75"] == 0.0


def test_censored_fields_when_no_seed_crosses() -> None:
    out = episodes_to_identify(np.full((3, 5), 0.5))
    assert out["fraction_identified_by_episode"] == [0.0] * 5
    # Every quantile lies beyond the budget, so all three are censored.
    assert out["censored_median"] is None
    assert out["censored_p25"] is None
    assert out["censored_p75"] is None


def test_censored_median_exceeds_the_conditional_median_when_mixed() -> None:
    # 4 seeds: crossings at 0, 1 and never (x2). Conditional median over
    # the two crossers is 0.5; only 50% of all blocks ever cross, so the
    # censored median is 1 and the censored p75 is beyond the budget.
    posterior = np.array([
        [0.95, 0.95, 0.95],
        [0.10, 0.95, 0.95],
        [0.10, 0.10, 0.10],
        [0.10, 0.10, 0.10],
    ])
    out = episodes_to_identify(posterior)
    assert out["median"] == 0.5
    assert out["fraction_identified"] == 0.5
    assert out["fraction_identified_by_episode"] == [0.25, 0.5, 0.5]
    assert out["censored_p25"] == 0.0
    assert out["censored_median"] == 1.0
    assert out["censored_p75"] is None


def test_censored_fields_respect_the_threshold() -> None:
    posterior = np.array([[0.5, 0.7, 0.85]])
    out = episodes_to_identify(posterior, threshold=0.8)
    assert out["fraction_identified_by_episode"] == [0.0, 0.0, 1.0]
    assert out["censored_median"] == 2.0
    out_high = episodes_to_identify(posterior, threshold=0.9)
    assert out_high["fraction_identified_by_episode"] == [0.0, 0.0, 0.0]
    assert out_high["censored_median"] is None


def test_existing_identification_fields_are_unchanged() -> None:
    # The figures and the ablation plots read these; adding the censored
    # fields must not perturb them.
    posterior = np.array([
        [0.5, 0.95, 0.95, 0.95],
        [0.5, 0.5, 0.5, 0.95],
        [0.1, 0.1, 0.1, 0.1],
    ])
    out = episodes_to_identify(posterior)
    assert out["per_seed"] == [1, 3, None]
    assert out["median"] == 2.0
    assert out["p25"] == 1.5
    assert out["p75"] == 2.5
    assert out["fraction_identified"] == pytest.approx(2 / 3)
    assert out["n_episodes_budget"] == 4
    assert out["threshold"] == 0.9


def test_survival_curve_is_monotone_and_ends_at_the_crossing_fraction() -> None:
    rng = np.random.default_rng(0)
    posterior = rng.uniform(0.0, 1.0, size=(20, 8))
    out = episodes_to_identify(posterior)
    curve = out["fraction_identified_by_episode"]
    assert len(curve) == 8
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert curve[-1] == pytest.approx(out["fraction_identified"])


# ---------------------------------------------------------------- formatters


def test_fmt_pm_prints_no_negative_zero() -> None:
    # AIF-MP minus Oracle on the Multi-Door Tiger final window is about
    # -0.0034, which plain %.2f renders "-0.00": a signed difference whose
    # rounded value is exactly zero.
    assert make_tables.fmt_pm(-0.0034, 0.0067) == "$0.00 \\pm 0.01$"
    # A value that genuinely rounds away from zero keeps its sign.
    assert make_tables.fmt_pm(-0.006, 0.01) == "$-0.01 \\pm 0.01$"
    assert make_tables.fmt_frac(-0.001) == "$0.00$"


def test_fmt_censored_median_marks_quantiles_beyond_the_budget() -> None:
    # None means the survival curve never reached that quantile inside the
    # budget, which must print as a bound rather than be dropped or shown
    # as the budget itself.
    text = make_tables.fmt_censored_median({
        "n_episodes_budget": 5,
        "censored_p25": 1.0,
        "censored_median": None,
        "censored_p75": None,
    })
    assert text == "${>}5$ [$2.0$, ${>}5$]"


def test_tt_escapes_every_latex_special_it_can_meet() -> None:
    # A bare % comments out the rest of the line, so an unescaped one in a
    # caption silently truncates it.
    assert make_tables.tt("a_b%c&d#e$f") == (
        "\\texttt{a\\_b\\%c\\&d\\#e\\$f}"
    )


def test_best_bolds_the_maximum() -> None:
    assert make_tables._best({"a": 0.8, "b": 0.79, "c": 0.1}) == {"a"}
    # Ties at display precision are all marked.
    assert make_tables._best({"a": 0.800, "b": 0.804, "c": 0.1}) == {"a", "b"}
    # Non-finite entries never win and never block a winner.
    assert make_tables._best({"a": float("nan"), "b": 0.5}) == {"b"}
    assert make_tables._best({"a": float("nan")}) == set()


def test_best_marks_nothing_when_every_competitor_ties() -> None:
    # The concrete case is a benchmark where nothing identifies: bolding
    # every 0.00 would assert a best method the data does not contain.
    assert make_tables._best({"a": 0.0, "b": 0.0, "c": 0.0}) == set()
    assert make_tables._best({"a": 0.42, "b": 0.42}) == set()


# ---------------------------------------------------------------- emitter


def _synthetic_run(
    n_seeds: int, n_episodes: int, metric: str, mass_key: str, value: float,
    nan_blocks: int = 0,
) -> dict:
    # `nan_blocks` blocks are entirely undefined in the reported metric,
    # as reward_normalised is on a theta whose achievable reward is zero.
    # json.dumps writes these as the bare `NaN` token, which is what the
    # canonical runners write too.
    metric_rows = [
        [float("nan")] * n_episodes if i < nan_blocks else [value] * n_episodes
        for i in range(n_seeds)
    ]
    per_episode = {
        metric: metric_rows,
        mass_key: [[0.95] * n_episodes for _ in range(n_seeds)],
        "planning_time_s": [[2.0] * n_episodes for _ in range(n_seeds)],
        "n_steps": [[4] * n_episodes for _ in range(n_seeds)],
    }
    return {
        "config": {"n_seeds": n_seeds},
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "theta_trues": list(range(n_seeds)),
        "per_episode": per_episode,
        "curves": {},
        "derived": {},
    }


def _write_synthetic_output(root: Path) -> None:
    specs = [
        ("door_code", "cumulative_reward", "posterior_mass_on_true_theta", 25),
        ("minigrid_door_code", "cumulative_reward",
         "posterior_mass_on_true_theta", 10),
        ("factored_tiger", "reward_normalised",
         "posterior_mass_on_persistent_theta", 5),
        ("rock_sample", "reward_normalised",
         "posterior_mass_on_persistent_theta", 10),
    ]
    for i, (dirname, metric, mass_key, n_episodes) in enumerate(specs):
        env_dir = root / dirname
        env_dir.mkdir(parents=True)
        for j, planner in enumerate(
            ["aif", "psrl", "bamcp", "oracle", "random", "vbp"]
        ):
            # VBP is deliberately absent from rock_sample, so the timing
            # table has a row mixing real cells with an absent-result cell.
            # Some factored_tiger blocks are undefined throughout, so the
            # per-column block counts diverge from the header count in both
            # the main table (PSRL) and the deferred one (VBP). Oracle is
            # given undefined blocks too: its identification cells are
            # suppressed, so it is the row that must name fewer columns in
            # the effective-n clause than every other row does.
            if (dirname, planner) == ("rock_sample", "vbp"):
                continue
            nan_blocks = {"psrl": 2, "vbp": 1, "oracle": 3}.get(planner, 0)
            run = _synthetic_run(
                6, n_episodes, metric, mass_key, 0.1 * (i + 1) + 0.01 * j,
                nan_blocks=nan_blocks if dirname == "factored_tiger" else 0,
            )
            (env_dir / f"{planner}.json").write_text(json.dumps(run))


def _run_emitter(input_dir: Path, output_dir: Path) -> dict[str, str]:
    subprocess.run(
        ["uv", "run", "python", "scripts/make_tables.py",
         "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
        check=True, capture_output=True, cwd=REPO_ROOT,
    )
    return {p.name: p.read_text() for p in sorted(output_dir.glob("*.tex"))}


@pytest.fixture(scope="module")
def synthetic_tables(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """One emitter run over the synthetic tree, shared by the tests below."""
    root = tmp_path_factory.mktemp("emitter")
    _write_synthetic_output(root / "output")
    return _run_emitter(root / "output", root / "tables")


def test_emitter_runs_and_skips_absent_planners(
    synthetic_tables: dict[str, str],
) -> None:
    tables = synthetic_tables

    assert set(tables) >= {
        "scalars_door_code.tex", "scalars_minigrid_door_code.tex",
        "scalars_factored_tiger.tex", "scalars_rock_sample.tex",
        "timing.tex", "deferred_baselines.tex", "pairing.tex",
    }
    # No simscaling sweep on disk -> the table is skipped, not faked.
    assert "simscaling.tex" not in tables
    for name, text in tables.items():
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}")
        assert r"\toprule" in text and r"\bottomrule" in text
    # Absent planners (BOSS, IDS, ...) must not appear as rows.
    assert "BOSS" not in tables["scalars_door_code.tex"]
    # Oracle is emitted, as a reference bound (reviewer item 5).
    assert "Oracle" in tables["scalars_door_code.tex"]
    assert "Reference bounds" in tables["scalars_door_code.tex"]
    # planning_time_s / n_steps = 2.0 / 4 = 0.50 s per decision.
    assert "0.50" in tables["timing.tex"]


def test_emitter_fails_loudly_on_an_empty_output_tree(tmp_path: Path) -> None:
    (tmp_path / "output").mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        _run_emitter(tmp_path / "output", tmp_path / "tables")


def test_no_caption_discloses_an_overlap_under_the_current_protocol(
    synthetic_tables: dict[str, str],
) -> None:
    # `report.windows` is disjoint on all four benchmarks, factored_tiger
    # included since its (3, 3) of 5 episodes became (2, 2). No caption
    # may claim otherwise, and the emitter must not have warned.
    for name, text in synthetic_tables.items():
        assert "disjoint" not in text, name


def test_caption_discloses_an_overlapping_window_pair(tmp_path: Path) -> None:
    # The guard `_overlap_clause` exists for: an n_episodes cut that makes
    # a configured window pair overlap again. Truncating factored_tiger to
    # 3 episodes puts (2, 2) back in that state -- episode 2 lands in both
    # windows -- and the emitter must say so rather than resize the
    # configured protocol behind the reader's back. Written here rather
    # than into the shared tree so that tree keeps every benchmark's real
    # episode count.
    root = tmp_path / "output"
    _write_synthetic_output(root)
    for path in (root / "factored_tiger").glob("*.json"):
        run = json.loads(path.read_text())
        run["n_episodes"] = 3
        run["per_episode"] = {
            key: [row[:3] for row in rows]
            for key, rows in run["per_episode"].items()
        }
        path.write_text(json.dumps(run))
    tables = _run_emitter(root, tmp_path / "tables")

    tiger = tables["scalars_factored_tiger.tex"]
    assert r"\emph{not} disjoint" in tiger
    assert "share 1 of the 3 episodes (episode 2" in tiger
    # ... and only there: the untouched benchmarks stay disjoint.
    assert "disjoint" not in tables["scalars_door_code.tex"]


# ------------------------------------------------- exact Bayes-adaptive reference


def _write_exact_reference(
    root: Path,
    dirname: str,
    *,
    sweep_rates: tuple[float, float, float],
    mean_regret: float,
) -> None:
    """One reference.json, in the schema run_exact_reference.py emits."""
    tie_breaks = ["exact_lowest_index", "exact_highest_index"]
    payload = {
        "config": {
            "env_kwargs": {"K": 3},
            "n_seeds": 25,
            "n_episodes": 25,
            "max_steps": 6,
            "exact_tie_breaks": tie_breaks,
        },
        "action_agreement": {
            "n_decisions": 1386,
            "optimal_set_rate": sweep_rates[0],
            "lowest_index_match_rate": 0.55,
            "mean_optimal_set_size": 2.34,
            "mean_value_regret": mean_regret,
            "max_value_regret": 11.474998,
            "tie_tolerance_sweep": {
                tol: {"optimal_set_rate": rate}
                for tol, rate in zip(
                    ("1e-06", "1e-07", "1e-08"), sweep_rates
                )
            },
        },
        "block": {
            # Highest-index first, so the band cannot pass by taking the
            # arms in the order they happen to appear.
            "normalised_return": {
                "exact_highest_index": 0.9456, "exact_lowest_index": 0.96,
                "aif": 0.888, "bamcp": 0.6112,
            },
            "normalised_return_sem": {
                "exact_highest_index": 0.0115, "exact_lowest_index": 0.0083,
                "aif": 0.0794, "bamcp": 0.1428,
            },
        },
    }
    path = root / "exact_reference" / dirname / "reference.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_exact_reference_tables_are_skipped_when_the_stages_have_not_run(
    synthetic_tables: dict[str, str],
) -> None:
    # Both stages are deferred from the cluster slate, so an output/ tree
    # without them is normal: the tables must be absent, not empty.
    assert "exact_agreement.tex" not in synthetic_tables
    assert "exact_block.tex" not in synthetic_tables


def test_exact_reference_tables_report_the_reference_numbers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    _write_synthetic_output(root)
    # Door-Code's rate is flat across the sweep; Multi-Door Tiger's is not.
    _write_exact_reference(
        root, "door_code", sweep_rates=(0.9618, 0.9618, 0.9589),
        mean_regret=0.0030452,
    )
    _write_exact_reference(
        root, "factored_tiger", sweep_rates=(0.9427, 0.9427, 0.884),
        mean_regret=0.2382673,
    )
    tables = _run_emitter(root, tmp_path / "tables")

    agreement = tables["exact_agreement.tex"]
    assert r"Door-Code & $1386$ & $0.96$ & $0.55$ & $2.34$" in agreement
    # Two decimals would print Door-Code's mean regret as an exact 0.00.
    assert "$0.00305$" in agreement
    assert "$0.238$" in agreement
    # Three significant figures, not three decimals.
    assert "$11.5$" in agreement
    # A flat sweep collapses to one value; a moving one keeps its range.
    assert "Door-Code $0.96$;" in agreement
    assert "Multi-Door Tiger $0.88$--$0.96$" not in agreement
    assert "Multi-Door Tiger $0.88$--$0.94$" in agreement
    # The reduced instance is disclosed, since the row label is the
    # headline benchmark's name.
    assert r"25$\times$25 blocks$\times$episodes" in agreement

    block = tables["exact_block.tex"]
    # Band = what the two tie-breaks bracket, lowest value first whatever
    # order the arms appear in; AIF-MP at 1.96*SEM = 0.0794*1.96 = 0.156.
    assert r"$[0.95, 0.96]$ & $0.89 \pm 0.16$ & $0.61 \pm 0.28$" in block
    # The band must not be read as an optimality ratio.
    assert "not an optimality ratio" in block
    # Episode-0 is the underpowered slice and must not be quietly printed.
    assert "Episode-0 return" in block and "standard errors exceed" in block


def test_caption_reports_the_block_count_behind_each_interval(
    synthetic_tables: dict[str, str],
) -> None:
    # Two of the six synthetic factored_tiger PSRL blocks are undefined
    # throughout, so its early and final intervals are over 4 blocks while
    # its final-mass interval is over all 6 -- different n's under one
    # caption, which is the bug this clause closes. The full-block column
    # is not named: only the intervals that deviate are.
    tiger = synthetic_tables["scalars_factored_tiger.tex"]
    assert "PSRL (early $4$, final $4$, of 6)" in tiger
    # ... and the clause must name only rows the table actually prints.
    assert "VBP" not in tiger.split(r"\caption")[1]
    # Every block is defined on door_code, so the clause stays out.
    assert "fewer blocks than the run holds" not in (
        synthetic_tables["scalars_door_code.tex"]
    )
    # VBP's own table is where its factored_tiger counts belong.
    assert "Multi-Door Tiger VBP (early $5$, final $5$, of 6)" in (
        synthetic_tables["deferred_baselines.tex"]
    )
    # Oracle reaches the clause too (three undefined blocks), and its
    # suppressed final-mass cell must not get a block count claimed for it.
    assert "Oracle (early $3$, final $3$, of 6)" in tiger


def test_effective_n_clause_names_only_the_columns_the_row_prints() -> None:
    row = {"early_n": 4, "final_n": 4, "mass_n": 3, "n_seeds": 6}
    full = make_tables._effective_n_clause(
        [("PSRL", row, ("early_n", "final_n", "mass_n"))]
    )
    assert "PSRL (early $4$, final $4$, final mass $3$, of 6)" in full
    # An arm whose final-mass cell is suppressed must not carry a
    # final-mass block count: that would name an interval the table does
    # not print. Positional counts could not express the difference.
    suppressed = make_tables._effective_n_clause(
        [("Oracle", row, ("early_n", "final_n"))]
    )
    assert "Oracle (early $4$, final $4$, of 6)" in suppressed
    assert "final mass" not in suppressed
    # Nothing below the header count -> no clause at all.
    assert make_tables._effective_n_clause([(
        "AIF-MP", {"early_n": 6, "final_n": 6, "mass_n": 6, "n_seeds": 6},
        ("early_n", "final_n", "mass_n"),
    )]) == ""


def test_oracle_identification_cells_are_omitted_and_explained(
    synthetic_tables: dict[str, str],
) -> None:
    # 5e1605c dropped Oracle from the figures' identification panels
    # (mass, Brier, episodes-to-identify) because it plans on a one-hot at
    # theta-star. The table's three identification columns are those same
    # quantities, so they go too -- otherwise a table and a figure
    # disagree about whether the quantity applies to Oracle at all.
    tiger = synthetic_tables["scalars_factored_tiger.tex"]
    rows = {
        line.split("&")[0].strip(): line
        for line in tiger.splitlines() if "&" in line
    }
    assert rows["Oracle"].count(make_tables.NOT_APPLICABLE_CELL) == 3
    # Both reward columns stay: Oracle is still the reward bound.
    assert make_tables.NOT_APPLICABLE_CELL not in (
        rows["Oracle"].split("&")[1] + rows["Oracle"].split("&")[2]
    )
    # Random keeps its identification cells (passive-identification floor).
    assert make_tables.NOT_APPLICABLE_CELL not in rows["Random"]
    assert make_tables.NOT_APPLICABLE_CELL not in rows["AIF-MP"]
    # ... and the omission is explained where the reader meets it.
    caption = tiger.split(r"\caption")[1]
    assert "identification columns are omitted for Oracle" in caption
    assert "not an identification baseline" in caption
    assert "Random is kept in both" in caption


def test_timing_separates_an_absent_result_from_a_non_finite_one(
    synthetic_tables: dict[str, str],
) -> None:
    # VBP has no rock_sample JSON in the synthetic tree, which is not the
    # same claim as "not implemented there" -- the emitter cannot see which
    # it is, so it must not print a marker whose caption says the latter.
    timing = synthetic_tables["timing.tex"]
    vbp = next(line for line in timing.splitlines() if line.startswith("VBP"))
    assert vbp.count(r"\emph{n/r}") == 1
    assert "no result JSON" in timing
    assert "not implemented for that benchmark" not in timing


def test_pairing_prose_follows_the_measured_overlaps(
    synthetic_tables: dict[str, str],
) -> None:
    # Every synthetic run shares theta_trues, so the paragraph must report
    # pairing rather than assert the canonical runs' unpaired verdict.
    pairing = synthetic_tables["pairing.tex"]
    assert r"\textbf{are paired across methods}" in pairing
    assert r"\textbf{not paired across" not in pairing
    assert "unpaired" not in pairing.split(r"\begin{tabular}")[1]


def test_pairing_prose_reports_a_mixed_verdict(tmp_path: Path) -> None:
    root = tmp_path / "output"
    _write_synthetic_output(root)
    path = root / "door_code" / "psrl.json"
    run = json.loads(path.read_text())
    run["theta_trues"] = [99, *run["theta_trues"][1:]]  # one index diverges
    path.write_text(json.dumps(run))
    pairing = _run_emitter(root, tmp_path / "tables")["pairing.tex"]

    assert "comparisons below share" in pairing
    assert r"\textbf{holds only within}" in pairing
    assert r"\textbf{are paired across methods}" not in pairing
    assert "unpaired" in pairing.split(r"\begin{tabular}")[1]


# The paper's Tables 1-4 and 8, keyed by benchmark -> planner -> (early,
# early CI, final, final CI, final mass, mass CI, crossing fraction). The
# identification *median* is deliberately absent: this PR replaces the
# conditional median with a right-censored one, so it is expected to
# differ and the change is documented in the PR body.
#
# PROVENANCE. Every cell here was transcribed from the printed
# sections/{evaluation,appendix}.tex by this repo's commit c0d2b65 ("Add
# tests/test_report.py, including the drift detector"), against the
# `output/` tree recorded in dvc.lock at that commit -- EXCEPT the
# fourteen Multi-Door Tiger early/final cells (seven arms times two
# columns), which this commit recomputed (see (a) below). That pin is
# what makes a red test here
# readable: it says the paper's numbers and the results have parted
# company, and the transcription commit tells you which side moved.
#
# A cell can go stale for two unrelated reasons, and a red test has to be
# read differently in each case:
#
#   (a) the REPORTING PROTOCOL was corrected, so the number the emitter
#       computes is right and the paper's is wrong. The results on disk
#       did not move. The only instance is Multi-Door Tiger's early and
#       final columns, marked below: the windows were (3, 3) of 5
#       episodes and so shared episode 3 between them, and this commit
#       corrected them to a disjoint (2, 2) in params.yaml
#       `report.windows`. Those numbers are therefore the recomputed
#       ones and do not match the printed Table 3 until the paper is
#       edited to match -- so this dict is deliberately AHEAD of the
#       paper for those cells, and green here is not evidence that
#       Table 3 is current. No rerun is involved; `dvc repro tables`
#       re-emits from the same JSONs. Nothing else moved, Multi-Door
#       Tiger's own mass and crossing-fraction cells included: the
#       window change cannot reach a final-episode or whole-run
#       quantity.
#
#   (b) the RESULTS moved, so the paper is stale against a rerun. PR
#       #13's per-outer-seed planner RNG fix reruns the tree-search arms
#       and is expected to turn several of these red that way. The fix
#       is then to regenerate tests/golden/slice_*.json, update these
#       literals and the paper together, and name the producing run in
#       the commit message.
#
# Tell them apart by whether the run JSONs moved: (a) leaves
# tests/golden/slice_*.json byte-identical, (b) does not.
PUBLISHED_SCALARS = {
    "door_code": {
        "aif": (0.47, 0.09, 0.80, 0.12, 0.87, 0.06, 0.88),
        "psrl": (-0.94, 0.04, 0.80, 0.11, 0.48, 0.02, 0.00),
        "bamcp": (0.00, 0.00, 0.00, 0.00, 0.99, 0.01, 0.99),
        "boss": (0.00, 0.00, 0.00, 0.00, 0.96, 0.03, 0.93),
        "ids": (0.00, 0.00, 0.00, 0.00, 0.99, 0.01, 0.99),
        "vbp": (-0.16, 0.05, -0.48, 0.13, 0.16, 0.06, 0.09),
        "belief_q": (-1.00, 0.00, -0.64, 0.02, 0.15, 0.02, 0.00),
        "pomcp": (0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 1.00),
    },
    "minigrid_door_code": {
        "aif": (0.36, 0.07, 0.42, 0.08, 1.00, 0.00, 1.00),
        "psrl": (-0.05, 0.12, 0.51, 0.12, 0.92, 0.05, 0.92),
        "bamcp": (0.00, 0.00, 0.00, 0.00, 0.43, 0.06, 0.16),
        "boss": (0.00, 0.00, 0.00, 0.00, 0.45, 0.06, 0.16),
        "ids": (0.00, 0.00, 0.00, 0.00, 0.44, 0.06, 0.16),
        "vbp": (-0.66, 0.13, -0.54, 0.16, 0.37, 0.07, 0.25),
        "belief_q": (-0.11, 0.03, -0.05, 0.03, 0.59, 0.06, 0.32),
        "pomcp": (0.00, 0.00, 0.00, 0.00, 0.50, 0.06, 0.22),
    },
    # The early and final pairs here are case (a): recomputed under the
    # corrected disjoint (2, 2) windows, ahead of the printed Table 3.
    # The (3, 3) values they replace, for the paper diff, were
    # aif (0.72, 0.17, 1.00, 0.01), psrl (-3.83, 0.89, -0.06, 0.35),
    # bamcp/ids (0.21, 0.28, 0.97, 0.04), boss (-3.73, 0.82, 0.91, 0.07),
    # factored_bamcp (0.21, 0.19, 0.98, 0.02), vbp (-0.02, 0.23, 0.34,
    # 0.08). The mass and crossing-fraction cells are unchanged.
    "factored_tiger": {
        "aif": (0.59, 0.26, 1.00, 0.01, 0.97, 0.02, 0.99),
        "psrl": (-5.42, 1.25, 0.24, 0.33, 0.94, 0.03, 0.88),
        "bamcp": (-0.16, 0.42, 0.98, 0.03, 0.99, 0.01, 0.99),
        "boss": (-6.10, 1.23, 0.87, 0.11, 1.00, 0.00, 1.00),
        "ids": (-0.16, 0.42, 0.98, 0.03, 0.99, 0.01, 0.99),
        "factored_bamcp": (-0.16, 0.29, 1.00, 0.00, 1.00, 0.00, 1.00),
        "vbp": (-0.21, 0.34, 0.33, 0.10, 0.20, 0.06, 0.10),
    },
    "rock_sample": {
        "aif": (0.55, 0.08, 0.63, 0.10, 1.00, 0.00, 1.00),
        "psrl": (0.11, 0.07, 0.10, 0.07, 0.13, 0.01, 0.00),
        "bamcp": (0.42, 0.08, 0.30, 0.12, 0.62, 0.08, 0.20),
        "boss": (-0.31, 0.15, -0.28, 0.17, 0.18, 0.03, 0.00),
        "ids": (0.43, 0.08, 0.41, 0.10, 0.67, 0.08, 0.32),
        "factored_bamcp": (0.38, 0.08, 0.42, 0.11, 0.57, 0.09, 0.20),
        "vbp": (0.08, 0.07, 0.15, 0.11, 0.15, 0.02, 0.00),
    },
}

# The published cells are rounded to two decimals, so a value can sit up
# to half a display unit from its printed form. Real drift is an order of
# magnitude larger than this.
_DISPLAY_TOL = 6e-3

# Table 7, seconds per decision. Same provenance and same staleness
# reading as PUBLISHED_SCALARS above, except that case (a) cannot reach
# these: a per-decision time sums over every decision of the run, so no
# early/final window enters it. Every cell here is still the transcribed
# published value, so a red test in this table is case (b).
PUBLISHED_TIMING = {
    "door_code": {
        "aif": 0.04, "vbp": 0.05, "psrl": 0.06, "oracle": 0.06,
        "bamcp": 1.77, "pomcp": 1.45, "boss": 2.12, "ids": 1.85,
    },
    "minigrid_door_code": {
        "aif": 0.22, "vbp": 0.36, "psrl": 0.36, "oracle": 0.34,
        "bamcp": 4.17, "pomcp": 3.92, "boss": 4.62, "ids": 4.22,
    },
    "factored_tiger": {
        "aif": 0.39, "vbp": 0.25, "psrl": 0.25, "oracle": 0.25,
        "bamcp": 0.63, "boss": 0.86, "factored_bamcp": 0.63, "ids": 25.6,
    },
    "rock_sample": {
        "aif": 2.90, "vbp": 1.77, "psrl": 1.77, "oracle": 1.77,
        "bamcp": 0.42, "boss": 0.79, "factored_bamcp": 0.42, "ids": 46.6,
    },
}


GOLDEN_DIR = REPO_ROOT / "tests" / "golden"


def _canonical_output() -> Path | None:
    """The canonical output/ tree, or None when it is not on this machine.

    output/ is gitignored and multi-GB, so it is absent in CI, in fresh
    clones and in every git worktree. MPAIF_OUTPUT_DIR points the check at
    a tree elsewhere, which is how a worktree can still run it against the
    real results.
    """
    root = Path(os.environ.get("MPAIF_OUTPUT_DIR", REPO_ROOT / "output"))
    return root if (root / "door_code" / "aif.json").exists() else None


@functools.lru_cache(maxsize=None)
def _slice_runs(dirname: str) -> dict[str, dict]:
    """The committed slice of output/<dirname> (tests/golden/make_fixture.py)."""
    path = GOLDEN_DIR / f"slice_{dirname}.json"
    assert path.exists(), (
        f"no {path}; the drift detector's committed data is missing. "
        f"Regenerate with `uv run python tests/golden/make_fixture.py`"
    )
    return json.loads(path.read_text())["runs"]


# Sources of truth the drift detector runs against. The committed slice is
# always available, so — unlike the earlier skipif on output/ — the check
# never silently becomes a no-op. The canonical tree is added wherever it
# exists, and that is the arm that notices the slice itself going stale.
GOLDEN_SOURCES = ["slice"] + (
    ["canonical"] if _canonical_output() is not None else []
)


def _golden_run(source: str, dirname: str, planner: str) -> dict | None:
    if source == "slice":
        return _slice_runs(dirname).get(planner)
    path = _canonical_output() / dirname / f"{planner}.json"
    return json.loads(path.read_text()) if path.exists() else None


@pytest.mark.parametrize("source", GOLDEN_SOURCES)
@pytest.mark.parametrize("dirname", sorted(PUBLISHED_SCALARS))
def test_published_scalars_reproduce(dirname: str, source: str) -> None:
    """Drift detector: the paper's hand-typed scalars vs the emitter.

    A mismatch here does not necessarily mean the emitter is wrong — it
    means the published number no longer matches the results, which is
    exactly the condition nothing in this repo could previously detect.
    """
    bench = next(b for b in make_tables.BENCHMARKS if b.dirname == dirname)
    for planner, expected in PUBLISHED_SCALARS[dirname].items():
        run = _golden_run(source, dirname, planner)
        # Skipping an absent arm would silently turn this row's check into a
        # no-op, which is the failure mode the drift detector exists to
        # prevent. Every planner with a published row must be present.
        assert run is not None, (
            f"no {source} data for {dirname}/{planner}, so its published row "
            f"cannot be checked; drop it from PUBLISHED_SCALARS or restore "
            f"the run"
        )
        row = make_tables.scalar_row(run, bench)
        got = (
            row["early"], row["early_ci"],
            row["final"], row["final_ci"],
            row["mass"], row["mass_ci"],
            row["eti"]["fraction_identified"],
        )
        assert got == pytest.approx(expected, abs=_DISPLAY_TOL), (
            f"{dirname}/{planner} ({source}): published {expected}, computed "
            f"{tuple(round(v, 4) for v in got)}"
        )


@pytest.mark.parametrize("source", GOLDEN_SOURCES)
@pytest.mark.parametrize("dirname", sorted(PUBLISHED_TIMING))
def test_published_timing_reproduces(dirname: str, source: str) -> None:
    for planner, expected in PUBLISHED_TIMING[dirname].items():
        run = _golden_run(source, dirname, planner)
        assert run is not None, (
            f"no {source} data for {dirname}/{planner}, so its published "
            f"timing cell cannot be checked; drop it from PUBLISHED_TIMING "
            f"or restore the run"
        )
        got = per_decision_planning_time(
            run["per_episode"]["planning_time_s"],
            run["per_episode"]["n_steps"],
        )
        # The published cells carry 2 decimals below 10 and 1 above.
        tol = 0.05 if expected >= 10 else 5e-3
        assert got == pytest.approx(expected, abs=tol), (
            f"{dirname}/{planner} ({source}): published {expected}, computed "
            f"{got:.4f}"
        )
