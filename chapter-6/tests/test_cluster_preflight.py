"""Tests for cluster/preflight.py — the submit_all stage-coverage gate.

The first two tests pin failures a reviewer produced empirically against
an earlier version of the gate: it passed with a whole planner slate's
submit script deleted, and it accepted a commented-out call site as
"wired". Both would let an overnight submit silently skip work, which is
the exact failure the gate exists to prevent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_preflight():
    """Import cluster/preflight.py by path (cluster/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "cluster_preflight", REPO / "cluster" / "preflight.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cluster_preflight"] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load_preflight()


def _synthetic_tree(tmp_path, monkeypatch, *, stages, scripts, submit_all) -> None:
    """Point preflight at a throwaway repo tree.

    The canonical-stage floor and the three exemption maps are cleared:
    none of them is what the synthetic-tree tests exercise, and leaving
    them on would make every such tree fail for the wrong reason.
    """
    cluster = tmp_path / "cluster"
    cluster.mkdir(exist_ok=True)
    (cluster / "submit_all.sh").write_text(submit_all)
    for name in scripts:
        (cluster / name).write_text("#!/usr/bin/env bash\n")
    monkeypatch.setattr(pf, "REPO", tmp_path)
    monkeypatch.setattr(pf, "CLUSTER", cluster)
    monkeypatch.setattr(pf, "load_stages", lambda: stages)
    monkeypatch.setattr(pf, "CANONICAL_STAGES", frozenset())
    monkeypatch.setattr(pf, "LOCAL_ONLY", set())
    monkeypatch.setattr(pf, "COVERED_BY", {})
    monkeypatch.setattr(pf, "PENDING", {})
    monkeypatch.setattr(pf, "DEFERRED", {})


def test_repo_passes_its_own_gate(capsys) -> None:
    """The committed tree must be submittable, or submit_all aborts."""
    assert pf.main() == 0
    assert "preflight OK" in capsys.readouterr().out


def test_slate_needing_both_partitions_requires_both_scripts() -> None:
    """A stage running JAX *and* NumPy planners needs both submit
    scripts. Accepting either one let all four CPU baselines (bamcp,
    boss, factored_bamcp, ids) go unrun while the gate reported OK."""
    both = {"aif", "vbp", "oracle", "bamcp", "boss", "ids"}
    assert pf.required_partitions(both) == {"gpu", "cpu"}

    # And the single-partition slates still resolve to one.
    assert pf.required_partitions({"bamcp"}) == {"cpu"}
    assert pf.required_partitions({"aif"}) == {"gpu"}
    # `random` runs on both, so alone it forces neither specifically.
    assert pf.required_partitions({"random"}) == {"any"}


def test_commented_out_call_site_does_not_count_as_wired() -> None:
    """The gate regexes submit_all.sh; without stripping comments, a
    commented-out `collect` line still matched and the stage reported
    ok."""
    text = (
        "collect door_code cluster/submit_door_code.sh\n"
        "# collect rock_sample cluster/submit_rock_sample.sh\n"
        "bash cluster/submit_cpu_door_code.sh   # trailing comment\n"
    )
    stripped = pf._strip_comments(text)
    assert "submit_door_code.sh" in stripped
    assert "submit_cpu_door_code.sh" in stripped
    assert "submit_rock_sample.sh" not in stripped


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"foreach": {"aif": {}, "bamcp": {}}}, {"aif", "bamcp"}),
        ({"matrix": {"planner": ["aif", "ids"], "K": "${x}"}}, {"aif", "ids"}),
        ({"cmd": "uv run python s.py --planner bamcp --K 6"}, {"bamcp"}),
        # Unresolved interpolation is not a planner name.
        ({"cmd": "uv run python s.py --planner ${item.planner}"}, set()),
    ],
)
def test_planner_slate_extraction(body, expected) -> None:
    assert pf.planner_slate(body) == expected


def test_every_canonical_stage_declares_both_partitions() -> None:
    """Guards the structural GPU/CPU split: each canonical benchmark runs
    JAX and NumPy planners, so each needs two submit scripts."""
    stages = pf.load_stages()
    for stage in sorted(pf.CANONICAL_STAGES):
        slate = pf.planner_slate(stages[stage])
        assert pf.required_partitions(slate) == {"gpu", "cpu"}, stage


def test_deferred_stage_is_skipped_but_never_silent(monkeypatch, capsys) -> None:
    """A stage deferred on cost grounds must not fail the gate, and must
    still be printed. Silence is the failure mode that matters: an
    overnight submit that quietly omits a stage reads as a complete run.
    """
    monkeypatch.setitem(pf.DEFERRED, "door_code", "test reason 12345")
    assert pf.main() == 0
    out = capsys.readouterr().out
    assert "DEFERRED door_code" in out
    assert "test reason 12345" in out
    assert "deferred by choice" in out
    # It must NOT be reported as submittable.
    assert "ok       door_code  (" not in out


def test_deferred_is_not_pending(monkeypatch, capsys) -> None:
    """DEFERRED and PENDING must stay distinct. PENDING means 'someone is
    landing this, delete the entry when they do'; DEFERRED means 'we chose
    not to spend the SBU'. Collapsing them invites a future reader to
    'fix' a deferral by wiring a stage that costs 46% of the budget.
    """
    assert not set(pf.DEFERRED) & set(pf.PENDING), (
        "a stage cannot be both awaiting wiring and deliberately unwired"
    )
    # DEFERRED is checked first, so a stage in both would silently take the
    # deferral path and its coverage claim would never be verified.
    assert not set(pf.DEFERRED) & set(pf.COVERED_BY), (
        "a stage cannot be both deliberately unwired and covered by a sibling"
    )
    # Every deferral states its cost, so the decision can be revisited
    # with numbers rather than re-derived from scratch.
    for stage, reason in pf.DEFERRED.items():
        assert "SBU" in reason, f"{stage}: deferral reason states no cost"


# --- negative path -------------------------------------------------------
# Every test above asserts the gate PASSES. A gate that cannot be shown to
# fail is not a gate, and two fail-open holes (an unclassified planner name
# and an empty stage list, both reported "preflight OK", rc 0) survived
# precisely because nothing here asserted main() == 1.


def test_missing_submit_script_fails_the_gate(monkeypatch, capsys) -> None:
    """A stage with no submit script at all must block a submit run."""
    stages = pf.load_stages()
    stages["ghost_benchmark"] = {"foreach": {"aif": {}, "bamcp": {}}}
    monkeypatch.setattr(pf, "load_stages", lambda: stages)

    assert pf.main() == 1
    captured = capsys.readouterr()
    assert "MISSING  ghost_benchmark" in captured.out
    assert "preflight FAILED" in captured.err


def test_unwired_script_fails_the_gate(tmp_path, monkeypatch, capsys) -> None:
    """A submit script that exists but is never called is the original
    door_code_simscaling failure: shards no one ever submits."""
    _synthetic_tree(
        tmp_path, monkeypatch,
        stages={"ghost": {"foreach": {"aif": {}}}},
        scripts=["submit_ghost.sh"],
        submit_all="collect other cluster/submit_other.sh\n",
    )

    assert pf.main() == 1
    captured = capsys.readouterr()
    assert "UNWIRED  ghost" in captured.out
    assert "ok       ghost" not in captured.out
    assert "preflight FAILED" in captured.err


def test_unclassified_planner_fails_closed(tmp_path, monkeypatch, capsys) -> None:
    """An unrecognized planner name must be a hard problem.

    It used to yield an empty requirement set, which made the per-partition
    loop a no-op and landed the stage in `ok` — so a stage whose script
    submit_all.sh never calls reported "preflight OK" with rc 0. New
    planner names arrive with every new baseline, so this path is live.
    """
    with pytest.raises(ValueError, match="brand_new_planner"):
        pf.required_partitions({"aif", "brand_new_planner"})

    _synthetic_tree(
        tmp_path, monkeypatch,
        stages={"ghost": {"foreach": {"brand_new_planner": {}}}},
        scripts=["submit_ghost.sh"],
        submit_all="collect other cluster/submit_other.sh\n",
    )

    assert pf.main() == 1
    captured = capsys.readouterr()
    assert "UNKNOWN  ghost" in captured.out
    assert "ok       ghost" not in captured.out


def test_no_run_stages_cannot_report_ok(monkeypatch, capsys) -> None:
    """The floor: an empty stage list printed "all 0 run-stages are
    submittable" and passed, so a truncated or misparsed dvc.yaml read as
    a clean bill of health."""
    monkeypatch.setattr(pf, "load_stages", lambda: {})
    monkeypatch.setattr(pf, "LOCAL_ONLY", set())
    monkeypatch.setattr(pf, "PENDING", {})
    monkeypatch.setattr(pf, "DEFERRED", {})

    assert pf.main() == 1
    captured = capsys.readouterr()
    assert "NOSTAGE  door_code" in captured.out
    assert "run-stages are submittable" not in captured.out


def test_typo_in_an_exemption_map_fails_the_gate(monkeypatch, capsys) -> None:
    """A misspelled key is worse than no key: the name it meant to write
    stays unexempted *and* unchecked, and the branch is dead."""
    monkeypatch.setattr(pf, "LOCAL_ONLY", pf.LOCAL_ONLY | {"rock_sample_vis"})

    assert pf.main() == 1
    assert "BADKEY   rock_sample_vis" in capsys.readouterr().out


def test_forward_refs_are_narrow_and_loud(capsys) -> None:
    """The one escape hatch from the key check: a stage whose dvc.yaml
    entry lands in another in-flight PR. Each entry must be in use, must
    still be unresolved, and must be printed."""
    used = (
        set(pf.LOCAL_ONLY) | set(pf.COVERED_BY)
        | set(pf.PENDING) | set(pf.DEFERRED)
    )
    assert set(pf.FORWARD_REFS) <= used, "forward ref exempts nothing"

    landed = set(pf.FORWARD_REFS) & set(pf.load_stages())
    assert not landed, (
        f"forward reference(s) have landed, drop them from FORWARD_REFS so "
        f"the key check covers them again: {sorted(landed)}"
    )

    assert pf.main() == 0
    out = capsys.readouterr().out
    for key in pf.FORWARD_REFS:
        assert f"FORWARD  {key}" in out


@pytest.mark.parametrize(
    "cover_body, cover_name, expected",
    [
        # The declared script is not there at all.
        (None, "submit_sibling.sh", "MISSING  ghost"),
        # Present and called, but it does not submit this stage — the
        # failure a declaration-only exemption would wave through.
        ("sbatch --export=ALL,STAGE=other cluster/job_aggregate.sh\n",
         "submit_sibling.sh", "UNCOVERED ghost"),
        # Names the stage in a comment only.
        ("# STAGE=ghost is submitted here one day\n",
         "submit_sibling.sh", "UNCOVERED ghost"),
        # Submits the stage, but submit_all.sh never runs it.
        ("sbatch --export=ALL,STAGE=ghost cluster/job_aggregate.sh\n",
         "submit_orphan.sh", "UNWIRED  ghost"),
    ],
)
def test_covered_by_is_verified_not_trusted(
    tmp_path, monkeypatch, capsys, cover_body, cover_name, expected,
) -> None:
    """COVERED_BY exempts a stage from the `submit_<stage>.sh` naming
    convention, so it is the one place a stage can claim coverage it does
    not have. Each way that claim can be false must fail the gate — an
    exemption taken on trust is the same fail-open hole as an unclassified
    planner, and here it hides a stage that never gets submitted."""
    _synthetic_tree(
        tmp_path, monkeypatch,
        stages={"ghost": {"cmd": "uv run python s.py --planner aif"}},
        scripts=[],
        submit_all="bash cluster/submit_sibling.sh\n",
    )
    if cover_body is not None:
        (tmp_path / "cluster" / cover_name).write_text(cover_body)
    monkeypatch.setattr(pf, "COVERED_BY", {"ghost": cover_name})

    assert pf.main() == 1
    out = capsys.readouterr().out
    assert expected in out
    assert "ok       ghost" not in out


def test_covered_by_accepts_a_real_rider(tmp_path, monkeypatch, capsys) -> None:
    """The positive path: a stage genuinely submitted by a sibling's script
    passes, and says which script covers it."""
    _synthetic_tree(
        tmp_path, monkeypatch,
        stages={"ghost": {"cmd": "uv run python s.py --planner aif"}},
        scripts=[],
        submit_all="bash cluster/submit_sibling.sh\n",
    )
    (tmp_path / "cluster" / "submit_sibling.sh").write_text(
        "sbatch --export=ALL,STAGE=ghost cluster/job_aggregate.sh\n"
    )
    monkeypatch.setattr(pf, "COVERED_BY", {"ghost": "submit_sibling.sh"})

    assert pf.main() == 0
    assert "ok       ghost  (via submit_sibling.sh)" in capsys.readouterr().out


def test_pending_softens_a_partially_landed_script_pair(
    tmp_path, monkeypatch, capsys,
) -> None:
    """PENDING gaps are warnings, not blockers — including after an owner
    lands one script of a required pair. Only the zero-scripts case was
    softened, so a half-landed pair flipped the gate to hard-fail and
    blocked the whole overnight submit."""
    _synthetic_tree(
        tmp_path, monkeypatch,
        stages={"ghost": {"foreach": {"aif": {}, "bamcp": {}}}},
        scripts=["submit_ghost.sh"],
        submit_all="collect ghost cluster/submit_ghost.sh\n",
    )
    monkeypatch.setitem(pf.PENDING, "ghost", "cpu script lands in PR 99")

    assert pf.main() == 0
    captured = capsys.readouterr()
    assert "PENDING  MISSING  ghost" in captured.out
    assert "cpu script lands in PR 99" in captured.out
    assert "ghost" in captured.err  # named in the skip warning
