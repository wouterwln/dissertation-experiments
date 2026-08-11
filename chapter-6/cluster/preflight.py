"""Check that every DVC run-stage is actually submittable by submit_all.sh.

`submit_all.sh` historically hard-coded the five canonical stages, so
ablation stages added later (door_code_simscaling, door_code_preference)
were never submitted by it and silently produced no cluster output. This
check makes that failure loud and immediate instead of discovering it
after an overnight run.

Four things are verified per stage, all derived from `dvc.yaml` rather
than from a hand-maintained list:

1. The stage's planner slate is read from its `foreach` keys,
   `matrix.planner`, or a literal `--planner X` in its command.
2. Each partition that slate needs has a submit script. The GPU/CPU
   split is structural — `submit_<stage>.sh` covers the JAX planners and
   `submit_cpu_<stage>.sh` the BA-RL slate — so a stage running both
   needs *both*. Accepting either one lets a whole slate go unrun. The
   exception is a stage submitted as a rider inside a sibling stage's
   script (see COVERED_BY), which is verified against that script.
3. `submit_all.sh` actually calls each of those scripts, checked against
   the file with comments stripped so a commented-out call cannot
   satisfy it.
4. A stage with a `scripts/plot_<stage>.py` is covered by a plot job, so
   an enabled ablation cannot produce shards and an aggregate but no
   figure.

Every check fails closed. A slate naming a planner this file does not
know is a hard problem, not an empty requirement set: unknown names used
to skip checks 2-4 entirely and report a genuinely unwired stage as ok.
For the same reason the run-stage list is floored at the canonical
benchmarks, and every LOCAL_ONLY / COVERED_BY / PENDING / DEFERRED key
must name a real stage — a typo there silently exempts the stage it
meant to name.

Run standalone, or let `submit_all.sh` call it before it submits
anything:

    uv run python cluster/preflight.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CLUSTER = REPO / "cluster"

# Stages that legitimately have no cluster submit script. Plot stages are
# checked via rule 4 against their run-stage, not on their own.
LOCAL_ONLY = {
    "rock_sample_viz",    # renders animations from an existing aif.json
    "plot_environments",  # static schematics, no run data
    # Emits .tex from result JSONs that already exist — a seconds-long
    # postprocess, and the only run-stage here that is not named plot_*.
    # Unlike the figures, it is NOT chained on the cluster: run
    # `dvc repro tables` after pulling the run outputs back.
    "tables",
}

# The five canonical benchmarks. Each must exist as a dvc.yaml run-stage:
# without that floor, a truncated, renamed or misparsed dvc.yaml yields an
# empty stage list and the gate reports "all 0 run-stages are submittable"
# and passes, which is the one answer it must never give.
CANONICAL_STAGES = frozenset({
    "door_code",
    "minigrid_door_code",
    "rock_sample",
    "rock_sample_kscaling",
    "factored_tiger",
})

# Stages submitted as a rider inside a *sibling* stage's submit script,
# mapped to that script. The `submit_<stage>.sh` naming convention is how
# rules 2-3 locate a stage's script, so a stage submitted under another
# name reads as MISSING even though it does run.
#
# Declaring it here does not exempt it — the coverage is verified: the
# named script must exist, must export `STAGE=<stage>` (so it genuinely
# submits this stage rather than merely mentioning it in a comment), and
# must itself be called by submit_all.sh.
COVERED_BY: dict[str, str] = {
    # One extra AIF cell at Katt-2019's listen_cost, submitted at the end
    # of the preference sweep's script into its own output directory. Its
    # figure is the preference one (--listen-cost-json), not its own.
    "factored_tiger_listen_cost": "submit_factored_tiger_preference.sh",
}

# Gaps that are known and already owned, reported as warnings rather than
# hard failures so an overnight submit is not blocked by in-flight work.
# This covers *every* finding for the stage, not just "no scripts at all":
# the moment an owner lands one of a required pair, a per-stage check
# would otherwise flip from warning to blocker mid-flight.
# Delete an entry the moment its owner lands — a stage left here is a
# stage submit_all will not run. Empty is the steady state; the sole
# entry (door_code_preference) was cleared when the sweeps PR landed.
PENDING: dict[str, str] = {}

# Stages deliberately excluded from the cluster slate on cost grounds, with
# the reason. Distinct from PENDING: these are NOT waiting on anyone, and
# wiring one is a budget decision, not a fix. Keep the SBU figure in the
# reason so a future reader can re-decide with numbers rather than guessing.
DEFERRED: dict[str, str] = {
    # Too cheap to be worth a cluster template, not too expensive to run
    # locally — the opposite end of the axis from a stage deferred on cost
    # (rock_sample_depth_ids used to sit above these on exactly those
    # grounds; it was removed from dvc.yaml rather than shipped frozen and
    # unrun, and its rationale now lives in the params.yaml
    # rock_sample_depth block). Both stages are ~11 min single-core
    # combined at the shipped n_seeds=25 (measured: ~21 s/seed door_code,
    # ~5 s/seed factored_tiger, all four arms), i.e. ~0.5 SBU if a genoa
    # template existed. Raising n_seeds to 800 — what the episode-0 value
    # slice needs for 2 sigma — is ~5.8 h local / ~15 SBU, still small enough
    # that a template only buys wall-clock. Run both with
    # `dvc repro exact_reference_door_code exact_reference_factored_tiger`.
    "exact_reference_door_code": (
        "~0.5 SBU; local-only by design. See the shared note above."
    ),
    "exact_reference_factored_tiger": (
        "~0.5 SBU; local-only by design. See the shared note above."
    ),
    # Deferred on consumer grounds, not cost: <=44 SBU worst case (10 genoa
    # tasks x 2 cores x the 2 h cap), and probably far less. Nothing in the
    # repo reads output/aif_cpu_timing/aif.json — no stage deps on it, and
    # the only seconds-per-decision conversion is in plot_door_code.py,
    # which reads output/door_code/. The table emitter that was meant to
    # consume it does not, so submitting this buys a file no figure or table
    # opens. Wire it (submit_cpu_aif_timing.sh is written and works) the
    # moment a consumer lands; until then `dvc repro aif_cpu_timing` covers
    # anyone who wants the number by hand.
    "aif_cpu_timing": (
        "<=44 SBU, but nothing reads output/aif_cpu_timing/aif.json yet. "
        "See the note above; flip this the moment a consumer lands."
    ),
}

# Keys in the maps above must name a real dvc.yaml stage, or a typo
# silently exempts the stage it meant to name. The one escape hatch is a
# stage whose dvc.yaml entry lands in another in-flight PR: such a name is
# a legitimate forward reference until that PR merges. Every entry states
# which PR resolves it, and empty is the steady state — the sole entry
# (rock_sample_depth_ids) was cleared when the matched-depth PR landed.
FORWARD_REFS: dict[str, str] = {}

# Which partition each planner runs on. `submit_*.sh` iterates the JAX
# planners on gpu_a100; `submit_cpu_*.sh` iterates the NumPy/Numba slate
# on genoa. See cluster/README.md.
GPU_PLANNERS = {
    "aif", "vbp", "psrl", "oracle",
    "persistent_oracle", "persistent_oracle_vbp",
}
CPU_PLANNERS = {
    "belief_q", "boss", "bamcp", "pomcp", "ids", "factored_bamcp",
}
# Submitted on both partitions; on its own it forces neither.
EITHER_PLANNERS = {"random"}


def _strip_comments(text: str) -> str:
    """Drop full-line and trailing shell comments.

    Without this, commenting out a `collect ...` line still satisfies the
    "is it called" check — the string is present, just inert.
    """
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


def load_stages() -> dict:
    with (REPO / "dvc.yaml").open() as fh:
        return yaml.safe_load(fh).get("stages", {})


def run_stages(stages: dict) -> list[str]:
    return sorted(
        name for name in stages
        if not name.startswith("plot") and name not in LOCAL_ONLY
    )


def planner_slate(body: dict) -> set[str]:
    """Planners a stage runs, from foreach / matrix.planner / --planner."""
    foreach = body.get("foreach")
    if isinstance(foreach, dict):
        return set(foreach)
    if isinstance(foreach, list):
        return {v for v in foreach if isinstance(v, str)}

    matrix = body.get("matrix") or {}
    if isinstance(matrix.get("planner"), list):
        return set(matrix["planner"])

    cmd = body.get("cmd")
    cmd = " ".join(cmd) if isinstance(cmd, list) else (cmd or "")
    # Skip unresolved ${item.*} interpolations — those come from matrix.
    return {
        p for p in re.findall(r"--planner\s+(\S+)", cmd)
        if not p.startswith("${")
    }


def map_key_problems(stages: dict) -> tuple[list[str], list[str]]:
    """Validate the exemption-map keys against dvc.yaml.

    Returns (problems, forward-reference notes). A key that names no
    stage is a typo, and a typo is worse than useless: the misspelled
    stage is never exempted, and the stage it meant to name is never
    checked. Declared forward references are reported, never silent.
    """
    declared = set(stages)
    problems: list[str] = []
    forwards: list[str] = []
    for label, keys in (
        ("LOCAL_ONLY", LOCAL_ONLY),
        ("COVERED_BY", COVERED_BY),
        ("PENDING", PENDING),
        ("DEFERRED", DEFERRED),
    ):
        for key in sorted(keys):
            if key in declared:
                continue
            if key in FORWARD_REFS:
                forwards.append(
                    f"FORWARD  {key}  — in {label}; {FORWARD_REFS[key]}"
                )
            else:
                problems.append(
                    f"BADKEY   {key}  — listed in {label} but dvc.yaml "
                    "declares no such stage; typo? (the stage it meant to "
                    "name is not exempted, and not checked either)"
                )
    return problems, forwards


def required_partitions(slate: set[str]) -> set[str]:
    """Partitions a slate needs. Empty slate -> unknown, require at least
    one script (reported by the caller as a single-script stage).

    Raises ValueError on a planner with no known partition. Returning an
    empty set instead made the caller's check loop a no-op, so a stage
    whose submit script submit_all.sh never calls reported "ok" — the
    exact unwired case this gate exists to catch. New planner names must
    be classified here before their stage can pass.
    """
    unknown = slate - GPU_PLANNERS - CPU_PLANNERS - EITHER_PLANNERS
    if unknown:
        raise ValueError(
            f"unclassified planner(s) {', '.join(sorted(unknown))}; add "
            "each to GPU_PLANNERS / CPU_PLANNERS / EITHER_PLANNERS in "
            "cluster/preflight.py so its partition is known"
        )
    need = set()
    if slate & GPU_PLANNERS:
        need.add("gpu")
    if slate & CPU_PLANNERS:
        need.add("cpu")
    if not need and (slate & EITHER_PLANNERS or not slate):
        need.add("any")
    return need


def plot_findings(stage: str, submit_all: str) -> list[str]:
    """Rule 4: an enabled stage with a plot script must get a plot job, or
    it produces shards and an aggregate but no figure."""
    if not (REPO / "scripts" / f"plot_{stage}.py").exists():
        return []
    if re.search(rf"\b{re.escape(stage)}\b", submit_all):
        return []
    return [
        f"NOPLOT   {stage}  — scripts/plot_{stage}.py exists but "
        f"submit_all.sh submits no plot job for it"
    ]


def covered_findings(stage: str, submit_all: str) -> list[str]:
    """Verify a COVERED_BY declaration instead of trusting it.

    A declaration that is merely believed is the same fail-open hole as an
    unclassified planner: the stage skips rules 2-3 and reports ok. So the
    covering script has to exist, has to export this stage's name to
    job_aggregate.sh, and has to be reachable from submit_all.sh.
    """
    name = COVERED_BY[stage]
    script = CLUSTER / name
    if not script.exists():
        return [
            f"MISSING  {stage}  — declared covered by {name}, which does "
            f"not exist"
        ]
    body = _strip_comments(script.read_text())
    if not re.search(rf"STAGE={re.escape(stage)}\b", body):
        return [
            f"UNCOVERED {stage}  — {name} never exports STAGE={stage}, so "
            f"it does not submit this stage (comments do not count)"
        ]
    if not re.search(rf"\b{re.escape(name)}\b", submit_all):
        return [
            f"UNWIRED  {stage}  — covered by {name}, but submit_all.sh "
            f"never calls it"
        ]
    return []


def main() -> int:
    stages = load_stages()
    submit_all_raw = (CLUSTER / "submit_all.sh").read_text()
    submit_all = _strip_comments(submit_all_raw)

    problems: list[str] = []
    pending: list[tuple[str, str]] = []
    ok: list[str] = []

    deferred: list[str] = []

    key_problems, forwards = map_key_problems(stages)
    problems.extend(key_problems)

    declared = set(run_stages(stages))
    for stage in sorted(CANONICAL_STAGES - declared):
        problems.append(
            f"NOSTAGE  {stage}  — canonical benchmark has no dvc.yaml "
            "run-stage; dvc.yaml is truncated, renamed or misparsed"
        )

    for stage in run_stages(stages):
        if stage in DEFERRED:
            deferred.append(stage)
            continue

        if stage in COVERED_BY:
            findings = (
                covered_findings(stage, submit_all)
                + plot_findings(stage, submit_all)
            )
            if not findings:
                ok.append(f"ok       {stage}  (via {COVERED_BY[stage]})")
            elif stage in PENDING:
                pending.extend((stage, f) for f in findings)
            else:
                problems.extend(findings)
            continue

        slate = planner_slate(stages[stage])
        scripts = {
            "gpu": CLUSTER / f"submit_{stage}.sh",
            "cpu": CLUSTER / f"submit_cpu_{stage}.sh",
        }
        present = {k for k, p in scripts.items() if p.exists()}
        findings: list[str] = []

        try:
            need = required_partitions(slate)
        except ValueError as exc:
            need = set()
            findings.append(f"UNKNOWN  {stage}  — {exc}")

        if not present:
            findings.append(
                f"MISSING  {stage}  — no submit_{stage}.sh or "
                f"submit_cpu_{stage}.sh (slate: "
                f"{', '.join(sorted(slate)) or 'unknown'})"
            )
        elif not findings:
            # Skipped when the slate is unclassified: which partitions it
            # needs is exactly what is unknown, so checking would guess.
            wanted = present if need == {"any"} else need
            for part in sorted(wanted):
                script = scripts[part]
                if not script.exists():
                    findings.append(
                        f"MISSING  {stage}  — slate needs the {part} partition "
                        f"({', '.join(sorted(slate & (GPU_PLANNERS if part == 'gpu' else CPU_PLANNERS)))}) "
                        f"but {script.name} does not exist"
                    )
                elif not re.search(rf"\b{re.escape(script.name)}\b", submit_all):
                    findings.append(
                        f"UNWIRED  {stage}  — {script.name} exists but "
                        f"submit_all.sh never calls it"
                    )

            findings.extend(plot_findings(stage, submit_all))

        # Findings are per-stage, so no substring scan over other stages'
        # problems: `stage in problem_text` matched any stage whose name is
        # a prefix of another's (door_code / door_code_simscaling) and only
        # gave the right answer by sort order.
        if not findings:
            ok.append(f"ok       {stage}  ({'+'.join(sorted(present))})")
        elif stage in PENDING:
            pending.extend((stage, f) for f in findings)
        else:
            problems.extend(findings)

    for line in ok:
        print(f"  {line}")
    for stage in deferred:
        print(f"  DEFERRED {stage}  — {DEFERRED[stage]}")
    for line in forwards:
        print(f"  {line}")
    for stage, line in pending:
        print(f"  PENDING  {line}  ({PENDING[stage]})")
    for line in problems:
        print(f"  {line}")

    if problems:
        print(
            f"\npreflight FAILED: {len(problems)} problem(s). A submit_all "
            "run would silently skip work. Fix the scripts, or add the "
            "stage to LOCAL_ONLY here if it needs no cluster job.",
            file=sys.stderr,
        )
        return 1

    suffix = f" ({len(deferred)} deferred by choice)" if deferred else ""
    print(f"\npreflight OK: all {len(ok)} run-stages are submittable{suffix}.")
    if pending:
        stalled = sorted({stage for stage, _ in pending})
        print(
            f"WARNING: {len(stalled)} declared-pending stage(s) will be "
            f"skipped by this run: {', '.join(stalled)}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
