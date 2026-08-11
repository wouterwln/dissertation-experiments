"""Emit the paper's scalar / timing / ablation tables as .tex from output/.

Every number in the paper's Tables 1-4 (per-benchmark scalar summaries),
7 (planning time per decision), 8 (deferred ablation baselines) and 9
(BAMCP simulation-budget sweep) used to be a hand-typed LaTeX literal,
audited once by hand. This script computes them all from the canonical
`output/` JSONs so they cannot drift silently. It also emits the two
appendix tables of the exact Bayes-adaptive reference (per-decision
agreement, and block return against the tie-break band).

Conventions, all documented at their source:

  * Early / final windows: per-seed then across-seed mean +/- 1.96*SEM.
    See `learningpomdp/experiment/report.py` for why that convention and
    not the two alternatives. The window sizes come from
    `report.windows` in params.yaml, so the emitter's protocol cannot
    drift from the paper's unnoticed.
  * A caption states the block count behind each interval, not just the
    block count in the run header: a block whose whole window is NaN
    (`reward_normalised` where no positive reward was obtainable) drops
    out of that column, so one table can carry several `n`.
  * Where the early and final windows are not disjoint the caption says
    so and names the shared episodes. Every configured pair is disjoint
    today, so the clause is dormant: it is the guard that keeps a future
    `runner.n_episodes` change from quietly turning the contrast into one
    computed partly on shared episodes. The fix for an overlap is a
    window size in params.yaml, never a rescoping inside this emitter.
  * Identification: the **crossing fraction** is the primary number.
    The conditional median over crossing seeds compares different
    subpopulations of blocks across methods, so the secondary number is
    the right-censored median from
    `derived.episodes_to_identify.censored_*`.
  * Planning time: `per_episode.planning_time_s` is the *episode total*.
    The per-decision figure divides by `per_episode.n_steps`.
  * Oracle and Random are emitted as explicit reference bounds. Bold
    marks the best **learning** agent, so a bound cannot silently win a
    column, and the caption states the Oracle headroom. Oracle's
    identification cells are omitted (`NO_IDENTIFICATION`), which is the
    same exclusion the figures make.

Planners are looked up from an explicit ordered list, never by globbing,
so a missing arm is a visible warning rather than a silently shorter
table. In the timing table an absent result JSON prints as
`NO_RESULT_CELL` rather than sharing the `--` a present-but-non-finite
value gets: the emitter cannot see why a file is missing, and the caption
no longer claims to know.

Usage:
    uv run python scripts/make_tables.py \\
        --input-dir output --output-dir output/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from learningpomdp.experiment.door_code_runner import episodes_to_identify
from learningpomdp.experiment.report import (
    final_episode_mean_ci,
    paired_window_difference,
    per_decision_planning_time,
    theta_pairing,
    window_mean_ci,
    window_overlap,
)


@dataclass(frozen=True)
class Benchmark:
    dirname: str
    label: str          # display name in captions / row groups
    tex_label: str      # \label{...} the paper already uses
    metric: str         # per_episode key holding the reported return
    metric_word: str    # caption wording for that metric
    mass_key: str       # per_episode key holding true-persistent-theta mass
    n_early: int
    n_final: int
    family: str | None  # shared-task family note (reviewer item 11)


PARAMS_PATH = Path(__file__).resolve().parents[1] / "params.yaml"


def load_windows(path: Path = PARAMS_PATH) -> dict[str, tuple[int, int]]:
    """`report.windows` from params.yaml, as {dirname: (n_early, n_final)}.

    The window sizes live in params.yaml rather than here so that the
    emitter's protocol is a tracked parameter (`dvc.yaml`'s `tables`
    stage declares `report.windows`) instead of a literal buried in a
    script, where it could drift from the paper's protocol paragraph
    with nothing able to detect it -- the same failure mode this emitter
    removes one layer up.
    """
    with path.open() as f:
        windows = yaml.safe_load(f)["report"]["windows"]
    return {
        name: (int(spec["n_early"]), int(spec["n_final"]))
        for name, spec in windows.items()
    }


_WINDOWS = load_windows()


# Metrics follow sections/evaluation.tex's protocol paragraph: raw
# episode reward on the Door-Code family, normalized reward where the
# achievable reward varies with theta. The window sizes come from
# `report.windows` in params.yaml.
_DOOR_CODE_FAMILY = (
    "Door-Code and MiniGrid Door-Code are the same identification task "
    "-- one permutation parameter with probe-bit semantics -- differing "
    "in whether it is embedded in a grid, so they are not independent "
    "benchmarks"
)

BENCHMARKS = (
    Benchmark(
        "door_code", "Door-Code", "tab:door-code-scalars",
        "cumulative_reward", "mean returns",
        "posterior_mass_on_true_theta", *_WINDOWS["door_code"],
        _DOOR_CODE_FAMILY,
    ),
    Benchmark(
        "minigrid_door_code", "MiniGrid Door-Code", "tab:minigrid-scalars",
        "cumulative_reward", "mean returns",
        "posterior_mass_on_true_theta", *_WINDOWS["minigrid_door_code"],
        _DOOR_CODE_FAMILY,
    ),
    Benchmark(
        "factored_tiger", "Multi-Door Tiger", "tab:factored-tiger-scalars",
        "reward_normalised", "normalized rewards",
        "posterior_mass_on_persistent_theta", *_WINDOWS["factored_tiger"],
        None,
    ),
    Benchmark(
        "rock_sample", "RockSample-derived", "tab:rock-sample-scalars",
        "reward_normalised", "normalized rewards",
        "posterior_mass_on_persistent_theta", *_WINDOWS["rock_sample"],
        None,
    ),
)

PLANNER_LABELS = {
    "aif": "AIF-MP",
    "psrl": "PSRL",
    "bamcp": "BAMCP",
    "boss": "BOSS",
    "ids": "IDS",
    "factored_bamcp": "F-BA-POMCP",
    "vbp": "VBP",
    "belief_q": "Belief-Q",
    "pomcp": "POMCP",
    # PR #12 landed the two persistent-oracle arms as `persistent_oracle`
    # (AIF inner solve -- the reported reference arm, differing from `aif`
    # in exactly one respect: theta_persistent is given) and
    # `persistent_oracle_vbp` (VBP inner solve). The unsuffixed name
    # therefore carries the AIF variant and names no inner solver in its
    # label. Any further arm has to be added to PLANNER_LABELS,
    # REFERENCE_SLATE and TIMING_ORDER together, or it is dropped without
    # a warning.
    "persistent_oracle": "Persistent Oracle",
    "persistent_oracle_vbp": "Persistent Oracle (VBP)",
    "oracle": "Oracle",
    "random": "Random",
}

# Learning agents: the rows bold competes over.
LEARNING_SLATE = ("aif", "psrl", "bamcp", "boss", "ids", "factored_bamcp")
# Reference bounds, emitted but excluded from bold (reviewer item 5).
# rock_sample-only, so the loader warns and skips them on the other three
# benchmarks (which have no episodic theta, where they would equal
# `oracle`). Note that `persistent_oracle_vbp` over-senses and never
# scores above plain `vbp`, so it bounds nothing -- it is here to be
# emitted, not because it is a bound.
REFERENCE_SLATE = (
    "persistent_oracle", "persistent_oracle_vbp", "oracle", "random",
)
# Ablation baselines the paper defers to its appendix.
DEFERRED_SLATE = ("vbp", "belief_q", "pomcp")
# Timing table row order (matches the paper's Table 7 ordering).
TIMING_ORDER = (
    "aif", "vbp", "psrl", "oracle", "persistent_oracle",
    "persistent_oracle_vbp", "belief_q", "bamcp", "pomcp", "boss",
    "factored_bamcp", "ids", "random",
)

GENERATED_BY = "scripts/make\\_tables.py"
GENERATED_COMMENT = "% Generated by scripts/make_tables.py — do not edit by hand."

# Timing-table placeholder for "no result JSON on disk", kept distinct
# from the "--" a present-but-non-finite value prints: one is a gap in
# output/, the other is a property of a run that exists, and a single
# marker for both made the caption assert something the emitter cannot
# see (that the planner is unimplemented there).
NO_RESULT_CELL = r"\emph{n/r}"

# Placeholder for a cell whose quantity is not defined *for that arm*,
# as opposed to missing or non-finite.
NOT_APPLICABLE_CELL = r"\emph{n/a}"

# Arms whose identification cells (final mass, crossing fraction,
# censored median) are suppressed. Oracle plans on a one-hot at
# theta-star, so its logged filter mass measures only what a Bayesian
# filter learns from watching purely exploitative behaviour: Oracle is a
# reward bound, not an identification baseline. Commit 5e1605c dropped it
# from the identification panels (mass, Brier, episodes-to-identify) of
# `plot_paper_figures.py` and all four dashboards for exactly that
# reason, and these three columns are those same quantities, so keeping
# them here would leave a table and a figure disagreeing about whether
# the quantity applies to Oracle at all. Random stays: it is a
# meaningful passive-identification floor (same commit).
#
# `scalar_table`'s caption sentence is written for one arm. A second name
# here needs that sentence rewritten with its own reason, or the table
# would print `n/a` cells without saying why.
NO_IDENTIFICATION = ("oracle",)


def tt(text: str) -> str:
    r"""\texttt{...} with the LaTeX-unsafe characters escaped.

    Only `_` was escaped before, which is enough for the paths currently
    passed but silently emits invalid LaTeX for the rest of the set (a
    bare `%` comments out the remainder of the line, which would drop
    the end of a caption). None of the replacements introduces another
    trigger, so the order does not matter.
    """
    for char in "_%&#$":
        text = text.replace(char, "\\" + char)
    return "\\texttt{" + text + "}"


# --------------------------------------------------------------------------
# loading


def load_runs(env_dir: Path, planners: tuple[str, ...]) -> dict[str, dict]:
    """Load the named planners from env_dir, warning about absent ones."""
    runs: dict[str, dict] = {}
    for name in planners:
        path = env_dir / f"{name}.json"
        if not path.exists():
            print(
                f"warning: no {path} — skipping planner '{name}' in "
                f"{env_dir.name}",
                file=sys.stderr,
            )
            continue
        with path.open() as f:
            runs[name] = json.load(f)
    return runs


# --------------------------------------------------------------------------
# formatting


def _two_dp(value: float) -> str:
    """Two decimals, without a negative zero.

    A value in $(-0.005, 0)$ formats as ``-0.00'' under plain `%.2f`,
    which reads as a signed difference where the rounded value is
    exactly zero. Adding 0.0 to the rounded value maps -0.0 to 0.0.
    """
    return f"{round(value, 2) + 0.0:.2f}"


def fmt_pm(mean: float, ci: float, bold: bool = False) -> str:
    if not np.isfinite(mean):
        return "--"
    body = f"{_two_dp(mean)} \\pm {_two_dp(ci)}"
    return f"$\\boldsymbol{{{body}}}$" if bold else f"${body}$"


def fmt_frac(value: float, bold: bool = False) -> str:
    if not np.isfinite(value):
        return "--"
    body = _two_dp(value)
    return f"$\\boldsymbol{{{body}}}$" if bold else f"${body}$"


def fmt_censored_median(eti: dict) -> str:
    """Censored median [p25, p75], counting episodes from one.

    A quantile the survival curve never reaches within the budget prints
    as `{>}budget` rather than being dropped.
    """
    budget = int(eti["n_episodes_budget"])

    def one(key: str) -> str:
        value = eti[key]
        return f"{value + 1:.1f}" if value is not None else f"{{>}}{budget}"

    median, p25, p75 = (
        one("censored_median"), one("censored_p25"), one("censored_p75")
    )
    return f"${median}$ [${p25}$, ${p75}$]"


def fmt_time(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.005:
        return "$<0.01$"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def _best(values: dict[str, float]) -> set[str]:
    """Names whose value ties the maximum at two-decimal display precision.

    Empty when two or more competitors all tie, because then the maximum
    distinguishes nothing: bolding the whole column asserts a winner that
    the data does not contain. The concrete case is a benchmark where no
    method identifies, where the crossing fraction is 0.00 throughout and
    bold would mark every row as best.
    """
    finite = {k: v for k, v in values.items() if np.isfinite(v)}
    if not finite:
        return set()
    rounded = {k: round(v, 2) for k, v in finite.items()}
    top = max(rounded.values())
    winners = {k for k, v in rounded.items() if v == top}
    if len(winners) == len(rounded) > 1:
        return set()
    return winners


# --------------------------------------------------------------------------
# row computation


def scalar_row(run: dict, bench: Benchmark) -> dict:
    ret = np.asarray(run["per_episode"][bench.metric], dtype=float)
    mass = np.asarray(run["per_episode"][bench.mass_key], dtype=float)
    early, early_ci, early_n = window_mean_ci(ret, bench.n_early, "early")
    final, final_ci, final_n = window_mean_ci(ret, bench.n_final, "final")
    mass_mean, mass_ci, mass_n = final_episode_mean_ci(mass)
    # Recomputed from the raw posterior rather than read out of
    # `run["derived"]`: JSONs written before the censored fields existed
    # carry a `derived` block without them, and recomputing keeps the
    # tables and the runner on one code path.
    eti = episodes_to_identify(mass)
    return {
        "early": early, "early_ci": early_ci, "early_n": early_n,
        "final": final, "final_ci": final_ci, "final_n": final_n,
        "mass": mass_mean, "mass_ci": mass_ci, "mass_n": mass_n,
        "eti": eti,
        "n_seeds": int(run["n_seeds"]),
        "n_episodes": int(run["n_episodes"]),
    }


def scalar_body(
    rows: dict[str, dict], learning: tuple[str, ...],
    reference: tuple[str, ...], n_cols: int,
) -> list[str]:
    """Learning rows, then a labelled reference-bound group."""
    best = {
        col: _best({n: rows[n][col] for n in learning if n in rows})
        for col in ("early", "final", "mass")
    }
    best["frac"] = _best(
        {n: rows[n]["eti"]["fraction_identified"] for n in learning if n in rows}
    )
    lines: list[str] = []

    def emit(name: str, compete: bool) -> str:
        r = rows[name]
        if name in NO_IDENTIFICATION:
            identification = [NOT_APPLICABLE_CELL] * 3
        else:
            identification = [
                fmt_pm(r["mass"], r["mass_ci"],
                       compete and name in best["mass"]),
                fmt_frac(r["eti"]["fraction_identified"],
                         compete and name in best["frac"]),
                fmt_censored_median(r["eti"]),
            ]
        return " & ".join([
            PLANNER_LABELS[name],
            fmt_pm(r["early"], r["early_ci"], compete and name in best["early"]),
            fmt_pm(r["final"], r["final_ci"], compete and name in best["final"]),
            *identification,
        ]) + r" \\"

    for name in learning:
        if name in rows:
            lines.append(emit(name, True))
    present_refs = [n for n in reference if n in rows]
    if present_refs:
        lines.append(r"\midrule")
        note = r"\emph{Reference bounds (not learning agents; excluded from bold)}"
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{{note}}} \\")
        for name in present_refs:
            lines.append(emit(name, False))
    return lines


# --------------------------------------------------------------------------
# tables


_SCALAR_HEADER = (
    r"Method & Early & Final & Final mass & Identified & "
    r"Median ep. (censored) \\"
)


def _table(body: list[str], colspec: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])


def _overlap_clause(bench: Benchmark, n_episodes: int) -> str:
    """Caption clause disclosing a non-disjoint early/final window pair.

    Empty when the windows are disjoint, which every pair in
    `report.windows` now is -- Multi-Door Tiger's (3, 3) of 5 episodes
    was the one overlapping pair and was corrected to (2, 2) there. So
    this clause fires on no current benchmark and is kept as the guard:
    a later `runner.n_episodes` cut would re-create the overlap, and it
    must surface rather than have the two windows silently average some
    of the same episodes into both sides of a before/after contrast.

    When it does fire the windows are still reported at their configured
    sizes and the shared episodes named. Rescoping them here would put
    the emitter's protocol at odds with the `report.windows` the paper
    documents; the fix belongs in params.yaml.
    """
    overlap = window_overlap(n_episodes, bench.n_early, bench.n_final)
    if not overlap:
        return ""
    first, last = n_episodes - bench.n_final + 1, bench.n_early
    shared = (
        f"episode {first}" if first == last else f"episodes {first}--{last}"
    )
    print(
        f"warning: {bench.dirname}: early ({bench.n_early}) and final "
        f"({bench.n_final}) windows overlap in {overlap} of "
        f"{n_episodes} episodes; the caption discloses it",
        file=sys.stderr,
    )
    return (
        f"The two windows are \\emph{{not}} disjoint here: "
        f"${bench.n_early} + {bench.n_final} > {n_episodes}$, so they share "
        f"{overlap} of the {n_episodes} episodes ({shared}, counting from "
        f"one) and the contrast is partly computed on the same data. The "
        f"window sizes are reported as configured rather than silently "
        f"rescoped here. "
    )


_INTERVAL_COLUMNS = (("early_n", "early"), ("final_n", "final"),
                     ("mass_n", "final mass"))


def _effective_n_clause(entries: list[tuple[str, dict, tuple[str, ...]]]) -> str:
    """Caption clause naming every interval that used fewer blocks.

    Empty when every cell used every block. A block whose whole window
    is NaN drops out of that column's mean, so the block count behind an
    interval is not in general the count in the run header, and quoting
    only the header count next to the intervals is exactly the kind of
    unchecked claim this emitter exists to remove.

    `entries` are (row label, row, printed columns): only rows the table
    prints, and per row only the columns it prints -- naming a row or a
    column that is not in the table would be a new false claim of its
    own, and the identification columns are suppressed for some arms
    (`NO_IDENTIFICATION`). Columns are named rather than positional so a
    row that prints two of them cannot be misread as the first two.

    `reward_normalised` is the only reported metric that is ever
    undefined, so the clause can name it whenever it fires.
    """
    deviating = []
    for label, r, columns in entries:
        low = [
            f"{name} ${r[key]}$"
            for key, name in _INTERVAL_COLUMNS
            if key in columns and r[key] < r["n_seeds"]
        ]
        if low:
            deviating.append(f"{label} ({', '.join(low)}, of {r['n_seeds']})")
    if not deviating:
        return ""
    return (
        " A block enters a column only where its window is defined "
        "(\\texttt{reward\\_normalised} is undefined on episodes where no "
        "positive reward was obtainable), so these intervals average "
        "fewer blocks than the run holds: " + "; ".join(deviating) + "."
    )


def scalar_table(bench: Benchmark, runs: dict[str, dict]) -> str:
    rows = {name: scalar_row(run, bench) for name, run in runs.items()}
    body = [_SCALAR_HEADER, r"\midrule"]
    body += scalar_body(rows, LEARNING_SLATE, REFERENCE_SLATE, 6)
    any_row = next(iter(rows.values()))
    caption = (
        f"{bench.label} scalar summaries. "
        f"Early and final are {bench.metric_word} over the first "
        f"{bench.n_early} and last {bench.n_final} episodes of a block, "
        f"averaged within a block and then across blocks, with 95\\% "
        f"confidence intervals of the across-block mean. "
        + _overlap_clause(bench, any_row["n_episodes"]) +
        f"Final mass is posterior mass on the true persistent parameter in "
        f"the final episode. "
        f"Identified is the fraction of blocks whose true-parameter mass "
        f"reaches {any_row['eti']['threshold']} within the budget; it is "
        f"the primary identification number because the median first "
        f"crossing is defined only on blocks that cross. "
        f"Median ep. is the first-crossing median with non-crossing "
        f"blocks right-censored at the "
        f"{any_row['eti']['n_episodes_budget']}-episode budget, counting "
        f"from one, with censored quartiles; a quantile the survival curve "
        f"never reaches within the budget prints as ``$>n$''. "
        f"It is the Kaplan-Meier quantile, i.e. the earliest episode at "
        f"which the crossing fraction reaches the quantile, so it differs "
        f"from an interpolated percentile over crossing blocks alone even "
        f"where every block crosses. "
        f"Bold marks the best learning agent per column; Oracle and Random "
        f"are reported as bounds and excluded from bold. "
    )
    suppressed = [PLANNER_LABELS[n] for n in NO_IDENTIFICATION if n in rows]
    if suppressed:
        caption += (
            f"The three identification columns are omitted for "
            f"{' and '.join(suppressed)} (``{NOT_APPLICABLE_CELL}''), which "
            f"is a reward bound and not an identification baseline: it plans "
            f"on a one-hot at $\\theta^\\star$, so the mass its filter logs "
            f"measures only what a Bayesian filter learns from watching "
            f"purely exploitative behaviour. The identification panels of "
            f"the figures exclude it for the same reason. Random is kept in "
            f"both, as a meaningful passive-identification floor. "
        )
    if "oracle" in rows and "aif" in rows:
        # Normalized first, so a gap inside half a display unit of zero
        # cannot print as a signed zero (see `_two_dp`).
        gap = round(rows["oracle"]["final"] - rows["aif"]["final"], 2) + 0.0
        caption += (
            f"Oracle's final window is {_two_dp(rows['oracle']['final'])}, "
            f"i.e. {gap:+.2f} against AIF-MP. "
        )
    if bench.family:
        caption += f"Note that {bench.family}. "
    caption += (
        f"Generated by {GENERATED_BY} from {tt('output/' + bench.dirname)} "
        f"over {any_row['n_seeds']} blocks of "
        f"{any_row['n_episodes']} episodes."
        + _effective_n_clause([
            (
                PLANNER_LABELS[name], rows[name],
                ("early_n", "final_n") if name in NO_IDENTIFICATION
                else ("early_n", "final_n", "mass_n"),
            )
            for name in (*LEARNING_SLATE, *REFERENCE_SLATE) if name in rows
        ])
    )
    return _table(body, "lccccc", caption, bench.tex_label)


def timing_table(all_runs: dict[str, dict[str, dict]]) -> str:
    body = [
        "Method & " + " & ".join(b.label for b in BENCHMARKS) + r" \\",
        r"\midrule",
    ]
    for name in TIMING_ORDER:
        cells = []
        for bench in BENCHMARKS:
            run = all_runs[bench.dirname].get(name)
            if run is None:
                # No result JSON on disk. Whether that is because the
                # planner is unimplemented on this benchmark or because
                # the run is missing is not visible from output/, so the
                # two are not distinguished in the caption either.
                cells.append(NO_RESULT_CELL)
                continue
            cells.append(fmt_time(per_decision_planning_time(
                run["per_episode"]["planning_time_s"],
                run["per_episode"]["n_steps"],
            )))
        if set(cells) <= {NO_RESULT_CELL, "--"}:
            continue
        body.append(" & ".join([PLANNER_LABELS[name], *cells]) + r" \\")
    caption = (
        "Planning time per decision in seconds, over all decisions of all "
        "blocks. \\texttt{planning\\_time\\_s} in the result JSONs is an "
        "episode total, so each cell is "
        "$\\sum \\text{time} / \\sum \\text{steps}$ over the block "
        "$\\times$ episode grid. "
        "Message-passing planners (AIF-MP, VBP, PSRL, Oracle) ran on one "
        "NVIDIA A100 GPU; sample-based planners ran on AMD EPYC CPU cores, "
        "so columns are not matched-hardware comparisons. "
        f"``{NO_RESULT_CELL}'' marks a planner with no result JSON under "
        f"{tt('output/')} for that benchmark, which covers both a planner "
        f"not implemented there and a run that was never done -- the two "
        f"are indistinguishable from {tt('output/')}, and the emitter names "
        f"each absent file on stderr. ``--'' marks a run that is present "
        f"but carries no finite planning time. "
        f"Generated by {GENERATED_BY}."
    )
    return _table(body, "l" + "c" * len(BENCHMARKS), caption,
                  "tab:appx-timing")


def deferred_table(all_runs: dict[str, dict[str, dict]]) -> str:
    body = ["Environment & " + _SCALAR_HEADER, r"\midrule"]
    entries: list[tuple[str, dict, tuple[str, ...]]] = []
    first = True
    for bench in BENCHMARKS:
        rows = {
            name: scalar_row(run, bench)
            for name, run in all_runs[bench.dirname].items()
            if name in DEFERRED_SLATE
        }
        if not rows:
            continue
        if not first:
            body.append(r"\midrule")
        first = False
        for i, name in enumerate(n for n in DEFERRED_SLATE if n in rows):
            entries.append((
                f"{bench.label} {PLANNER_LABELS[name]}", rows[name],
                ("early_n", "final_n", "mass_n"),
            ))
            prefix = bench.label if i == 0 else ""
            body.append(f"{prefix} & " + " & ".join([
                PLANNER_LABELS[name],
                fmt_pm(rows[name]["early"], rows[name]["early_ci"]),
                fmt_pm(rows[name]["final"], rows[name]["final_ci"]),
                fmt_pm(rows[name]["mass"], rows[name]["mass_ci"]),
                fmt_frac(rows[name]["eti"]["fraction_identified"]),
                fmt_censored_median(rows[name]["eti"]),
            ]) + r" \\")
    caption = (
        "Scalar summaries for the deferred ablation baselines, under the "
        "conventions of the per-benchmark scalar tables. Returns are raw "
        "on the Door-Code variants and normalized on Multi-Door Tiger and "
        "RockSample-derived. No bold: these rows are not in competition "
        f"with the main slate. Generated by {GENERATED_BY}."
        + _effective_n_clause(entries)
    )
    return _table(body, "ll" + "c" * 5, caption, "tab:appx-deferred")


def simscaling_table(input_dir: Path) -> str | None:
    sweep_dir = input_dir / "door_code_simscaling"
    cells: list[tuple[int, dict]] = []
    for child in sorted(sweep_dir.glob("sims*")) if sweep_dir.is_dir() else []:
        path = child / "bamcp.json"
        if not path.exists():
            continue
        try:
            sims = int(child.name[len("sims"):])
        except ValueError:
            continue
        with path.open() as f:
            cells.append((sims, json.load(f)))
    if not cells:
        print(
            f"warning: no sims<N>/bamcp.json under {sweep_dir} — skipping "
            f"the simulation-budget table",
            file=sys.stderr,
        )
        return None
    bench = BENCHMARKS[0]  # the sweep is door_code, same metric and windows
    body = [
        r"Simulations & Early & Final & Final mass & Identified & "
        r"Median ep. (censored) \\",
        r"\midrule",
    ]
    entries: list[tuple[str, dict, tuple[str, ...]]] = []
    for sims, run in sorted(cells):
        r = scalar_row(run, bench)
        label = f"${sims:,}$".replace(",", "\\,")
        entries.append(
            (f"{label} simulations", r, ("early_n", "final_n", "mass_n"))
        )
        body.append(" & ".join([
            label,
            fmt_pm(r["early"], r["early_ci"]),
            fmt_pm(r["final"], r["final_ci"]),
            fmt_pm(r["mass"], r["mass_ci"]),
            fmt_frac(r["eti"]["fraction_identified"]),
            fmt_censored_median(r["eti"]),
        ]) + r" \\")
    n_seeds = sorted(cells)[0][1]["n_seeds"]
    caption = (
        "BAMCP on Door-Code under a simulation-budget sweep "
        f"(${n_seeds}$ blocks per budget, all other settings as in the "
        "main experiment), under the conventions of the per-benchmark "
        f"scalar tables. Generated by {GENERATED_BY}."
        + _effective_n_clause(entries)
    )
    return _table(body, "cccccc", caption, "tab:appx-simscaling")


# --------------------------------------------------------------------------
# exact Bayes-adaptive reference


# The reduced instances `scripts/run_exact_reference.py` solves exactly,
# in table order. These are not the headline configurations, so the row
# label alone would read as the main slate's K -- the reduction is stated
# per row in the caption by `_instance_clause` instead.
_EXACT_INSTANCES = (
    ("door_code", "Door-Code"),
    ("factored_tiger", "Multi-Door Tiger"),
)

# Arms the block table reports against the exact band, in column order.
_EXACT_COMPARED = ("aif", "bamcp")


def load_exact_reference(input_dir: Path) -> list[tuple[str, dict]]:
    """Load each reduced instance's reference.json, warning about absent ones.

    Returns [(label, payload)] in `_EXACT_INSTANCES` order. Empty when the
    stages have not been run, which both tables treat as "skip", not as
    "emit an empty table" -- these two stages are deferred from the
    cluster slate (see `cluster/preflight.py`), so an `output/` tree
    without them is a normal state rather than an error.
    """
    cells: list[tuple[str, dict]] = []
    for dirname, label in _EXACT_INSTANCES:
        path = input_dir / "exact_reference" / dirname / "reference.json"
        if not path.exists():
            print(
                f"warning: no {path} — skipping '{label}' in the exact "
                f"Bayes-adaptive reference tables",
                file=sys.stderr,
            )
            continue
        with path.open() as f:
            cells.append((label, json.load(f)))
    return cells


def _fmt_sig3(value: float) -> str:
    """Three significant figures.

    Value regret spans two reward scales across the instances -- Door-Code
    terminal reward is order 1, Multi-Door Tiger's tiger penalty is 100 --
    so the two-decimal convention of the scalar tables would print
    Door-Code's mean regret as `0.00` and assert an exactness the run does
    not show.
    """
    if not np.isfinite(value):
        return "--"
    return f"${float(f'{value:.3g}'):g}$"


def _instance_clause(payload: dict) -> str:
    """Caption clause naming the reduced instance a row was solved on."""
    config = payload["config"]
    return (
        f"$K={config['env_kwargs']['K']}$, "
        f"{config['n_seeds']}$\\times${config['n_episodes']} blocks"
        f"$\\times$episodes at $\\text{{max\\_steps}}={config['max_steps']}$"
    )


def _tolerance_clause(cells: list[tuple[str, dict]]) -> str:
    """Caption clause reporting the optimal-set rate across tie tolerances.

    The headline rate is scored at one relative tolerance, and whether it
    survives the others is exactly the objection the sweep exists to
    answer -- so the range goes in the caption rather than being left in
    the JSON for a reader who will not open it. It is stated per instance
    because the two do not move together: Door-Code is flat across the
    sweep and Multi-Door Tiger is not.
    """
    parts = []
    for label, payload in cells:
        rates = [
            cell["optimal_set_rate"]
            for cell in payload["action_agreement"]
            ["tie_tolerance_sweep"].values()
        ]
        low, high = _two_dp(min(rates)), _two_dp(max(rates))
        # A range whose ends agree at display precision is the flat case,
        # and printing it as `0.96--0.96` reads as a formatting fault
        # rather than as the stability it reports.
        parts.append(
            f"{label} ${low}$" if low == high
            else f"{label} ${low}$--${high}$"
        )
    return (
        " Re-scoring every decision at relative tie tolerances "
        "$10^{-6}$/$10^{-7}$/$10^{-8}$ moves the optimal-set rate over "
        + "; ".join(parts) + "."
    )


def exact_agreement_table(cells: list[tuple[str, dict]]) -> str:
    """Per-decision agreement between AIF-MP and the exact optimal set."""
    body = [
        r"Instance & Decisions & In optimal set & Lowest-index match & "
        r"$|A^\star|$ & Mean regret & Max regret \\",
        r"\midrule",
    ]
    for label, payload in cells:
        agree = payload["action_agreement"]
        body.append(" & ".join([
            label,
            f"${agree['n_decisions']}$",
            fmt_frac(agree["optimal_set_rate"]),
            fmt_frac(agree["lowest_index_match_rate"]),
            fmt_frac(agree["mean_optimal_set_size"]),
            _fmt_sig3(agree["mean_value_regret"]),
            _fmt_sig3(agree["max_value_regret"]),
        ]) + r" \\")
    instances = "; ".join(
        f"{label} ({_instance_clause(payload)})" for label, payload in cells
    )
    caption = (
        "Agreement between AIF-MP and the exact Bayes-adaptive optimum on "
        "reduced instances the optimum is enumerable on (" + instances +
        "). Every quantity is per decision and scored against the exact "
        "action values \\emph{at the belief AIF-MP itself holds}, so none "
        "of it depends on the trajectory an exact arm would have taken or "
        "on how that arm breaks ties. \\emph{In optimal set} is the "
        "fraction of decisions whose AIF-MP action is exactly optimal; "
        "\\emph{lowest-index match} is the fraction matching one "
        "arbitrary member of that set, and the gap between the two "
        "columns is why membership, not equality, is the reported "
        "statistic. $|A^\\star|$ is the mean size of the optimal set. "
        "Regret is exact action value forgone, in that instance's reward "
        "units, and is tolerance-free by construction."
        + _tolerance_clause(cells) +
        f" Generated by {GENERATED_BY}."
    )
    return _table(body, "lcccccc", caption, "tab:appx-exact-agreement")


def exact_block_table(cells: list[tuple[str, dict]]) -> str:
    """Block return against the band the two exact tie-breaks bracket."""
    body = [
        r"Instance & Exact (band) & " +
        " & ".join(PLANNER_LABELS[name] for name in _EXACT_COMPARED) +
        r" \\",
        r"\midrule",
    ]
    for label, payload in cells:
        block = payload["block"]
        returns, sems = block["normalised_return"], block["normalised_return_sem"]
        exact = [
            returns[name] for name in payload["config"]["exact_tie_breaks"]
        ]
        row = [label, f"$[{_two_dp(min(exact))}, {_two_dp(max(exact))}]$"]
        row += [
            fmt_pm(returns[name], 1.96 * sems[name])
            if name in returns else NO_RESULT_CELL
            for name in _EXACT_COMPARED
        ]
        body.append(" & ".join(row) + r" \\")
    caption = (
        "Block return on the reduced instances of "
        "Table~\\ref{tab:appx-exact-agreement}, as mean $\\pm 1.96$SEM "
        "across blocks. The exact column is a band, not a number: optimal "
        "actions tie structurally on both instances (every probe station "
        "on Door-Code, both listen actions under a symmetric prior on "
        "Multi-Door Tiger), and while every tied action is equally optimal "
        "\\emph{for the episode} the choice decides which bit of $\\theta$ "
        "is learned and so carries into the next episode's prior. The band "
        "is what the lowest- and highest-index tie-breaks bracket. "
        "\\textbf{This is a descriptive comparison of belief "
        "trajectories, not an optimality ratio}: an episode-scoped optimum "
        "never buys information that repays only in a later episode, so "
        "the true block optimum lies above both ends of the band and the "
        "AIF-MP gap read off this table is an underestimate. Multi-Door "
        "Tiger's normalization denominator varies by block. Episode-0 "
        "return, the one slice where a value gap against the optimum is "
        "well posed, is not reported here: at these block counts its "
        "standard errors exceed the gap. "
        f"Generated by {GENERATED_BY}."
    )
    return _table(
        body, "lc" + "c" * len(_EXACT_COMPARED),
        caption, "tab:appx-exact-block",
    )


# --------------------------------------------------------------------------
# pairing (reviewer item 8)


# Mechanism only: every claim here is a property of the code, at the
# named call sites. What the canonical runs actually did is measured in
# `_pairing_conclusion` instead of asserted here.
_PAIRING_MECHANISM = r"""\paragraph{What is and is not paired across methods.}
Every block is one draw of $\theta^\star$ from a dedicated RNG consumed
for nothing else (\texttt{run\_outer\_loop} in
\texttt{door\_code\_runner.py}), and the same
\texttt{-{}-seed} is passed to every planner, so a shared shard layout
makes $\theta^\star$ identical seed-for-seed across methods.
Shard layout is what breaks that: \texttt{shard\_seed = seed +
shard\_index} $\times$ \texttt{1\_000\_003}
(\texttt{run\_door\_code.py}) re-seeds per shard, so two submissions of
one benchmark with different shard counts draw different (but
identically distributed) $\theta^\star$ sequences.
Environment randomness is common random numbers only at the seed level:
\texttt{obs\_seed} is a deterministic function of the outer seed
(the same loop) but probe bit-flips are drawn lazily
(\texttt{DoorCodeEnv.\_make\_obs}), so two planners consume the
same stream at different offsets and their noise realizations are not
step-matched.
Which of the canonical runs share a layout is measured, not assumed:
"""


def _pairing_conclusion(results: list[dict]) -> str:
    r"""The pairing verdict, derived from the measured overlaps.

    Asserting a verdict in the prose would be the same defect as a
    hand-typed literal, relocated: whether blocks are paired is a
    property of whichever runs happen to be in `output/`, so a rerun
    that unified the shard layouts would fill the table below with real
    paired deltas while a hard-coded paragraph went on calling them
    unpaired.
    """
    n = len(results)
    if not n:
        return "no comparison was available.\n"
    n_paired = sum(1 for r in results if r["paired"])
    if n_paired == n:
        verdict = (
            f"all {n} comparisons below share $\\theta^\\star$ at every "
            f"seed index, so blocks \\textbf{{are paired across methods}} "
            f"and every $\\Delta$ below is a per-block paired difference"
        )
    elif n_paired == 0:
        verdict = (
            f"none of the {n} comparisons below shares $\\theta^\\star$ at "
            f"every seed index, so blocks are \\textbf{{not paired across "
            f"methods}} and no paired $\\Delta$ is reportable"
        )
    else:
        verdict = (
            f"{n_paired} of the {n} comparisons below share "
            f"$\\theta^\\star$ at every seed index and carry a paired "
            f"$\\Delta$; the other {n - n_paired} do not and are marked "
            f"``unpaired'', so pairing \\textbf{{holds only within}} the "
            f"groups of runs that shared a shard layout"
        )
    return (
        verdict + ". The confidence intervals in the scalar tables are "
        "unpaired either way: each is computed from one method's blocks "
        "alone.\n"
    )


def pairing_block(all_runs: dict[str, dict[str, dict]]) -> str:
    results: list[dict] = []
    body = [
        r"Environment & Comparison & Shared $\theta^\star$ & "
        r"Paired $\Delta$ early & Paired $\Delta$ final \\",
        r"\midrule",
    ]
    first = True
    for bench in BENCHMARKS:
        runs = all_runs[bench.dirname]
        if "aif" not in runs:
            continue
        base = runs["aif"]
        others = [
            n for n in (*LEARNING_SLATE, *REFERENCE_SLATE, *DEFERRED_SLATE)
            if n != "aif" and n in runs
        ]
        if not others:
            continue
        if not first:
            body.append(r"\midrule")
        first = False
        for i, name in enumerate(others):
            pairing = theta_pairing(base["theta_trues"], runs[name]["theta_trues"])
            results.append(pairing)
            shared = (
                f"${pairing['n_shared_theta']}/{pairing['n_compared']}$"
            )
            if pairing["paired"]:
                de = paired_window_difference(
                    base["per_episode"][bench.metric],
                    runs[name]["per_episode"][bench.metric],
                    bench.n_early, "early",
                )
                df = paired_window_difference(
                    base["per_episode"][bench.metric],
                    runs[name]["per_episode"][bench.metric],
                    bench.n_final, "final",
                )
                cells = [fmt_pm(de[0], de[1]), fmt_pm(df[0], df[1])]
            else:
                cells = ["unpaired", "unpaired"]
            prefix = bench.label if i == 0 else ""
            body.append(" & ".join([
                prefix,
                f"AIF-MP $-$ {PLANNER_LABELS[name]}",
                shared, *cells,
            ]) + r" \\")
    caption = (
        "Per-block paired differences of the window means, AIF-MP minus "
        "each other method, with 95\\% confidence intervals of the paired "
        "mean. ``Shared $\\theta^\\star$'' counts seed indices at which "
        "the two runs drew the same true parameter; a paired difference is "
        "reported only where every index matches, and pairs on "
        "$\\theta^\\star$ alone -- environment noise is not step-matched. "
        "A block whose window is undefined under either method drops out "
        "of that difference. "
        f"Generated by {GENERATED_BY}."
    )
    return "\n".join([
        _PAIRING_MECHANISM + _pairing_conclusion(results),
        _table(body, "llccc", caption, "tab:appx-pairing"),
    ])


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the paper's tables as .tex from output/.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output"),
        help="directory holding the per-benchmark result subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/tables"),
        help="directory to write the .tex files into",
    )
    args = parser.parse_args()

    slate = (*LEARNING_SLATE, *REFERENCE_SLATE, *DEFERRED_SLATE)
    all_runs: dict[str, dict[str, dict]] = {}
    for bench in BENCHMARKS:
        env_dir = args.input_dir / bench.dirname
        if not env_dir.is_dir():
            raise FileNotFoundError(
                f"no {env_dir}; run `dvc repro {bench.dirname}` first"
            )
        all_runs[bench.dirname] = load_runs(env_dir, slate)
        if not all_runs[bench.dirname]:
            raise FileNotFoundError(
                f"no planner results in {env_dir}; "
                f"run `dvc repro {bench.dirname}` first"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def write(name: str, text: str) -> None:
        path = args.output_dir / name
        path.write_text(f"{GENERATED_COMMENT}\n{text}")
        written.append(path)

    for bench in BENCHMARKS:
        write(
            f"scalars_{bench.dirname}.tex",
            scalar_table(bench, all_runs[bench.dirname]),
        )
    write("timing.tex", timing_table(all_runs))
    write("deferred_baselines.tex", deferred_table(all_runs))
    sims = simscaling_table(args.input_dir)
    if sims is not None:
        write("simscaling.tex", sims)
    exact_cells = load_exact_reference(args.input_dir)
    if exact_cells:
        write("exact_agreement.tex", exact_agreement_table(exact_cells))
        write("exact_block.tex", exact_block_table(exact_cells))
    write("pairing.tex", pairing_block(all_runs))

    for path in written:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
