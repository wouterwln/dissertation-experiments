"""Shared pieces of the three goal-prior preference figures.

Each sweep varies only AIF's *own* goal prior and plots AIF's metrics
against it, which on its own answers "how sensitive is AIF-MP to C?" — a
sensitivity analysis. The reviewer's question is the other one: "is
AIF-MP's advantage an artifact of a favourable goal prior?", i.e. does
AIF-MP still beat the baselines at the reward-matched cell.

That second question costs nothing extra to answer. The baselines here
are goal-prior *independent*: they plan on the env reward table the
runners precompute and never read C, so one canonical number per
baseline is the correct level across every cell of every sweep, and it is
already on disk from the canonical stage. This module loads those
numbers and draws them as reference levels — horizontal lines on the
value axis of a line panel, and constant reference columns beside a
heatmap (a heatmap has no value axis, and a column that is flat down the
sweep axis is itself the statement "this baseline does not read C").

Deliberately excluded from the reference slate: PSRL and Oracle (both
wrap VBP and therefore *do* read C, so their canonical numbers are not
constant along a sweep axis and a flat level would be a lie), and VBP
itself.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.lines as mlines
import numpy as np

# Repo-root dissertation theme (single source of truth for figure colors).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        REPO_ROOT = _p
        break
import dissertation_theme as theme  # noqa: E402

# Planners whose action selection never reads the goal prior C, so one
# canonical value is a valid level across a whole preference sweep.
# PSRL / Oracle / VBP are absent on purpose — they read C via VBP.
C_INDEPENDENT_PLANNERS = frozenset({
    "bamcp", "pomcp", "factored_bamcp", "boss", "ids", "belief_q", "random",
})

PLANNER_LABELS = {
    "bamcp": "BAMCP",
    "pomcp": "POMCP (literal)",
    "factored_bamcp": "Factored BA-POMCP",
    "boss": "BOSS",
    "ids": "IDS",
    "belief_q": "Belief-Q",
    "random": "Random",
}
# Tick-sized labels for the heatmap reference columns.
PLANNER_TICK_LABELS = {
    "pomcp": "POMCP",
    "factored_bamcp": "F-BAMCP",
    "belief_q": "BeliefQ",
}


@dataclass(frozen=True)
class Reference:
    """One goal-prior-independent baseline's canonical metrics."""

    planner: str
    n_seeds: int
    metrics: dict[str, float]

    @property
    def label(self) -> str:
        return PLANNER_LABELS.get(self.planner, self.planner)

    @property
    def tick_label(self) -> str:
        return PLANNER_TICK_LABELS.get(self.planner, self.label)

    @property
    def color(self):
        return theme.planner_color(self.planner)

    @property
    def linestyle(self) -> str:
        return theme.planner_linestyle(self.planner)


def load_references(
    reference_dir: Path,
    planners: Sequence[str],
    metrics_fn: Callable[[dict], dict[str, float]],
) -> list[Reference]:
    """Load ``reference_dir/<planner>.json`` and score it exactly like a
    swept cell, so levels and cells are the same quantity by construction."""
    unknown = set(planners) - C_INDEPENDENT_PLANNERS
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} read the goal prior C (or are unknown), so a "
            f"single canonical value is not a valid level across a preference "
            f"sweep; reference planners must be drawn from "
            f"{sorted(C_INDEPENDENT_PLANNERS)}"
        )
    references: list[Reference] = []
    for planner in planners:
        path = Path(reference_dir) / f"{planner}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"missing reference run {path}; the preference figure draws "
                f"the canonical baselines as reference levels, so the "
                f"canonical stage has to have run first"
            )
        with path.open() as f:
            summary = json.load(f)
        references.append(Reference(
            planner=planner,
            n_seeds=int(summary["n_seeds"]),
            metrics=metrics_fn(summary),
        ))
    return references


def reference_values(references: Iterable[Reference], metric: str) -> list[float]:
    """Finite values of ``metric`` across the reference slate."""
    return [
        r.metrics[metric] for r in references
        if np.isfinite(r.metrics[metric])
    ]


def draw_level_lines(ax, references: Iterable[Reference], metric: str) -> None:
    """Horizontal reference level per baseline on a value-axis panel."""
    for ref in references:
        value = ref.metrics[metric]
        if not np.isfinite(value):
            continue
        ax.axhline(
            value, color=ref.color, linestyle=ref.linestyle, lw=1.3,
            alpha=0.9, zorder=0,
        )


def append_reference_columns(
    grid: np.ndarray, references: Sequence[Reference], metric: str,
) -> np.ndarray:
    """``(n_rows, n_cols)`` -> ``(n_rows, n_cols + 1 + n_refs)``.

    One all-NaN spacer column, then one constant column per reference. The
    columns are constant by construction: these baselines never read C.
    """
    grid = np.asarray(grid, dtype=float)
    n_rows = grid.shape[0]
    spacer = np.full((n_rows, 1), np.nan)
    columns = [
        np.full((n_rows, 1), references[i].metrics[metric])
        for i in range(len(references))
    ]
    return np.hstack([grid, spacer, *columns]) if references else grid


def reference_tick_labels(references: Sequence[Reference]) -> list[str]:
    """Tick labels for the spacer + reference columns."""
    return ["", *(ref.tick_label for ref in references)] if references else []


def value_limits(grid: np.ndarray) -> tuple[float, float]:
    """(vmin, vmax) spanning the swept cells only.

    The reference columns are drawn under this same scale and saturate when
    a baseline falls outside it. Stretching the scale to reach them is what
    the earlier version did, and one distant baseline then flattens the
    cells: BOSS at -6.10 on a return panel whose cells span ~0.2 leaves
    every cell the same colour, which defeats the heatmap. Cell-vs-level
    comparability is carried by the annotated number and the
    direction-aware bolding (`beats_all`), neither of which reads the hue.
    """
    values = [v for v in np.asarray(grid, dtype=float).ravel() if np.isfinite(v)]
    if not values:
        return (0.0, 1.0)
    return (float(min(values)), float(max(values)))


def beats_all(
    value: float, references: Iterable[Reference], metric: str, better: str,
) -> bool:
    """Does ``value`` beat every reference level on this metric?

    ``better`` is "high" or "low"; anything else (a metric with no
    better/worse direction, e.g. an action count) returns False.
    """
    levels = reference_values(references, metric)
    if not np.isfinite(value) or not levels:
        return False
    if better == "high":
        return value > max(levels)
    if better == "low":
        return value < min(levels)
    return False


def legend_handles(references: Iterable[Reference]) -> list[mlines.Line2D]:
    """Legend proxies for the reference levels."""
    return [
        mlines.Line2D(
            [], [], color=ref.color, linestyle=ref.linestyle, lw=1.6,
            label=f"{ref.label} (canonical, ignores C)",
        )
        for ref in references
    ]


def seed_caveat(sweep_n_seeds: int, references: Sequence[Reference]) -> str:
    """Caption note on comparing swept cells against reference levels.

    The sweeps run at reduced seed count, so a cell-vs-level comparison
    carries a wider standard error than a within-canonical-figure one.
    """
    if not references:
        return f"Swept cells run at n_seeds={sweep_n_seeds}."
    ref_seeds = sorted({r.n_seeds for r in references})
    inflation = np.sqrt(max(ref_seeds) / sweep_n_seeds)
    seeds = ref_seeds[0] if len(ref_seeds) == 1 else (
        f"{ref_seeds[0]}-{ref_seeds[-1]}"
    )
    return (
        f"Reference levels are the canonical runs (n_seeds={seeds}); swept "
        f"cells run at n_seeds={sweep_n_seeds}, i.e. ~{inflation:.1f}x the "
        f"standard error — read cell-vs-level gaps accordingly."
    )


def assert_cells_match(
    discovered: Iterable[tuple], expected: Iterable[tuple], input_dir: Path,
) -> None:
    """Fail loudly when the rendered cell set is not the params grid.

    Guards the silent-hole failure mode: a cell directory whose name the
    plot script's regex does not match (a float-formatted ``good2.0_bad4.0``,
    say) would otherwise just disappear from the figure.
    """
    discovered, expected = set(discovered), set(expected)
    missing = sorted(expected - discovered)
    extra = sorted(discovered - expected)
    if missing or extra:
        raise ValueError(
            f"cell directories under {input_dir} do not match the params "
            f"grid: missing={missing} unexpected={extra}"
        )


def load_params() -> dict:
    """Parse the repo-root ``params.yaml`` (the sweep grids live there)."""
    import yaml

    with (REPO_ROOT / "params.yaml").open() as f:
        return yaml.safe_load(f)


def format_logit(value: float) -> str:
    """Tick label for a swept logit: ``2`` not ``2.0``, ``2.5`` kept."""
    return f"{value:g}"
