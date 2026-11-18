"""Plot the door-code goal-prior preference ablation (AIF only).

Reads output/door_code_preference/succ<S>_fail<F>/aif.json for the
(success_logit, failure_logit) grid in params.door_code_preference and
renders one heatmap per metric:

  1. Early terminal reward (mean over the first --early-window episodes)
  2. Early probe visits (same window)
  3. Final posterior mass on theta_true (last episode)
  4. Median episodes-to-identify (mass >= 0.9), with the fraction of
     seeds that crossed in parentheses; blank = never crossed

The success logit sets how strongly SUCCESS attracts the planner, the
failure logit how strongly FAILURE repels it; the probe/commit boundary
should track their gap. The canonical (4, -4) cell is outlined.

Reference levels
----------------
Right of the gap, every panel carries one constant column per canonical
goal-prior-independent baseline (_REFERENCE_PLANNERS, read from
--reference-dir). Those planners consume the env reward table and never
read C, so one canonical number is their level across the whole sweep —
which is what turns this figure from "how sensitive is AIF-MP to its own
goal prior?" into "does AIF-MP still beat the baselines at the
reward-matched cell?". They cost nothing to add: the numbers are already
on disk from the canonical door_code stage. A cell is bolded when it beats
every reference level on that panel's metric.

House convention (shared with plot_factored_tiger_preference.py): the row
axis runs from the mildest penalty at the top to the harshest at the
bottom. door_code's failure_logit is negative, so that is descending
order — the opposite sort from Factored Tiger, whose bad_logit is a
positive magnitude.

Usage:
    uv run python scripts/plot_door_code_preference.py \\
        --input-dir output/door_code_preference \\
        --output output/door_code_preference/figure.pdf
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Repo-root dissertation theme (single source of truth for figure colors).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402

import _preference as pref  # noqa: E402
from _cards import HEATMAP_CARD_SIZE_IN, save_panel_cards  # noqa: E402
from learningpomdp.experiment.door_code_runner import episodes_to_identify  # noqa: E402

# Float-tolerant: DVC formats each cell directory from the params value, so
# a `4.0` in params.yaml yields `succ4.0_fail-4.0`. An `\d+`-only pattern
# would skip that directory and silently drop the cell from the figure.
_CELL_RE = re.compile(r"^succ(-?\d+(?:\.\d+)?)_fail(-?\d+(?:\.\d+)?)$")

# Canonical baselines drawn as reference levels. Each plans on the env
# reward table and ignores C (PSRL/Oracle read it via VBP, so a flat level
# would be wrong for them); Random is left out because a uniform-action
# floor is not a level the cells have to clear, and POMCP / Belief-Q because
# plot.paper_slate keeps the ablation-only planners out of the results
# figures.
_REFERENCE_PLANNERS = ("bamcp", "boss", "ids")


def _discover_cells(input_dir: Path) -> dict[tuple[float, float], Path]:
    """Map (success_logit, failure_logit) -> aif.json path."""
    cells: dict[tuple[float, float], Path] = {}
    for child in input_dir.iterdir():
        m = _CELL_RE.match(child.name)
        if child.is_dir() and m:
            cells[(float(m.group(1)), float(m.group(2)))] = child / "aif.json"
    return cells


def _cell_metrics(summary: dict, early_window: int) -> dict[str, float]:
    reward = np.asarray(summary["per_episode"]["terminal_reward"], dtype=float)
    probes = np.asarray(summary["per_episode"]["probe_visits"], dtype=float)
    mass = np.asarray(
        summary["per_episode"]["posterior_mass_on_true_theta"], dtype=float,
    )
    # Right-censored median crossing episode, counted from one: the
    # convention of the scalar tables (make_tables.scalar_row). The runner's
    # `derived` block holds the zero-indexed median over crossing blocks
    # only, which reads low against the tables.
    ident = episodes_to_identify(mass)
    censored = ident["censored_median"]
    return {
        "early_reward": float(reward[:, :early_window].mean()),
        "early_probes": float(probes[:, :early_window].mean()),
        "final_mass": float(mass[:, -1].mean()),
        "identify_median": (
            float(censored) + 1.0 if censored is not None else np.nan
        ),
        "identify_fraction": float(ident["fraction_identified"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot door-code goal-prior preference ablation.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/door_code_preference"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("output/door_code"),
        help="canonical run directory supplying the baseline reference levels",
    )
    parser.add_argument(
        "--early-window", type=int, default=5,
        help="number of leading episodes in the early-window means",
    )
    parser.add_argument(
        "--cards-dir", type=Path, default=None,
        help="if set, also write each heatmap as its own PDF card here",
    )
    args = parser.parse_args()
    theme.apply_rcparams()
    metrics_fn = partial(_cell_metrics, early_window=args.early_window)

    cells = _discover_cells(args.input_dir)
    if not cells:
        raise FileNotFoundError(
            f"no succ<S>_fail<F> subdirectories under {args.input_dir}; "
            f"run `dvc repro door_code_preference` first"
        )
    params = pref.load_params()
    sweep = params["door_code_preference"]
    pref.assert_cells_match(
        cells,
        (
            (float(s), float(f)) for s, f in
            itertools.product(sweep["success_logits"], sweep["failure_logits"])
        ),
        args.input_dir,
    )
    canonical_cell = (
        float(params["env"]["success_logit"]),
        float(params["env"]["failure_logit"]),
    )
    succ_values = sorted({s for s, _ in cells})
    # House convention: mildest penalty on top. failure_logit is negative
    # here, so mildest = largest = descending order.
    fail_values = sorted({f for _, f in cells}, reverse=True)

    summaries: dict[tuple[float, float], dict] = {}
    for key, path in cells.items():
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run `dvc repro door_code_preference` first"
            )
        with path.open() as f:
            summaries[key] = json.load(f)

    references = pref.load_references(
        args.reference_dir, _REFERENCE_PLANNERS, metrics_fn,
    )

    n_rows, n_cols = len(fail_values), len(succ_values)
    grids = {
        name: np.full((n_rows, n_cols), np.nan)
        for name in (
            "early_reward", "early_probes", "final_mass",
            "identify_median", "identify_fraction",
        )
    }
    for (succ, fail), summary in summaries.items():
        i = fail_values.index(fail)
        j = succ_values.index(succ)
        for name, value in metrics_fn(summary).items():
            grids[name][i, j] = value

    cfg = next(iter(summaries.values()))["config"]
    # Themed ramps: bad -> good diverges terracotta -> teal; the sequential
    # panels ride the teal path ramp (reversed where lower is better).
    # `better` drives both the ramp direction and the bolding.
    seq_cmap = theme.path_cmap()
    panels = [
        ("early_reward", f"Early return (episodes 0-{args.early_window - 1})",
         theme.diverging_cmap(), "{:.2f}", "high"),
        ("early_probes", f"Early probe visits (episodes 0-{args.early_window - 1})",
         seq_cmap, "{:.1f}", None),
        ("final_mass", "Final posterior mass on $\\theta^\\star$",
         seq_cmap, "{:.2f}", "high"),
        ("identify_median", "Median episodes to identify (fraction)",
         seq_cmap.reversed(), "{:.1f}", "low"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.0))
    title = (
        f"Door-code goal-prior preference ablation (AIF) — "
        f"K={cfg['env']['K']}, L={cfg['env']['lock_length']}, "
        f"probe_noise={cfg['env']['probe_noise']}, "
        f"n_seeds={cfg['n_seeds']}, n_episodes={cfg['n_episodes']}"
    )
    fig.suptitle(title, fontsize=11)

    ink = theme.PALETTE["ink"]
    n_plot_cols = n_cols + (1 + len(references) if references else 0)
    x_labels = (
        [pref.format_logit(s) for s in succ_values]
        + pref.reference_tick_labels(references)
    )
    fractions = pref.append_reference_columns(
        grids["identify_fraction"], references, "identify_fraction",
    )
    card_axes = []
    for ax, (name, panel_title, cmap, fmt, better) in zip(axes, panels):
        grid = pref.append_reference_columns(grids[name], references, name)
        vmin, vmax = pref.value_limits(grids[name])
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        for i in range(n_rows):
            for j in range(n_plot_cols):
                if references and j == n_cols:
                    continue  # spacer column
                if np.isnan(grid[i, j]):
                    label = "--"
                elif name == "identify_median":
                    label = f"{fmt.format(grid[i, j])}\n({fractions[i, j]:.2f})"
                else:
                    label = fmt.format(grid[i, j])
                bold = j < n_cols and pref.beats_all(
                    grid[i, j], references, name, better,
                )
                ax.text(
                    j, i, label, ha="center", va="center", fontsize=9,
                    fontweight="bold" if bold else "normal",
                )
        if references:
            ax.axvline(n_cols, color=ink, lw=0.8, alpha=0.5)
        if canonical_cell[0] in succ_values and canonical_cell[1] in fail_values:
            j = succ_values.index(canonical_cell[0])
            i = fail_values.index(canonical_cell[1])
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor=ink, lw=2,
            ))
        ax.set_xticks(range(n_plot_cols), x_labels)
        for tick, ref in zip(ax.get_xticklabels()[n_cols + 1:], references):
            tick.set(rotation=30, ha="right", fontsize=8, color=ref.color)
        ax.set_yticks(range(n_rows), [pref.format_logit(f) for f in fail_values])
        ax.set_xlabel("success logit")
        ax.set_ylabel("failure logit")
        ax.set_title(panel_title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        card_axes.append((name, (ax, cbar.ax)))

    fig.text(
        0.5, 0.004,
        "Right of the gap: canonical baselines as reference levels. They plan "
        "on the env reward table and never read C, so one value is their "
        "level across every cell (flat columns by construction). Bold = the "
        "cell beats every reference level on that panel's metric. The colour "
        "scale spans the swept cells only, so a reference column outside that "
        "range clips to the end of the scale; read those off the annotated "
        "value.\n"
        + pref.seed_caveat(int(cfg["n_seeds"]), references),
        ha="center", va="bottom", fontsize=8, color=ink,
    )

    plt.tight_layout(rect=(0, 0.09, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"figure saved to {args.output}", flush=True)

    if args.cards_dir is not None:
        save_panel_cards(
            fig, card_axes, args.cards_dir, "door_code_preference",
            size=HEATMAP_CARD_SIZE_IN,
        )
        print(f"cards saved to {args.cards_dir}", flush=True)

    plt.close(fig)


if __name__ == "__main__":
    main()
