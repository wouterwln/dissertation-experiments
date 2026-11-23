"""Plot the Factored Tiger goal-prior preference ablation (AIF only).

Reads output/factored_tiger_preference/good<G>_bad<B>/aif.json for the
(good_logit, bad_logit) grid in params.factored_tiger_preference and
renders one heatmap per metric:

  1. Early normalised return (mean over the first --early-window episodes)
  2. Early LISTEN actions (same window) — information gathering
  3. Early tiger-door opens (same window) — blind commitment, the
     behaviour the asymmetric prior is hand-set to suppress
  4. Final posterior mass on the persistent theta_true (last episode)
  5. Median episodes-to-identify (mass >= 0.9), with the fraction of
     seeds that crossed in parentheses; blank = never crossed

good_logit rewards opening a safe door, bad_logit penalises opening a
tiger door; the LISTEN-before-OPEN boundary should track their ratio. The
canonical (2, 4) cell — ratio 1:2 — is outlined. The (good=1, bad=10)
cell reaches the *environment's* own reward ratio,
open_safe_reward : open_tiger_penalty = 1:10; bad_logit stops there
because larger values underflow the float32 safe_log floor (see
params.yaml's factored_tiger_preference note).

Reference levels
----------------
Right of the gap, every panel carries one constant column per canonical
goal-prior-independent baseline (_REFERENCE_PLANNERS, read from
--reference-dir). Those planners consume the env reward table and never
read C, so one canonical number is their level across the whole sweep —
which is what turns this figure from "how sensitive is AIF-MP to its own
goal prior?" into "does AIF-MP still beat the baselines at the
reward-matched cell?". They cost nothing to add: the numbers are already
on disk from the canonical factored_tiger stage. A cell is bolded when it
beats every reference level on that panel's metric.

House convention (shared with plot_door_code_preference.py): the row axis
runs from the mildest penalty at the top to the harshest at the bottom.
Tiger's bad_logit is a positive magnitude, so that is ascending order —
the opposite sort from door_code, whose failure_logit is negative.

Usage:
    uv run python scripts/plot_factored_tiger_preference.py \\
        --input-dir output/factored_tiger_preference \\
        --output output/factored_tiger_preference/figure.pdf
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

# Float-tolerant: DVC formats each cell directory from the params value, so
# a `2.0` in params.yaml yields `good2.0_bad4.0`. An `\d+`-only pattern
# would skip that directory and silently drop the cell from the figure.
_CELL_RE = re.compile(r"^good(\d+(?:\.\d+)?)_bad(\d+(?:\.\d+)?)$")

# Canonical baselines drawn as reference levels. Each plans on the env
# reward table and ignores C (PSRL/Oracle read it via VBP, so a flat level
# would be wrong for them); Random is left out because a uniform-action
# floor is not a level the cells have to clear.
_REFERENCE_PLANNERS = ("bamcp", "factored_bamcp", "boss", "ids")


def _discover_cells(input_dir: Path) -> dict[tuple[float, float], Path]:
    """Map (good_logit, bad_logit) -> aif.json path."""
    cells: dict[tuple[float, float], Path] = {}
    for child in input_dir.iterdir():
        m = _CELL_RE.match(child.name)
        if child.is_dir() and m:
            cells[(float(m.group(1)), float(m.group(2)))] = child / "aif.json"
    return cells


def _cell_metrics(summary: dict, early_window: int) -> dict[str, float]:
    ep = summary["per_episode"]
    # reward_normalised is NaN on episodes with no safe doors (nothing to
    # divide by), so use nanmean — same convention as the runner curves.
    reward = np.asarray(ep["reward_normalised"], dtype=float)
    listens = np.asarray(ep["n_listens"], dtype=float)
    tiger = np.asarray(ep["n_tiger_opens"], dtype=float)
    mass = np.asarray(ep["posterior_mass_on_persistent_theta"], dtype=float)
    ident = summary["derived"]["episodes_to_identify"]
    return {
        "early_reward": float(np.nanmean(reward[:, :early_window])),
        "early_listens": float(listens[:, :early_window].mean()),
        "early_tiger_opens": float(tiger[:, :early_window].mean()),
        "final_mass": float(mass[:, -1].mean()),
        "identify_median": (
            float(ident["median"]) if ident["median"] is not None else np.nan
        ),
        "identify_fraction": float(ident["fraction_identified"]),
    }


def _listen_cost_note(
    path: Path, metrics_fn, grids, good_values, bad_values, canonical_cell,
) -> str:
    """One-line report of the Katt-2019 ``listen_cost`` arm.

    ``listen_cost`` is the one reward term the goal prior provably cannot
    represent (a state-only C cannot price an *action*), so the honest
    comparison is AIF against itself: same goal prior, same seeds, the
    environment charging per LISTEN as in Katt 2019 against the 0.0 the
    rest of the suite runs at. The runner's ``max_possible_reward`` is
    ``n_safe · open_safe_reward`` and ignores listen cost, so the return
    gap below is the raw price of the listening the planner still chooses.
    """
    with Path(path).open() as f:
        arm = json.load(f)
    listen_cost = float(arm["config"]["env"]["listen_cost"])
    if listen_cost == 0.0:
        raise ValueError(
            f"{path} was run at listen_cost=0.0, but it is meant to be the "
            f"listen_cost > 0 arm"
        )
    m = metrics_fn(arm)
    i = bad_values.index(canonical_cell[1])
    j = good_values.index(canonical_cell[0])
    base = {name: grid[i, j] for name, grid in grids.items()}
    good, bad = (pref.format_logit(v) for v in canonical_cell)
    return (
        f"Katt-2019 arm — listen_cost={listen_cost:g} at the canonical "
        f"({good}, {bad}) cell, the one reward term a state-only goal prior "
        f"cannot encode (n_seeds={arm['n_seeds']}): return "
        f"{m['early_reward']:+.2f} vs {base['early_reward']:+.2f}, LISTENs "
        f"{m['early_listens']:.1f} vs {base['early_listens']:.1f}, tiger "
        f"opens {m['early_tiger_opens']:.2f} vs "
        f"{base['early_tiger_opens']:.2f}, final mass "
        f"{m['final_mass']:.2f} vs {base['final_mass']:.2f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Factored Tiger goal-prior preference ablation.",
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("output/factored_tiger_preference"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("output/factored_tiger"),
        help="canonical run directory supplying the baseline reference levels",
    )
    parser.add_argument(
        "--listen-cost-json", type=Path, default=None,
        help="optional AIF run at Katt 2019's listen_cost=1.0 and the "
             "canonical goal prior; reported as a caption note",
    )
    parser.add_argument(
        "--early-window", type=int, default=2,
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
            f"no good<G>_bad<B> subdirectories under {args.input_dir}; "
            f"run `dvc repro factored_tiger_preference` first"
        )
    params = pref.load_params()
    sweep = params["factored_tiger_preference"]
    pref.assert_cells_match(
        cells,
        (
            (float(g), float(b)) for g, b in
            itertools.product(sweep["good_logits"], sweep["bad_logits"])
        ),
        args.input_dir,
    )
    canonical_cell = (
        float(params["factored_tiger"]["env"]["good_logit"]),
        float(params["factored_tiger"]["env"]["bad_logit"]),
    )
    good_values = sorted({g for g, _ in cells})
    # House convention: mildest penalty on top. bad_logit is a positive
    # magnitude here, so mildest = smallest = ascending order.
    bad_values = sorted({b for _, b in cells})

    summaries: dict[tuple[float, float], dict] = {}
    for key, path in cells.items():
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; "
                f"run `dvc repro factored_tiger_preference` first"
            )
        with path.open() as f:
            summaries[key] = json.load(f)

    references = pref.load_references(
        args.reference_dir, _REFERENCE_PLANNERS, metrics_fn,
    )

    n_rows, n_cols = len(bad_values), len(good_values)
    grids = {
        name: np.full((n_rows, n_cols), np.nan)
        for name in (
            "early_reward", "early_listens", "early_tiger_opens",
            "final_mass", "identify_median", "identify_fraction",
        )
    }
    for (good, bad), summary in summaries.items():
        i = bad_values.index(bad)
        j = good_values.index(good)
        for name, value in metrics_fn(summary).items():
            grids[name][i, j] = value

    cfg = next(iter(summaries.values()))["config"]
    # Themed ramps: return diverges terracotta -> teal around zero; the
    # sequential panels ride the teal path ramp (reversed where lower is
    # better). `better` drives both the ramp direction and the bolding.
    seq_cmap = theme.path_cmap()
    window = f"episodes 0-{args.early_window - 1}"
    panels = [
        ("early_reward", f"Early normalised return ({window})",
         theme.diverging_cmap(), "{:.2f}", "high"),
        ("early_listens", f"Early LISTEN actions ({window})",
         seq_cmap, "{:.1f}", None),
        ("early_tiger_opens", f"Early tiger-door opens ({window})",
         seq_cmap.reversed(), "{:.2f}", "low"),
        ("final_mass", "Final posterior mass on persistent $\\theta^\\star$",
         seq_cmap, "{:.2f}", "high"),
        ("identify_median", "Median episodes to identify (fraction)",
         seq_cmap.reversed(), "{:.1f}", "low"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.0))
    title = (
        f"Factored Tiger goal-prior preference ablation (AIF) — "
        f"K={cfg['env']['K']}, "
        f"listen_accuracy={cfg['env']['listen_accuracy']}, "
        f"reward=+{cfg['env']['open_safe_reward']}/"
        f"-{cfg['env']['open_tiger_penalty']}, "
        f"listen_cost={cfg['env']['listen_cost']}, "
        f"n_seeds={cfg['n_seeds']}, n_episodes={cfg['n_episodes']}"
    )
    fig.suptitle(title, fontsize=11)

    ink = theme.PALETTE["ink"]
    n_plot_cols = n_cols + (1 + len(references) if references else 0)
    x_labels = (
        [pref.format_logit(g) for g in good_values]
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
                    color=pref.annotation_color(im, grid[i, j]),
                )
        if references:
            ax.axvline(n_cols, color=ink, lw=0.8, alpha=0.5)
        if canonical_cell[0] in good_values and canonical_cell[1] in bad_values:
            j = good_values.index(canonical_cell[0])
            i = bad_values.index(canonical_cell[1])
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor=ink, lw=2,
            ))
        ax.set_xticks(range(n_plot_cols), x_labels)
        for tick, ref in zip(ax.get_xticklabels()[n_cols + 1:], references):
            tick.set(rotation=30, ha="right", fontsize=8, color=ref.color)
        ax.set_yticks(range(n_rows), [pref.format_logit(b) for b in bad_values])
        ax.set_xlabel("good logit")
        ax.set_ylabel("bad logit")
        ax.set_title(panel_title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        card_axes.append((name, (ax, cbar.ax)))

    notes = [
        "Right of the gap: canonical baselines as reference levels. They plan "
        "on the env reward table and never read C, so one value is their "
        "level across every cell (flat columns by construction). Bold = the "
        "cell beats every reference level on that panel's metric. The colour "
        "scale spans the swept cells only, so a reference column outside that "
        "range clips to the end of the scale; read those off the annotated "
        "value.",
        pref.seed_caveat(int(cfg["n_seeds"]), references),
    ]
    if args.listen_cost_json is not None:
        notes.append(_listen_cost_note(
            args.listen_cost_json, metrics_fn, grids, good_values, bad_values,
            canonical_cell,
        ))
    fig.text(
        0.5, 0.004, "\n".join(notes), ha="center", va="bottom", fontsize=8,
        color=ink,
    )

    plt.tight_layout(rect=(0, 0.11, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"figure saved to {args.output}", flush=True)

    if args.cards_dir is not None:
        save_panel_cards(
            fig, card_axes, args.cards_dir, "factored_tiger_preference",
            size=HEATMAP_CARD_SIZE_IN,
        )
        print(f"cards saved to {args.cards_dir}", flush=True)

    plt.close(fig)


if __name__ == "__main__":
    main()
