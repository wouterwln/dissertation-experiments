"""Plot the RockSample goal-prior preference ablation (AIF only).

Reads output/rock_sample_preference/good<G>_bad<B>/aif.json for the
6-cell cross in params.rock_sample_preference.cells and renders one
column per metric:

  1. Early normalised return (mean over the first --early-window episodes)
  2. Early SENSE actions (same window) — information gathering
  3. Early bad-rock samples (same window) — blind commitment, the
     behaviour the asymmetric prior is hand-set to suppress
  4. Final posterior mass on the persistent theta_true (last episode)
  5. Median episodes-to-identify (mass >= 0.9), annotated with the
     fraction of seeds that crossed; missing point = never crossed

good_logit rewards sampling a good rock, bad_logit penalises sampling a
bad one; the SENSE-before-SAMPLE boundary should track their ratio. The
swept cells are a *cross* through the canonical (2, 4) cell, not a full
grid, so this is deliberately NOT a heatmap: a grid with holes would
invite reading across cells that were never run. Row 1 varies bad_logit
at good=2 (the ratio axis), row 2 varies good_logit at bad=4 (the
orthogonal axis), and the canonical cell shared by both rows is ringed.
good == bad cells are the ones calibrated to the environment's own
symmetric +/-sample_reward. Row 1's bad-logit axis runs mildest ->
harshest left to right, matching the house convention the two heatmap
siblings apply top -> bottom; row 2's good-logit axis is simply
ascending.

Reference levels
----------------
Every panel also carries one horizontal level per canonical
goal-prior-independent baseline (_REFERENCE_PLANNERS, read from
--reference-dir). Those planners consume the env reward table and never
read C, so one canonical number is their level across the whole sweep —
which is what turns this figure from "how sensitive is AIF-MP to its own
goal prior?" into "does AIF-MP still beat the baselines at the
reward-matched cell?". They cost nothing to add: the numbers are already
on disk from the canonical rock_sample stage.

Usage:
    uv run python scripts/plot_rock_sample_preference.py \\
        --input-dir output/rock_sample_preference \\
        --output output/rock_sample_preference/figure.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# Repo-root dissertation theme (single source of truth for figure colors).
for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
    if (_p / "dissertation_theme.py").exists():
        sys.path.insert(0, str(_p))
        break
import dissertation_theme as theme  # noqa: E402

import _preference as pref  # noqa: E402
from _cards import save_legend_card, save_panel_cards  # noqa: E402

# Float-tolerant: DVC formats each cell directory from the params value, so
# a `2.0` in params.yaml yields `good2.0_bad4.0`. An `\d+`-only pattern
# would skip that directory and silently drop the cell from the figure.
_CELL_RE = re.compile(r"^good(\d+(?:\.\d+)?)_bad(\d+(?:\.\d+)?)$")

# Canonical baselines drawn as reference levels. Each plans on the env
# reward table and ignores C (PSRL/Oracle read it via VBP, so a flat level
# would be wrong for them); Random is left out so it does not stretch the
# panels' y ranges.
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
    # reward_normalised is NaN on episodes with no good rocks (nothing to
    # divide by), so use nanmean — same convention as the runner curves.
    reward = np.asarray(ep["reward_normalised"], dtype=float)
    senses = np.asarray(ep["n_senses"], dtype=float)
    bad = np.asarray(ep["n_bad_sampled"], dtype=float)
    mass = np.asarray(ep["posterior_mass_on_persistent_theta"], dtype=float)
    ident = summary["derived"]["episodes_to_identify"]
    return {
        "early_reward": float(np.nanmean(reward[:, :early_window])),
        "early_senses": float(senses[:, :early_window].mean()),
        "early_bad_samples": float(bad[:, :early_window].mean()),
        "final_mass": float(mass[:, -1].mean()),
        "identify_median": (
            float(ident["median"]) if ident["median"] is not None else np.nan
        ),
        "identify_fraction": float(ident["fraction_identified"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RockSample goal-prior preference ablation.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("output/rock_sample_preference"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("output/rock_sample"),
        help="canonical run directory supplying the baseline reference levels",
    )
    parser.add_argument(
        "--early-window", type=int, default=3,
        help="number of leading episodes in the early-window means",
    )
    parser.add_argument(
        "--cards-dir", type=Path, default=None,
        help="if set, also write each panel as its own PDF card here",
    )
    args = parser.parse_args()
    theme.apply_rcparams()
    metrics_fn = partial(_cell_metrics, early_window=args.early_window)

    cells = _discover_cells(args.input_dir)
    if not cells:
        raise FileNotFoundError(
            f"no good<G>_bad<B> subdirectories under {args.input_dir}; "
            f"run `dvc repro rock_sample_preference` first"
        )
    params = pref.load_params()
    pref.assert_cells_match(
        cells,
        (
            (float(c["good"]), float(c["bad"]))
            for c in params["rock_sample_preference"]["cells"].values()
        ),
        args.input_dir,
    )

    summaries: dict[tuple[float, float], dict] = {}
    for key, path in cells.items():
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run `dvc repro rock_sample_preference` first"
            )
        with path.open() as f:
            summaries[key] = json.load(f)
    metrics = {key: metrics_fn(summary) for key, summary in summaries.items()}
    references = pref.load_references(
        args.reference_dir, _REFERENCE_PLANNERS, metrics_fn,
    )

    # Canonical goal prior, which is also the centre of the cross: the two
    # arms are `good == good_fixed` and `bad == bad_fixed`.
    good_fixed = float(params["rock_sample"]["env"]["good_logit"])
    bad_fixed = float(params["rock_sample"]["env"]["bad_logit"])
    # The two arms of the cross. Each is (x label, [(x value, cell), ...],
    # x value of the shared canonical cell).
    arms = [
        (
            f"bad logit (good = {pref.format_logit(good_fixed)})",
            sorted((b, (g, b)) for (g, b) in cells if g == good_fixed),
            bad_fixed,
        ),
        (
            f"good logit (bad = {pref.format_logit(bad_fixed)})",
            sorted((g, (g, b)) for (g, b) in cells if b == bad_fixed),
            good_fixed,
        ),
    ]
    off_cross = sorted(
        c for c in cells if c[0] != good_fixed and c[1] != bad_fixed
    )
    if off_cross:
        print(
            f"note: {len(off_cross)} cell(s) lie on neither arm of the cross "
            f"and are not plotted: {off_cross}",
            flush=True,
        )

    cfg = next(iter(summaries.values()))["config"]
    window = f"episodes 0-{args.early_window - 1}"
    panels = [
        ("early_reward", f"Early normalised return ({window})"),
        ("early_senses", f"Early SENSE actions ({window})"),
        ("early_bad_samples", f"Early bad-rock samples ({window})"),
        ("final_mass", "Final posterior mass on persistent $\\theta^\\star$"),
        ("identify_median", "Median episodes to identify"),
    ]

    fig, axes = plt.subplots(
        len(arms), len(panels), figsize=(4.6 * len(panels), 4.2 * len(arms)),
    )
    title = (
        f"RockSample goal-prior preference cross (AIF) — "
        f"K={cfg['env']['K']}, grid={cfg['env']['grid_size']}, "
        f"eta={cfg['env']['sensor_noise_eta']}, "
        f"sample_reward=±{cfg['env']['sample_reward']}, "
        f"n_seeds={cfg['n_seeds']}, n_episodes={cfg['n_episodes']}"
    )
    fig.suptitle(title, fontsize=11)

    ink = theme.PALETTE["ink"]
    # Card names carry the swept axis: row 0 sweeps bad_logit, row 1
    # good_logit (the `arms` order above).
    card_axes = []
    arm_keys = ("bad_arm", "good_arm")
    for row, (x_label, arm, canonical_x) in enumerate(arms):
        x_values = [x for x, _ in arm]
        # The swept logit is an ordered axis, so it wears the theme's
        # ordinal teal ramp. Positions are ordinal (not the logit value)
        # so the 1-2-4-8 doubling doesn't stretch the panel.
        colors = theme.ordinal_colors(len(arm))
        positions = np.arange(len(arm))
        for col, (name, panel_title) in enumerate(panels):
            ax = axes[row, col]
            card_axes.append((f"{arm_keys[row]}_{name}", ax))
            values = np.array([metrics[cell][name] for _, cell in arm])
            pref.draw_level_lines(ax, references, name)
            ax.plot(positions, values, color=ink, alpha=0.35, lw=1.5, zorder=1)
            ax.scatter(
                positions, values, c=colors, s=90, zorder=2,
                edgecolors=ink, linewidths=0.6,
            )
            if canonical_x in x_values:
                i = x_values.index(canonical_x)
                ax.scatter(
                    [positions[i]], [values[i]], s=260, facecolors="none",
                    edgecolors=ink, linewidths=2.0, zorder=3,
                )
            if name == "identify_median":
                for i, (_, cell) in enumerate(arm):
                    if np.isnan(values[i]):
                        continue
                    ax.annotate(
                        f"{metrics[cell]['identify_fraction']:.2f}",
                        (positions[i], values[i]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=ink,
                    )
            ax.set_xticks(positions, [pref.format_logit(x) for x in x_values])
            ax.set_xlabel(x_label)
            ax.set_xlim(-0.4, len(arm) - 0.6)
            if row == 0:
                ax.set_title(panel_title, fontsize=10)
    ring = mlines.Line2D(
        [], [], color=ink, marker="o", markerfacecolor="none",
        markeredgewidth=2.0, markersize=11, linestyle="none",
        label=(
            f"canonical (good={pref.format_logit(good_fixed)}, "
            f"bad={pref.format_logit(bad_fixed)}) — shared by both rows"
        ),
    )
    fig.legend(
        handles=[ring, *pref.legend_handles(references)],
        loc="lower center", bbox_to_anchor=(0.5, 0.028), frameon=False,
        fontsize=9, ncol=1 + len(references),
    )
    fig.text(
        0.5, 0.004, pref.seed_caveat(int(cfg["n_seeds"]), references),
        ha="center", va="bottom", fontsize=8, color=ink,
    )

    plt.tight_layout(rect=(0, 0.075, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"figure saved to {args.output}", flush=True)

    if args.cards_dir is not None:
        # The shared legend is figure-level here (save_panel_cards hides it
        # during the crop), so it gets its own card like the dashboards'.
        save_panel_cards(
            fig, card_axes, args.cards_dir, "rock_sample_preference",
        )
        handles = [ring, *pref.legend_handles(references)]
        # ncol=2: these labels are sentences, not planner names — one row
        # would emit a ~14 in strip that shrinks to nothing at text width.
        save_legend_card(
            handles, [h.get_label() for h in handles],
            args.cards_dir / "rock_sample_preference_legend.pdf", ncol=2,
        )
        print(f"cards saved to {args.cards_dir}", flush=True)

    plt.close(fig)


if __name__ == "__main__":
    main()
