"""Repo-root visual theme shared across all dissertation chapters.

Single source of truth for figure colors, the matplotlib rcParams, and a couple
of small plotting helpers, so the whole dissertation can be re-skinned from one
place: edit `PALETTE` (or the colormap endpoints) here and every chapter's plots
pick up the change on the next render.

The palette is cover-aligned: it derives from the thesis LaTeX color macros
(muted teal `thesisPrimary #3A6F73`, dusty blue `thesisSecondary #8FAFC2`,
`thesisCharcoal #2A2A2A`, warm off-white `thesisBackground #F7F5F2`,
`thesisLightGrey #D8D4CE`), extended with a muted amber (gold/goal) and a
desaturated terracotta (pit/penalty/baseline) at the same saturation level so
figures and text read as one system. Figures are saved with a transparent
background (the thesis pages are white) so they sit directly on the page.

Each chapter is its own DVC project (its own `.dvc/`), so this file lives ABOVE
every chapter's DVC root and is imported at *runtime* rather than tracked as a
DVC dependency. After editing it, re-render a chapter's figures with
`dvc repro -f <plot-stage>` (the plot scripts themselves are tracked deps, so a
plain `dvc repro` already re-runs when those change).

Usage from a chapter script::

    import sys
    from pathlib import Path
    for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (_p / "dissertation_theme.py").exists():
            sys.path.insert(0, str(_p)); break
    import dissertation_theme as theme
    theme.apply_rcparams()
    goal_color = theme.PALETTE["goal"]
"""

from pathlib import Path

from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Thesis base colors (mirror the LaTeX \definecolor macros -- keep in sync).
# ---------------------------------------------------------------------------
THESIS = {
    "primary":    "#3A6F73",   # muted teal        (thesisPrimary)
    "secondary":  "#8FAFC2",   # dusty blue        (thesisSecondary)
    "charcoal":   "#2A2A2A",   # ink / dark accent (thesisCharcoal)
    "background": "#F7F5F2",   # warm off-white (thesisBackground macro; kept
                               # for reference -- figures save transparent
                               # because the printed pages are white)
    "light_grey": "#D8D4CE",   # rules / gridlines (thesisLightGrey)
    "white":      "#FFFFFF",
}

# ---------------------------------------------------------------------------
# Semantic palette. Tweak these to re-skin every figure in the dissertation.
# ---------------------------------------------------------------------------
PALETTE = {
    # -- Planners / methods (entity-stable across chapters) ------------------
    # The mainline result of the dissertation is AIF-MP: it wears the primary
    # cover teal. Direct-prior MP (the `nuijten` scheme, chapter 4's planner)
    # keeps its own steel blue everywhere so the two are never conflated.
    # Hexes below were tuned with the CVD/contrast validator against the page
    # surface; the low chroma of the brand teal is an accepted deviation
    # (mitigated by per-method markers + legends), everything else passes.
    "aif_mp":          THESIS["primary"],  # AIF-MP (hero)
    "bp":              "#B4553F",          # loopy BP baseline (terracotta)
    "direct_prior_mp": "#4679B5",          # Direct-prior MP (steel blue)
    "gradient":        "#B4553F",          # gradient VFE baseline (ch3/4; the
                                           # "baseline" terracotta -- never
                                           # co-plotted with loopy BP)
    "vbp":             "#A67E24",          # VBP action-channel scheme (dark
                                           # amber: a line-weight step of the
                                           # gold hue, >= 3:1 on the page)
    "rm_mp":           "#6E3D75",          # dyn-channel / RM-MP (plum; dark
                                           # enough to separate from the teal
                                           # under deuteranopia)
    "bp_marginal":     "#6E3D75",          # ch4 region-BP marginal ablation
    "region_extended": "#6F7D4E",          # region-extended BP (moss)

    # -- Maze / grid-world semantics -----------------------------------------
    "goal":       "#C49A3A",            # +1 reward / gold (muted amber)
    "goal_ink":   "#8A6A1E",            # dark amber for small goal text
    "penalty":    "#B4553F",            # -1 reward / pit (desat. terracotta)
    "start":      THESIS["primary"],    # start-state ring (teal)
    "sense":      THESIS["secondary"],  # SENSE / cue ring (dusty blue)
    "stochastic": "#7195AC",            # stochastic-transition border (a
                                        # deeper step of the dusty blue so the
                                        # rounded border clears 3:1 on cells)
    "sink":       THESIS["charcoal"],   # absorbing sink / wumpus
    "cell":       "#FBFAF8",            # empty cell fill (just off the page)
    "grid":       THESIS["light_grey"], # cell gridlines
    "ink":        THESIS["charcoal"],   # text drawn inside figures
    "muted":      "#8A837B",            # de-emphasized annotations / "other"

    # -- Trajectory (step order, early -> late): teal sequential ramp --------
    "path_lo":  "#B8D0D1",
    "path_mid": THESIS["primary"],
    "path_hi":  "#22474A",
    "path":     THESIS["primary"],      # single-color path uses (legends)

    # -- Observation-noise heatmap endpoints (low -> high noise). A warm-grey
    # field (not a hue) so the dusty-blue/teal accents stay reserved for
    # semantics; capped light so even the noisiest cell reads as soft ambient
    # shading, never a dark "wall".
    "fog_lo": "#FBFAF8",
    "fog_hi": "#ABA49B",
}

# Fixed categorical order for un-keyed multi-series plots (axes.prop_cycle).
CYCLE = [
    PALETTE["aif_mp"], PALETTE["bp"], PALETTE["direct_prior_mp"],
    PALETTE["vbp"], PALETTE["rm_mp"], PALETTE["region_extended"],
    PALETTE["muted"],
]

# Chapter-5 planning-method slugs -> entity color (used by the sweep plots).
METHOD_COLORS = {
    "active-inference":     PALETTE["aif_mp"],
    "loopy":                PALETTE["bp"],
    "nuijten":              PALETTE["direct_prior_mp"],
    "vbp-channel":          PALETTE["vbp"],
    "dyn-channel":          PALETTE["rm_mp"],
    "region-extended":      PALETTE["region_extended"],
    "precise-info-seeking": PALETTE["muted"],
    "loopy-vbp":            PALETTE["muted"],
}


# Per-method line markers: secondary (non-color) encoding so methods stay
# distinguishable for CVD readers and in greyscale print.
METHOD_MARKERS = {
    "active-inference":     "o",
    "loopy":                "s",
    "nuijten":              "^",
    "vbp-channel":          "D",
    "dyn-channel":          "v",
    "region-extended":      "P",
    "precise-info-seeking": "X",
    "loopy-vbp":            "*",
}


def method_color(method):
    """Entity color for a chapter-5 planning-method slug (muted fallback)."""
    return METHOD_COLORS.get(method, PALETTE["muted"])


def method_marker(method):
    """Line marker for a chapter-5 planning-method slug ('' fallback)."""
    return METHOD_MARKERS.get(method, "")


def noise_cmap():
    """Paper (clean) -> warm grey (noisy) colormap for the "fog" heatmap."""
    return LinearSegmentedColormap.from_list("fog", [PALETTE["fog_lo"], PALETTE["fog_hi"]])


def path_cmap():
    """Teal light -> dark colormap encoding trajectory step order (early -> late)."""
    return LinearSegmentedColormap.from_list(
        "path", [PALETTE["path_lo"], PALETTE["path_mid"], PALETTE["path_hi"]]
    )


def diverging_cmap():
    """Terracotta -> warm neutral -> teal, for bad/good fractions (midpoint 0.5)."""
    return LinearSegmentedColormap.from_list(
        "diverge", [PALETTE["penalty"], "#EFECE6", PALETTE["aif_mp"]]
    )


def ordinal_colors(n, lo=0.15, hi=1.0):
    """`n` ordered colors from the teal ramp (for ordered series, e.g. damping)."""
    import numpy as np
    cmap = path_cmap()
    return [cmap(v) for v in np.linspace(lo, hi, n)]


def apply_rcparams():
    """Apply the shared serif/sizing/color rcParams used across the dissertation.

    Figures are saved with a TRANSPARENT background (the thesis pages are
    white, so the figure sits directly on the page); ink, axes, and grid all
    take the thesis neutrals, and the default color cycle follows `CYCLE`.
    """
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        # Surfaces: transparent, so figures sit directly on the (white) page.
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.transparent": True,
        # Ink.
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        # Recessive grid.
        "grid.color": PALETTE["grid"],
        # Themed default cycle for un-keyed multi-series plots.
        "axes.prop_cycle": cycler(color=CYCLE),
    })


def savefig_both(fig, path_prefix, *, dpi=200, formats=("pdf", "png")):
    """Save `fig` to several formats sharing one path prefix (no extension).

    Returns the list of written paths. Creates parent dirs as needed.
    """
    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = prefix.with_suffix(f".{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    return written


def find_repo_root(start):
    """Walk up from `start` to the directory containing this theme module."""
    p = Path(start).resolve()
    for parent in [p, *p.parents]:
        if (parent / "dissertation_theme.py").exists():
            return parent
    return None
