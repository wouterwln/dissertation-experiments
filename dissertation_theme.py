"""Repo-root visual theme shared across all dissertation chapters.

Single source of truth for figure colors, the matplotlib rcParams, and a couple
of small plotting helpers, so the whole dissertation can be re-skinned from one
place: edit `PALETTE` (or the colormap endpoints) here and every chapter's plots
pick up the change on the next render.

The palette works in two registers. Algorithm lines use a COOL method family:
the BP -> VBP -> RM-MP -> AIF-MP progression sweeps violet-indigo -> azure ->
cyan-teal and resolves on the cover teal (`thesisPrimary #3A6F73`), which AIF-MP
wears as the deep, low-chroma hero while the earlier steps carry real chroma so
nothing reads faded. Warm hues (amber goal, terracotta pit) are reserved for
MAZE/GRID semantics only -- methods and grids never share a plot, so the cool
family owns the whole cool range. Other thesis neutrals mirror the LaTeX macros
(dusty blue `thesisSecondary #8FAFC2`, `thesisCharcoal #2A2A2A`, warm off-white
`thesisBackground #F7F5F2`, `thesisLightGrey #D8D4CE`). Figures are saved with a
transparent background (the thesis pages are white) so they sit directly on the page.

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
    # One COOL family: the BP -> VBP -> RM-MP -> AIF-MP progression sweeps
    # violet-indigo -> azure -> cyan-teal and resolves on the cover teal, which
    # AIF-MP wears as the deep low-chroma hero. The three blue-ish steps are
    # pushed wide (violet <-> cyan) so they hold apart under CVD; Direct-prior MP
    # is a cornflower satellite off the arc. Tuned with the CVD/contrast
    # validator: worst adjacent pair ~dE 12.6 (deutan, passes). The brand teal's
    # low chroma is the one accepted deviation, mitigated by per-method markers.
    "aif_mp":          THESIS["primary"],  # AIF-MP (hero; cover teal, deep anchor)
    "bp":              "#4A46A3",          # loopy BP (violet-indigo; cold origin)
    "direct_prior_mp": "#3F6FCB",          # Direct-prior MP (cornflower satellite)
    "gradient":        "#6E839A",          # gradient VFE baseline (ch3/4; recessive
                                           # cool slate -- never co-plotted with
                                           # loopy BP; flip to warm if you'd rather
                                           # it contrast the cool method line)
    "vbp":             "#2F97CE",          # VBP action-channel scheme (azure)
    "rm_mp":           "#12A6AE",          # dyn-channel / RM-MP (cyan-teal; the
                                           # bright step just before AIF)
    "bp_marginal":     "#4A46A3",          # marginal inference -> shares BP
    "region_extended": "#85925E",          # region-extended BP (moss; an extra
                                           # method, not in the cool progression
                                           # -- revisit if it needs to harmonize)

    # -- Maze / grid-world semantics -----------------------------------------
    # Vibrant + position-coded: warm and cool coexist here because grid marks
    # never overlap the way method lines do. Green good / coral bad / teal
    # agent / amber-orange navigation, with blue reserved for env stochasticity.
    "goal":       "#2FA25E",            # +1 reward / goal (clean green)
    "goal_ink":   "#1F7A45",            # dark green: small goal text / good actions
    "penalty":    "#D8452F",            # -1 reward / pit (coral-red)
    "start":      THESIS["primary"],    # start-state / agent ring (teal)
    "sense":      "#3AA5C4",            # SENSE / cue ring (cyan)
    "stochastic": "#6B7076",            # stochastic-transition border (cool slate
                                        # grey; blue is now the noise fog, so the
                                        # border goes neutral -- lighter than the
                                        # charcoal sink, which also carries chevrons)
    "arrow":      "#EF9A24",            # navigation / trajectory arrows (amber-
                                        # orange; warm pop vs green/red/blue)
    "sink":       THESIS["charcoal"],   # absorbing sink (ch4; charcoal)
    "wumpus":     "#35283F",            # wumpus monster (ch5; aubergine -- off
                                        # the black/gold/red "flag")
    "cell":       "#FBFAF8",            # empty cell fill (just off the page)
    "grid":       THESIS["light_grey"], # cell gridlines
    "ink":        THESIS["charcoal"],   # text drawn inside figures
    "muted":      "#8A837B",            # de-emphasized annotations / "other"

    # -- Trajectory (step order, early -> late): teal sequential ramp --------
    "path_lo":  "#B8D0D1",
    "path_mid": THESIS["primary"],
    "path_hi":  "#22474A",
    "path":     THESIS["primary"],      # single-color path uses (legends)

    # -- Observation-noise heatmap endpoints (low -> high noise): a cool dusty-
    # blue haze reads as uncertainty (ch4 has no cue/sense channel competing for
    # blue). Capped light so even the noisiest cell is soft ambient shading, never
    # a dark "wall". NB the ch4 convergence plot uses PALETTE["muted"] for its
    # faint traces, not this ramp -- so this can be blue without affecting it.
    "fog_lo": "#FBFAF8",
    "fog_hi": "#8CA9C6",
}

# Fixed categorical order for un-keyed multi-series plots (axes.prop_cycle).
CYCLE = [
    PALETTE["aif_mp"], PALETTE["bp"], PALETTE["direct_prior_mp"],
    PALETTE["vbp"], PALETTE["rm_mp"], PALETTE["region_extended"],
    PALETTE["muted"],
]

# Shared in-cell label sizing for the grid/maze renderers, so every chapter
# labels cells at one weight instead of ad-hoc per-figure sizes. Glyph
# convention across chapters: "A" = agent, "S" = SENSE (never start).
CELL_LABEL_SIZE = 10   # primary in-cell letter/number (goal, pit, terrain, agent)
CELL_BADGE_SIZE = 8    # small corner / step-number badges

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


# Chapter-6 planner slugs -> entity color (BA-POMDP bake-off plots). Unlike the
# method chapters (which own the cool range), ch6 is its OWN full-spectrum world:
# AIF-MP and VBP keep their method-family colors (teal, azure) since they recur
# in ch5; the ch6-specific BA-RL baselines spread across the wheel so up to 7
# lines separate under CVD (validated: worst pair
# dE 15.4, above the 12 target; all >= 3:1 on the page). POMCP + Factored
# BA-POMCP are BAMCP variants, so they share its coral and are told apart by
# linestyle. Only AIF's teal trips the chroma floor (accepted hero exception).
PLANNER_COLORS = {
    "aif":            PALETTE["aif_mp"],  # cover teal (hero, entity-stable)
    "psrl":           "#BE851F",          # amber
    "bamcp":          "#C0442E",          # coral (BAMCP family)
    "pomcp":          "#C0442E",          # coral (BAMCP family; dotted)
    "factored_bamcp": "#C0442E",          # coral (BAMCP family; dash-dot)
    "boss":           "#4E8C57",          # green
    "ids":            "#8E4FA0",          # purple
    "vbp":            PALETTE["vbp"],     # azure -- entity-stable with the method
                                          # family (same VBP as ch5), never pink
    "belief_q":       "#A15A2A",          # rust (dashboards only)
    "oracle":         PALETTE["ink"],     # charcoal bound (dashed)
    "random":         PALETTE["muted"],   # grey bound (dashed)
}
PLANNER_LINESTYLES = {
    "oracle": "--", "random": "--",       # bounds
    "pomcp": ":", "factored_bamcp": "-.",  # BAMCP variants (share its plum)
}
# Secondary (non-color) encoding for CVD readers / greyscale print.
PLANNER_MARKERS = {
    "aif": "o", "vbp": "D", "psrl": "^", "ids": "s", "bamcp": "v",
    "pomcp": "X", "factored_bamcp": "*", "boss": "P", "belief_q": "h",
}


def planner_color(name):
    """Entity color for a chapter-6 planner slug (muted fallback)."""
    return PLANNER_COLORS.get(name, PALETTE["muted"])


def planner_linestyle(name):
    """Linestyle for a chapter-6 planner slug (solid fallback)."""
    return PLANNER_LINESTYLES.get(name, "-")


def planner_marker(name):
    """Line marker for a chapter-6 planner slug ('' fallback)."""
    return PLANNER_MARKERS.get(name, "")


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
