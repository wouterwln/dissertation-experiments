"""Repo-root visual theme shared across all dissertation chapters.

Single source of truth for figure colors, the matplotlib rcParams, and a couple
of small plotting helpers, so the whole dissertation can be re-skinned from one
place: edit `PALETTE` (or the colormap endpoints) here and every chapter's plots
pick up the change on the next render.

Each chapter is its own DVC project (its own `.dvc/`), so this file lives ABOVE
every chapter's DVC root and is imported at *runtime* rather than tracked as a
DVC dependency. After editing it, re-render a chapter's figures with
`dvc repro -f <plot-stage>` (the plot scripts themselves are tracked deps, so a
plain `dvc repro` already re-runs when those change).

Usage from a chapter script::

    import sys
    from dissertation_theme import find_repo_root  # once it's on sys.path
    # ...or bootstrap by walking up to the dir that contains this file:
    from pathlib import Path
    for _p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (_p / "dissertation_theme.py").exists():
            sys.path.insert(0, str(_p)); break
    import dissertation_theme as theme
    theme.apply_rcparams()
    goal_color = theme.PALETTE["goal"]
"""

from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Semantic palette. Tweak these to re-skin every figure in the dissertation.
# ---------------------------------------------------------------------------
PALETTE = {
    # Planners / methods (kept consistent with the scaling plots).
    "direct_prior_mp": "#2c7fb8",   # Direct-prior MP
    "gradient":        "#c0392b",   # gradient VFE baseline
    "bp_marginal":     "#7b5ea7",   # region-BP marginal ablation

    # Maze / grid-world semantics.
    "goal":        "#2c8a3d",       # +1 reward
    "penalty":     "#c0392b",       # -1 reward
    "start":       "#1f4e79",       # start state
    "stochastic":  "#0f8b8d",       # stochastic-transition accent (teal)
    "sink":        "#2b2b2b",       # absorbing sink (near-black)
    "path":        "#e8730c",       # trajectory overlay (orange)

    # Observation-noise heatmap endpoints (low noise -> high noise). Capped at a
    # light, desaturated steel so even the highest-noise cell reads as soft
    # ambient shading (a field), never a dark "wall".
    "fog_lo":      "#ffffff",
    "fog_hi":      "#a6b5c4",
}


def noise_cmap():
    """White (clean) -> medium slate (noisy) colormap for the "fog" heatmap."""
    return LinearSegmentedColormap.from_list("fog", [PALETTE["fog_lo"], PALETTE["fog_hi"]])


def apply_rcparams():
    """Apply the shared serif/sizing rcParams used across the dissertation."""
    import matplotlib
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
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
