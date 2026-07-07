"""Shared helper: emit each already-drawn dashboard panel as its own PDF card.

The per-environment results plots (plot_door_code.py, ...) build a combined
diagnostic ``figure.pdf``. When run with ``--cards-dir`` they additionally call
``save_panel_cards`` to crop every panel to its own rearrangeable/resizable PDF,
and ``save_legend_card`` for a standalone shared legend. Cards are a pure add-on
— the combined figure is unchanged.
"""
from pathlib import Path

import matplotlib.pyplot as plt


def save_panel_cards(fig, named_axes, out_dir, prefix, pad=1.03):
    """Save each ``(name, ax)`` as ``out_dir/{prefix}_{name}.pdf``.

    Each card is the figure cropped to that axis's tight bounding box, so its
    title, tick/axis labels, and any axis-attached legend come along.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()
    for name, ax in named_axes:
        bbox = ax.get_tightbbox(renderer).transformed(inv).expanded(pad, pad)
        fig.savefig(out_dir / f"{prefix}_{name}.pdf", bbox_inches=bbox)


def save_legend_card(handles, labels, path, ncol=None):
    """Render ``handles``/``labels`` as a standalone horizontal legend PDF."""
    ncol = ncol or len(labels)
    fig = plt.figure(figsize=(max(4.0, 1.4 * len(labels)), 0.6))
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
