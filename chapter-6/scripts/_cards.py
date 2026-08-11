"""Shared helper: emit each already-drawn dashboard panel as its own PDF card.

The per-environment results plots (plot_door_code.py, ...) build a combined
diagnostic ``figure.pdf``. When run with ``--cards-dir`` they additionally call
``save_panel_cards`` to crop every panel to its own rearrangeable/resizable PDF,
and ``save_legend_card`` for a standalone shared legend. Cards are a pure add-on
— the combined figure is unchanged.

Every panel card is emitted at exactly ``CARD_SIZE_IN``, across benchmarks as
well as within one, so cards drop into a cross-environment comparison figure
without per-card rescaling. That relies on two things together: the dashboards
all lay their panels out at the same per-panel geometry (see
``CARD_SIZE_IN``), and the crop below pads every tight bounding box out to the
common size instead of taking whatever each panel's tick labels happened to
need.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


# Common card size, in inches: the SMALLEST box that fits every panel in the
# slate, not a round number with headroom -- slack here is dead margin on
# every card, since each one is padded out to this size.
#
# Every benchmark dashboard lays its panels out at 5.0 x 4.5 in per cell (2x4
# at (20, 9), 3x4 at (20, 13.5), 3x3 at (15, 13.5)); within a cell the drawn
# axes is smaller, and its tight bounding box then adds tick labels, axis
# labels and any axis-attached legend. Measured across the 36 current panels
# that lands at 359.1 x 311.3 pt = 4.99 x 4.32 in, which this rounds just
# past. Re-measure (emit with size=None and compare page sizes) if a panel
# gains a wider label; a panel needing more warns and is emitted oversized
# rather than silently cropped.
CARD_SIZE_IN = (5.0, 4.35)

# The preference-heatmap slate (plot_door_code_preference.py,
# plot_factored_tiger_preference.py) is its own card family: an annotated
# heatmap plus its colorbar is wider and squatter than a dashboard line
# panel, so padding it to CARD_SIZE_IN would break width instead of
# equalising it. Measured across the 9 current heatmap panels the tight
# boxes land at 5.41 x 3.54 in, which this rounds just past; the same
# re-measure rule as CARD_SIZE_IN applies.
HEATMAP_CARD_SIZE_IN = (5.45, 3.6)


def _fixed_size(bbox, size, label):
    """Grow ``bbox`` to exactly ``size``, keeping it centred.

    A panel whose tight bounding box already exceeds ``size`` in either
    direction keeps its own larger extent in that direction: clipping an axis
    label off a paper figure is a worse failure than one card breaking the
    common size, so the mismatch is reported and the content preserved.
    """
    want_w, want_h = size
    if bbox.width > want_w or bbox.height > want_h:
        print(
            f"warning: card '{label}' needs "
            f"{bbox.width:.2f}x{bbox.height:.2f} in, exceeding CARD_SIZE_IN "
            f"{want_w}x{want_h}; emitted oversized rather than cropped",
            file=sys.stderr,
        )
        want_w, want_h = max(want_w, bbox.width), max(want_h, bbox.height)
    dw, dh = (want_w - bbox.width) / 2.0, (want_h - bbox.height) / 2.0
    return Bbox.from_extents(
        bbox.x0 - dw, bbox.y0 - dh, bbox.x1 + dw, bbox.y1 + dh,
    )


def save_panel_cards(fig, named_axes, out_dir, prefix, pad=1.03,
                     size=CARD_SIZE_IN):
    """Save each ``(name, ax)`` as ``out_dir/{prefix}_{name}.pdf``.

    Each card is the figure cropped to that axis's tight bounding box, so its
    tick/axis labels and any axis-attached legend come along. The panel title
    does not: a card is placed in the paper under its own ``\\caption``, and a
    title baked into the PDF would duplicate it there. A panel that owns
    further axes — a heatmap and its colorbar, which matplotlib makes a
    sibling axes rather than a child artist — passes ``(name, (ax, cbar.ax))``
    and the card crops to their union; the first axes is the panel (its title
    is the one cleared).

    The crop is then padded out to ``size`` about its own centre, so every
    card is the same size regardless of how wide that panel's tick labels ran.
    Pass ``size=None`` for the old tight-crop behaviour.

    The title is cleared only for the crop and restored immediately, so the
    combined ``figure.pdf`` keeps its titles whether it is saved before or
    after the cards. Clearing it requires re-drawing before the bounding box
    is measured, hence the draw inside the loop rather than once above it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = fig.dpi_scale_trans.inverted()
    for name, axes in named_axes:
        group = list(axes) if isinstance(axes, (list, tuple)) else [axes]
        ax = group[0]
        title = ax.get_title()
        ax.set_title("")
        # Everything else on the canvas is hidden for the duration of the
        # save. The crop is a window onto the shared figure, and padding it
        # out to the common size widens that window past the panel's own
        # extent -- with the neighbours still drawn, the margin fills with
        # slivers of the adjacent axes and their titles instead of blank
        # space. Figure-level legends are hidden with the figure texts (a
        # dashboard's shared legend belongs to its own legend card, not to
        # whichever panel it happens to sit under). Prior visibility is
        # captured, not assumed, so an axis a dashboard deliberately hid
        # stays hidden.
        hidden = (
            [a for a in fig.axes if a not in group]
            + list(fig.texts) + list(fig.legends)
        )
        was_visible = [a.get_visible() for a in hidden]
        for artist in hidden:
            artist.set_visible(False)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = Bbox.union([
                a.get_tightbbox(renderer).transformed(inv) for a in group
            ]).expanded(pad, pad)
            if size is not None:
                bbox = _fixed_size(bbox, size, f"{prefix}_{name}")
            fig.savefig(out_dir / f"{prefix}_{name}.pdf", bbox_inches=bbox)
        finally:
            for artist, visible in zip(hidden, was_visible):
                artist.set_visible(visible)
            ax.set_title(title)


def save_legend_card(handles, labels, path, ncol=None):
    """Render ``handles``/``labels`` as a standalone horizontal legend PDF."""
    ncol = ncol or len(labels)
    fig = plt.figure(figsize=(max(4.0, 1.4 * len(labels)), 0.6))
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
