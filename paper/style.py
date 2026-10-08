"""Shared figure style of the paper: fonts, sizes, colors, panel labels.

Font: Arial if installed, else Liberation Sans (metric-compatible), Helvetica, DejaVu Sans; mathtext in the same sans
font.  Sizes >= 7 pt.  Colors: Okabe-Ito colorblind-safe palette, one color per time step delta, shared by all
figures (delta -> 0 is black).  Theory = solid lines, simulations = open circles, in every figure.

Usage:  from paper import style as ps; ps.apply(); ...; ps.panel_label(ax, "a"); ps.save(fig, path)
"""
import logging

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

ANN = 8      # in-panel annotations (axis-label size); never below 7 pt
WIDTH = 6.5  # \linewidth of the paper (article class, 1in margins), in inches

# Okabe-Ito palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7", gray="#999999")
# One color per time step delta, shared by all figures; delta -> 0 (continuous time) is black.
DELTA_COLORS = {0.05: OI["orange"], 0.1: OI["green"], 0.2: OI["sky"], 0.25: OI["blue"], 0.5: OI["vermillion"],
                1.0: OI["purple"], 0.0: OI["black"]}


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"):
        if name in names:
            return name
    return "sans-serif"


FONT = _font()

RC = {
    "font.size": 8,
    "font.family": FONT,
    "lines.linewidth": 0.7,
    "legend.fontsize": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "figure.dpi": 225,
    "figure.figsize": [4 / 2.54, 3 / 2.54],
    "path.simplify": True,
    "pdf.fonttype": 42,
    "image.interpolation": "none",
    "image.aspect": "auto",
    "svg.fonttype": "none",
    "legend.frameon": False,
    "patch.edgecolor": "none",
    "axes.formatter.limits": (-3, 3),
    "axes.spines.top": False,
    "axes.spines.right": False,
    # mathtext in the same sans font as the text
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": FONT,
    "mathtext.cal": "cmsy10",          # calligraphic capitals (\mathcal{T}) from Computer Modern; no sans script font
}


def apply():
    matplotlib.rcParams.update(RC)
    logging.getLogger("fontTools").setLevel(logging.ERROR)     # "'created' timestamp seems very low" on font subsetting


def panel_label(ax, letter, x=-0.02, y=1.0, **kw):
    """Bold lowercase panel label '(a)' at the top-left corner, outside the axes frame."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold", **kw)


def save(fig, path):
    """Write ``path`` (PDF, fonts embedded as TrueType) at 600 dpi for rasterized elements, and close the figure."""
    fig.savefig(path, dpi=600)
    plt.close(fig)
