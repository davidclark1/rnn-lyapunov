"""Figure spectra (paper Fig. fig:spectra): Lyapunov spectra at g = 3 and 5 for delta = 0.05, 0.1, 0.25, 0.5.

Lines: the single-site theory, s against 1 - F(s) (mean F over Sobol scrambles at the smallest regulator, closed at
the upper edge s_*).  Open circles: exponents of one simulated network, N = 4096 (every 100th; every 12th with
i/N < 0.07 in the enlarged panels).  (a), (b): g = 3; (c), (d): g = 5.

Data: paper.tables.spectra_table.    python -m paper.fig_spectra [--out DIR] [--root DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paper import style as ps  # noqa: E402
from paper import tables  # noqa: E402
from rnn_lyapunov import observables  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
DELTAS = (0.05, 0.1, 0.25, 0.5)
COLORS = ps.DELTA_COLORS
SIM = dict(ls="", marker="o", ms=3.0, mfc="white", mew=0.7)        # simulations: open circles
THL = dict(lw=1.0)                                                 # theory: lines


def main(out=None, root=None):
    ps.apply()
    table = tables.spectra_table(root=root)
    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for row, g in enumerate((3.0, 5.0)):
        axf, axp = axes[row]
        for k, d in enumerate(DELTAS):
            th, net = table[f"g{g:g}_delta{d:g}"]["theory"], table[f"g{g:g}_delta{d:g}"]["network"]
            N = net["N"]
            assert N == 4096
            lam = np.sort(net["exponents"])[::-1]
            r = (np.arange(N) + 0.5) / N
            rr, ss = observables.exponents_vs_rank(th["s"], th["F"], th["edge"])
            # subsampled ranks, staggered between time steps so that markers of different delta do not overlap
            full = np.arange(25 + 25 * k, N, 100)                           # every 100th rank (~41 markers)
            zoom = np.arange(3 + 3 * k, int(0.07 * N), 12)                  # every 12th rank with i/N < 0.07
            for ax, idx in ((axf, full), (axp, zoom)):
                ax.plot(rr, ss, color=COLORS[d], **THL, zorder=2)
                ax.plot(r[idx], lam[idx], color=COLORS[d], mec=COLORS[d], **SIM, zorder=3)
        axf.set_xlim(0, 1); axf.set_ylim(-3.1, 0.6); axf.set_yticks([-3, -2, -1, 0])
        axp.set_xlim(0, 0.07); axp.set_ylim(-0.02, 0.26 if g == 3 else 0.47)
        for ax in (axf, axp):
            ax.axhline(0, color="k", lw=0.5, zorder=1)
            ax.set_ylabel(r"$\lambda_i$")
            ax.text(0.97, 0.95, rf"$g={g:g}$", transform=ax.transAxes, ha="right", va="top", fontsize=ps.ANN)
        axf.set_xlabel(r"$i/N$"); axp.set_xlabel(r"$i/N$")
    # key on the first panel: delta as colored text (stacked in the order of the curves), line = theory, circle = simulation
    ax = axes[0, 0]
    for k, d in enumerate(DELTAS):
        ax.text(0.04, 0.40 - 0.085 * k, rf"$\delta={d:g}$", color=COLORS[d], transform=ax.transAxes, ha="left",
                va="center", fontsize=ps.ANN)
    ax.legend(handles=[Line2D([], [], color="0.3", label="theory", **THL),
                       Line2D([], [], color="0.3", mec="0.3", label="simulation", **SIM)],
              loc="center left", bbox_to_anchor=(0.2, 0.27), handlelength=1.6, borderaxespad=0)
    for ax, s in zip(axes.ravel(), "abcd"):
        ps.panel_label(ax, s)
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    out = Path(out or OUT)
    out.mkdir(parents=True, exist_ok=True)
    ps.save(fig, out / "spectra.pdf")
    print("wrote", out / "spectra.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output directory (default: figures/)")
    p.add_argument("--root", default=None, help="data directory (default: data/)")
    a = p.parse_args()
    main(a.out, a.root)
