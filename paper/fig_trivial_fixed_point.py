"""Figure trivial_fixed_point (paper Fig. fig:tfp, Appendix app:tfp): eigenvalue-plane geometry of F(s) at g = 0.8.

(a) The disk |mu| < g of the eigenvalues of J, the disk |mu - c_delta| < r_s and their overlap region (delta = 0.5,
    s = -0.6).  (b) As delta -> 0, the circle |mu - c_delta| = r_s approaches the line Re mu = 1 + s.  (c) F(s): the
    single-site readout (lines; equal to the exact overlap fraction by Gauss's law), simulated networks (N = 2000,
    open circles) and the delta -> 0 semicircle law (dashed).

Computed from scratch in seconds (no data):  python -m paper.fig_trivial_fixed_point [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paper import style as ps  # noqa: E402
from rnn_lyapunov import trivial_fixed_point as tfp  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
G = 0.8
COL = {d: ps.DELTA_COLORS[d] for d in (0.5, 0.2, 0.05)}


DOTS = dict(ls="", marker=".", ms=1.4, mew=0, color="0.8", rasterized=True, zorder=0)   # eigenvalues of J
YLIM = (-1.6, 1.9)                                       # shared by (a) and (b); equal aspect, so equal heights


def main(out=None):
    ps.apply()
    rng = np.random.default_rng(0)
    N = 2000
    mu = np.linalg.eigvals(rng.normal(0, G / np.sqrt(N), (N, N)))
    xa, xb = (-2.6, 1.25), (-1.05, 1.75)
    wa, wb = xa[1] - xa[0], xb[1] - xb[0]
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.35), gridspec_kw=dict(width_ratios=[wa, wb, 2.7]))
    ring = np.exp(1j * np.linspace(0, 2 * np.pi, 721))
    lab = dict(fontsize=ps.ANN, ha="center", va="center")
    white = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85)

    # (a) geometry at delta = 0.5, s = -0.6
    ax = axes[0]
    d, s = 0.5, -0.6
    col = COL[d]
    c, r = 1 - 1 / d, np.exp(d * s) / d
    ax.plot(mu.real, mu.imag, **DOTS)
    ax.plot(G * ring.real, G * ring.imag, color="k", lw=1.0)
    big = c + r * ring
    ax.plot(big.real, big.imag, color=col, lw=1.0)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 600), np.linspace(-1, 1, 600))
    z = xx + 1j * yy
    overlap = (np.abs(z) < G) & (np.abs(z - c) < r)
    ax.contourf(xx, yy, overlap.astype(float), levels=[0.5, 1.5], colors=[col], alpha=0.22, zorder=1)
    th = np.linspace(-np.pi, np.pi, 4001)
    arc = c + r * np.exp(1j * th)
    inner = np.abs(arc) < G
    ax.plot(arc.real[inner], arc.imag[inner], color=col, lw=2.4, solid_capstyle="butt")
    ax.text(-1.0, 1.68, r"$|\mu-c_\delta|=r_s$", color=col, **lab)
    ax.annotate(r"$|\mu(\theta)|<g$", xy=(0.36, -0.55), xytext=(0.62, -1.2), color=col, fontsize=ps.ANN, ha="center",
                va="center", arrowprops=dict(arrowstyle="-", color=col, lw=0.6, shrinkA=1, shrinkB=1))
    ax.text(-0.3, -0.36, "overlap\nregion", bbox=white, linespacing=1.0, **lab)
    t0 = 0.3
    p = c + r * np.exp(1j * t0)
    ax.plot([c, p.real], [0, p.imag], color=col, lw=0.7, ls="--")
    ax.plot([c], [0], "o", ms=3, color=col)
    ax.plot([p.real], [p.imag], "o", ms=3.5, color=col, zorder=4)
    ang = np.linspace(0, t0, 50)
    ax.plot(c + 0.6 * np.cos(ang), 0.6 * np.sin(ang), color=col, lw=0.7)
    ax.text(c + 0.68, 0.1, r"$\theta$", fontsize=ps.ANN, color=col, ha="left", va="center", bbox=white, zorder=5)
    ax.text(p.real + 0.08, p.imag + 0.12, r"$\mu(\theta)$", fontsize=ps.ANN, color=col, bbox=white)
    ax.text(c, -0.2, r"$c_\delta$", ha="center", va="center", fontsize=ps.ANN, color=col)
    ax.plot([c, c + r * np.cos(2.4)], [0, r * np.sin(2.4)], color=col, lw=0.7, ls=":")
    ax.text(c + 0.5 * r * np.cos(2.4) - 0.16, 0.5 * r * np.sin(2.4) + 0.1, r"$r_s$", fontsize=ps.ANN, color=col)
    ax.text(0.8, 0.95, r"$|\mu|=g$", **lab)
    ax.axhline(0, color="k", lw=0.5, zorder=0); ax.axvline(0, color="k", lw=0.5, zorder=0)
    ax.set_xlim(*xa); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\,\mu$"); ax.set_ylabel(r"$\mathrm{Im}\,\mu$")

    # (b) delta -> 0: circles approach the line Re mu = 1 + s; key: delta as colored text
    ax = axes[1]
    ax.plot(mu.real, mu.imag, **DOTS)
    ax.plot(G * ring.real, G * ring.imag, color="k", lw=1.0)
    for d in (0.5, 0.2, 0.05):
        c, r = 1 - 1 / d, np.exp(d * s) / d
        big = c + r * np.exp(1j * np.linspace(0, 2 * np.pi, 20001))
        big = np.where((np.abs(big.imag) < YLIM[1]) & (big.real > xb[0]), big, np.nan)
        ax.plot(big.real, big.imag, color=COL[d], lw=1.0)
    ax.plot([1 + s, 1 + s], list(YLIM), color="k", lw=1.0, ls="--")
    ax.text(1 + s + 0.07, 1.65, r"$\mathrm{Re}\,\mu=1+s$", fontsize=ps.ANN, ha="left", va="center")
    ax.text(-0.66, -0.95, r"$|\mu|=g$", **lab)
    for k, (d, txt) in enumerate([(0.5, r"$\delta=0.5$"), (0.2, r"$\delta=0.2$"), (0.05, r"$\delta=0.05$"),
                                  (0.0, r"$\delta\to0$")]):
        ax.text(1.0, 0.45 - 0.3 * k, txt, color=COL.get(d, "k"), fontsize=ps.ANN, ha="left", va="center")
    ax.set_xlim(*xb); ax.set_ylim(*YLIM); ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\,\mu$"); ax.set_ylabel(r"$\mathrm{Im}\,\mu$")
    ax.set_xticks([-1, 0, 1])

    # (c) cumulative distribution: single-site theory (= exact result) as lines, network as markers
    ax = axes[2]
    ss = np.linspace(-3.2, 0.1, 331)
    sp = np.linspace(-3.0, 0.0, 21)
    MK = dict(ls="", marker="o", ms=3.0, mfc="white", mew=0.7)
    for d in (0.5, 0.2, 0.05):
        lam = np.sort(tfp.exponents_from_eigenvalues(mu, d))
        ax.plot(ss, [tfp.single_site_cdf(G, d, x) for x in ss], color=COL[d], lw=1.0, zorder=2)
        ax.plot(sp, np.searchsorted(lam, sp, side="right") / N, mec=COL[d], zorder=4, **MK)
    ax.plot(ss, tfp.semicircle_cdf(G, ss), "k--", lw=1.0, zorder=3)
    ax.set_xlim(-3.2, 0.1); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$s$"); ax.set_ylabel(r"$F(s)$")
    ax.legend(handles=[Line2D([], [], color="0.3", lw=1.0, label="theory"),
                       Line2D([], [], color="0.3", mec="0.3", label="simulation", **MK)],
              loc="upper left", handlelength=1.4, labelspacing=0.3)
    for a_, l in zip(axes, "abc"):
        ps.panel_label(a_, l)
    fig.tight_layout(w_pad=0.8)
    out = Path(out or OUT) / "trivial_fixed_point.pdf"
    ps.save(fig, out)
    print("wrote", out, "| max |single-site - exact|:",
          max(abs(tfp.single_site_cdf(G, d, x) - tfp.overlap_fraction(G, d, x)) for d in (0.5, 0.2, 0.05)
              for x in np.linspace(-3, -0.2, 12)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output directory (default: figures/)")
    main(p.parse_args().out)
