"""Figure max_exponent_operator (paper Fig. fig:maxexp, Appendix app:maxexp): the operator T_s and lambda_1 = s_*.

(a) Low-lying spectrum of alpha_s^{-1} T_s against s at delta = 0.5, g = 3; s_* is where the bottom crosses zero.
(b) delta -> 0, g = 3: the Schrodinger potential 1 - g^2 C^d(tau), bound states psi_0, psi_1, and the zero mode
    C^x'(tau) (markers) on psi_1; right axis: s = -1 + sqrt(1 - E).
(c) s_*(g) (lines) against the median maximum exponent of simulated networks (open circles), delta -> 0, 0.25, 0.5, 1.

Data: data/max_exponent_operator.json (scripts/run_max_exponent.py) and simulated maximum exponents
(paper.tables.max_exponent_table).   python -m paper.fig_max_exponent [--out DIR] [--root DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paper import data, tables  # noqa: E402
from paper import style as ps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
G_A = 3.0          # gain of panels (a), (b)
D_B = 0.5          # time step of panel (a)
# Data keys of max_exponent_operator.json: "a" = Schrodinger problem (drawn on axS, panel (b)), "b" = spectrum of
# alpha_s^{-1} T_s (axT, panel (a)), "c" = s_*(g) (axC, panel (c)).


def main(out=None, root=None):
    ps.apply()
    R = data.max_exponent_data(root=root)
    a, b, c = R["a"], R["b"], R["c"]
    DC = ps.DELTA_COLORS
    OI = ps.OI
    SHADE = "0.92"

    fig = plt.figure(figsize=(ps.WIDTH, 2.4))
    # panel order follows the text (App. B.1): (a) spectrum of alpha_s^{-1} T_s at finite delta,
    # (b) delta -> 0 Schrodinger problem, (c) s_* vs simulations.  axS/axT keep their roles in the code below.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 1.0], left=0.075, right=0.99, bottom=0.165, top=0.93,
                          wspace=0.55)
    axT, axS, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- panel (b): SCS Schrodinger problem, g = G_A
    tau, V = np.array(a["tau"]), np.array(a["V"])
    E = np.array(a["E"]); Vinf = a["V_inf"]
    xl, ylo, yhi = 10.0, a["V_min"] - 0.09, 0.5
    axS.fill_between([-xl, xl], Vinf, yhi, color=SHADE, lw=0, zorder=0)
    axS.text(-xl + 0.4, yhi - 0.04, "continuum", fontsize=ps.ANN, color="0.35", va="top")
    axS.plot(tau, V, color="k", lw=1.0, zorder=3)
    axS.text(0.9, a["V_min"] + 0.02, r"$1-g^2C^d(\tau)$", fontsize=ps.ANN, ha="left", va="center")
    cols = [OI["blue"], OI["vermillion"]]
    amp = 0.16
    psi0 = np.array(a["psi"][0]); psi0 *= np.sign(psi0[np.argmin(np.abs(tau))])
    psi1 = np.array(a["psi"][1]); psi1 *= -np.sign(psi1[np.argmin(np.abs(tau - 1.0))])   # sign of C^x'(tau)
    sc0, sc1 = amp / np.abs(psi0).max(), amp / np.abs(psi1).max()
    for n, (En, psi, sc) in enumerate(((E[0], psi0, sc0), (E[1], psi1, sc1))):
        axS.plot([-xl, xl], [En, En], color=cols[n], lw=0.7, ls=(0, (3, 2)), zorder=1)
        axS.plot(tau, En + sc * psi, color=cols[n], lw=1.0, zorder=4)
    tx, cxp = np.array(a["cxp_tau"]), np.array(a["cxp"])
    pick = (np.abs(np.round(tx) - tx) < 1e-9) & (np.abs(tx) <= xl)                      # every 1 in tau
    axS.plot(tx[pick], E[1] + sc1 * cxp[pick], "o", ms=3.0, mfc="white", mec=cols[1], mew=0.7, zorder=5)
    axS.text(xl - 0.3, E[0] - 0.035, rf"$E_0={E[0]:.3f}$", ha="right", va="top", fontsize=ps.ANN, color=cols[0])
    axS.text(-xl + 0.3, E[1] - 0.035, r"$E_1=0$", ha="left", va="top", fontsize=ps.ANN, color=cols[1])
    axS.text(xl - 0.4, yhi - 0.04, rf"$g={G_A:g}$, $\delta\to0$", fontsize=ps.ANN, ha="right", va="top")
    axS.set_xlim(-xl, xl); axS.set_ylim(ylo, yhi)
    axS.set_xticks([-10, -5, 0, 5, 10])
    axS.set_xlabel(r"$\tau$"); axS.set_ylabel(r"$E$", labelpad=1)
    sax = axS.secondary_yaxis("right", functions=(lambda e: -1 + np.sqrt(np.clip(1 - e, 0, None)),
                                                    lambda s: 1 - (1 + s) ** 2))
    s_a = -1 + np.sqrt(1 - E[0])
    sax.set_ticks([-0.2, 0, s_a, 0.4], labels=["−0.2", "0", r"$s_*$", "0.4"])
    sax.get_yticklabels()[2].set_color(cols[0])
    sax.set_ylabel(r"$s$", labelpad=1, rotation=0, va="center")
    # C^{x prime}: \mathcal{0} is the Computer Modern prime (cmsy10), larger than the Liberation Sans \prime
    axS.legend(handles=[Line2D([], [], color=cols[0], lw=1.0, label=r"$\psi_0(\tau)$"),
                        Line2D([], [], color=cols[1], lw=1.0, label=r"$\psi_1(\tau)$"),
                        Line2D([], [], color=cols[1], ls="", marker="o", ms=3.0, mfc="white", mew=0.7,
                               label=r"$C^{x\mathcal{0}}(\tau)$")],
               loc="lower left", handlelength=1.4, borderaxespad=0.3, labelspacing=0.25)

    # ---- panel (a): spectrum of alpha_s^{-1} T_s at delta = D_B, g = G_A
    s = np.array(b["s"]); lev = np.array(b["levels"]); band = np.array(b["band"])
    s_u, s_st = b["s_u"], b["s_star"]
    yb0, yb1 = -1.8, 0.8
    axT.fill_between(s, band[:, 0], yb1 + 1, color=SHADE, lw=0, zorder=0)
    axT.text(-0.95, 0.55, "continuum", fontsize=ps.ANN, color="0.35", ha="left", va="center")
    for k in range(lev.shape[1] - 1, -1, -1):
        y = np.where(lev[:, k] < band[:, 0] - 1e-9, lev[:, k], np.nan)
        axT.plot(s, y, color=OI["blue"] if k == 0 else "0.5", lw=1.1 if k == 0 else 0.6, zorder=3)
    axT.axhline(0, color="k", lw=0.5, zorder=1)
    axT.plot([s_u, s_u], [yb0, yb1], color="k", lw=0.7, ls=(0, (3, 2)), zorder=1)
    axT.text(s_u + 0.05, yb1 - 0.06, r"$s_u$", fontsize=ps.ANN, ha="left", va="top")
    l1 = lev[:, 1]
    k1 = np.where(np.diff(np.sign(l1)))[0][0]
    s1 = s[k1] - l1[k1] * (s[k1 + 1] - s[k1]) / (l1[k1 + 1] - l1[k1])
    axT.plot([s1], [0], "o", ms=3.0, mfc="white", mec="0.4", mew=0.7, zorder=5)
    axT.plot([s_st], [0], "o", ms=3.8, color=OI["blue"], zorder=6)
    axT.annotate(r"$s_*$", xy=(s_st, 0), xytext=(s_st + 0.12, -0.32), fontsize=ps.ANN, color=OI["blue"],
                 ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=OI["blue"], shrinkA=0, shrinkB=2))
    axT.text(-0.55, -0.98, r"$b(s)$", fontsize=ps.ANN, color=OI["blue"], ha="left", va="center")
    axT.text(0.97, yb0 + 0.06, rf"$g={G_A:g}$, $\delta={D_B:g}$", fontsize=ps.ANN, ha="right", va="bottom")
    axT.set_xlim(s_u - 0.12, 1.0); axT.set_ylim(yb0, yb1)
    axT.set_xlabel(r"$s$"); axT.set_ylabel(r"spectrum of $\alpha_s^{-1}\mathcal{T}_s$", labelpad=1)

    # ---- (c) s_*(g) vs network max exponent
    sims = tables.max_exponent_table(root=root)
    order = (("1", 1.0), ("0.5", 0.5), ("0.25", 0.25), ("0", 0.0))
    for key, d in order:
        rows = c[key]
        axC.plot([r["g"] for r in rows], [r["s_star"] for r in rows], color=DC[d], lw=1.0, zorder=2)
        sm = sims[key]
        axC.plot(list(sm), [v["median"] for v in sm.values()], "o", ms=3.0, mfc="white", mec=DC[d], mew=0.7, zorder=3)
    axC.set_xscale("log")
    axC.set_xticks([1, 2, 5, 10, 20]); axC.set_xticklabels(["1", "2", "5", "10", "20"])
    axC.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axC.set_xlim(1, 22); axC.set_ylim(0, 1.3)
    axC.set_xlabel(r"$g$"); axC.set_ylabel(r"$\lambda_1$", labelpad=1)
    leg = axC.legend(handles=[Line2D([], [], color=DC[d], lw=1.0, marker="o", ms=3.0, mfc="white", mec=DC[d], mew=0.7,
                                     label=(r"$\delta\to0$" if d == 0 else rf"$\delta={d:g}$"))
                              for _, d in order[::-1]], loc="upper left", handlelength=1.4, labelspacing=0.25)
    axC.add_artist(leg)
    axC.legend(handles=[Line2D([], [], color="0.3", lw=1.0, label="theory"),
                        Line2D([], [], color="0.3", ls="", marker="o", ms=3.0, mfc="white", mew=0.7, label="simulation")],
               loc="lower right", handlelength=1.4, labelspacing=0.25)
    for ax, l in zip((axT, axS, axC), "abc"):
        ps.panel_label(ax, l)
    out = Path(out or OUT)
    out.mkdir(parents=True, exist_ok=True)
    ps.save(fig, out / "max_exponent_operator.pdf")
    print("wrote", out / "max_exponent_operator.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output directory (default: figures/)")
    p.add_argument("--root", default=None, help="data directory (default: data/)")
    a = p.parse_args()
    main(a.out, a.root)
