"""Figure dimension_entropy (paper Fig. fig:dimension_entropy): dimensions and entropy rate against g.

(a) Kaplan-Yorke dimension D_KY/N (black) and participation ratios PR^phi/N (green), PR^x/N (purple) at
    delta -> 0, 0.25, 0.5 (three panels sharing the y axis).  (b) Entropy rate h_KS/N, one color per delta.
Lines: theory (single-site theory for D_KY and h_KS, two-site cavity theory for PR); open circles: simulations.

Data: paper.tables.dimension_entropy_table.    python -m paper.fig_dimension_entropy [--out DIR] [--root DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paper import style as ps  # noqa: E402
from paper import tables  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
COLORS = ps.DELTA_COLORS
# one color per quantity in (a), disjoint from the delta colors of (b)
QCOL = {"ky": ps.OI["black"], "phi": ps.OI["green"], "x": ps.OI["purple"]}
QLAB = {"ky": r"$D_{\mathrm{KY}}/N$", "phi": r"$\mathrm{PR}^{\phi}/N$", "x": r"$\mathrm{PR}^{x}/N$"}
SIM = dict(ls="", marker="o", ms=3.0, mfc="white", mew=0.7)        # simulations: open circles
THL = dict(lw=1.0)                                                 # theory: lines


def main(out=None, root=None):
    ps.apply()
    t = tables.dimension_entropy_table(root=root)
    deltas = (0.0, 0.25, 0.5)                                      # 0.0 = delta -> 0 (continuous time)
    dlab = {0.0: r"$\delta\to0$", 0.25: r"$\delta=0.25$", 0.5: r"$\delta=0.5$"}
    fig = plt.figure(figsize=(ps.WIDTH, 2.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.42, 1.25], wspace=0.08)
    dim = [fig.add_subplot(gs[0, 0])]
    dim += [fig.add_subplot(gs[0, j], sharey=dim[0]) for j in (1, 2)]
    ent = fig.add_subplot(gs[0, 4])
    for ax, delta in zip(dim, deltas):
        pts = sorted((p for p in t["points"].values() if p["delta"] == delta), key=lambda p: p["g"])
        th = [p for p in pts if p.get("theory")]
        net = [p for p in pts if p.get("network")]
        sims = [p for p in pts if p.get("pr_sim")]
        dense = t["pr_theory_dense"][delta]
        ax.plot([p["g"] for p in th], [p["theory"]["ky_dimension"] for p in th], color=QCOL["ky"], **THL)
        ax.plot([p["g"] for p in net], [p["network"]["ky_dimension"] for p in net], color=QCOL["ky"], mec=QCOL["ky"], **SIM)
        for key in ("phi", "x"):
            ax.plot(dense["g"], dense[f"pr_{key}"], color=QCOL[key], **THL)
            ax.plot([p["g"] for p in sims], [p["pr_sim"][f"pr_{key}"] for p in sims], color=QCOL[key], mec=QCOL[key], **SIM)
        ax.text(0.95, 0.04, dlab[delta], transform=ax.transAxes, ha="right", va="bottom", fontsize=ps.ANN)
        ent.plot([p["g"] for p in th], [p["theory"]["entropy_rate"] for p in th], color=COLORS[delta], **THL)
        ent.plot([p["g"] for p in net], [p["network"]["entropy_rate"] for p in net], color=COLORS[delta],
                 mec=COLORS[delta], **SIM)
    dim[0].set_ylim(0, 0.185)
    dim[0].set_ylabel(r"dimension$/N$")
    for ax in dim[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    ent.set_ylabel(r"$h_{\mathrm{KS}}/N$"); ent.set_ylim(bottom=0)
    for ax in dim + [ent]:
        ax.set_xlabel(r"$g$"); ax.set_xlim(1, 20.5); ax.set_xticks([1, 10, 20])
    ent.set_xticks([1, 5, 10, 15, 20])
    ps.panel_label(dim[0], "a"); ps.panel_label(ent, "b")
    # key on the first panel: quantities as colored text (D_KY first, then PR^phi, PR^x), line = theory, circle =
    # simulation; (b): delta as colored text, in increasing order from the top
    for k, key in enumerate(("ky", "phi", "x")):
        dim[0].text(0.98, 0.97 - 0.085 * k, QLAB[key], color=QCOL[key], transform=dim[0].transAxes, ha="right",
                    va="top", fontsize=ps.ANN)
    dim[0].legend(handles=[Line2D([], [], color="0.3", label="theory", **THL),
                           Line2D([], [], color="0.3", mec="0.3", label="simulation", **SIM)],
                  loc="upper left", bbox_to_anchor=(0.02, 1.0), handlelength=1.3, borderaxespad=0, labelspacing=0.25)
    for k, d in enumerate((0.0, 0.25, 0.5)):
        ent.text(0.05, 0.95 - 0.09 * k, dlab[d], color=COLORS[d], transform=ent.transAxes,
                 ha="left", va="top", fontsize=ps.ANN)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.93)
    out = Path(out or OUT)
    out.mkdir(parents=True, exist_ok=True)
    ps.save(fig, out / "dimension_entropy.pdf")
    print("wrote", out / "dimension_entropy.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="output directory (default: figures/)")
    p.add_argument("--root", default=None, help="data directory (default: data/)")
    a = p.parse_args()
    main(a.out, a.root)
