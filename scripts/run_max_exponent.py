"""Data for the paper's figure max_exponent_operator: the operator T_s and the maximum exponent s_*(g).

    python scripts/run_max_exponent.py                      # -> results/max_exponent_operator.json (minutes, GPU optional)

Output keys (figure panels in brackets):
  "a" [panel (b)]  continuous time, g = 3: Schrodinger problem H = -d^2/dtau^2 + 1 - g^2 C^d(tau): potential, bound
                   states, C^x'(tau) (the zero mode); lag-window and grid checks.
  "b" [panel (a)]  delta = 0.5, g = 3: low-lying spectrum of alpha_s^{-1} T_s against s; window checks.
  "c" [panel (c)]  s_*(g) for delta = 0.25, 0.5, 1 and delta -> 0 on 30 values of g (31 at delta = 1, from 1.2),
                   checked against the theory pipeline's upper edge ("edge_lib"); DMFT lag-window and lag-grid
                   checks ("c_checks").

The operator is truncated to |tau| <= L with psi = 0 outside; beyond the DMFT lag window, C^d(tau) is continued by
its large-lag value <phi'>^2.  Network maximum exponents for panel (c) come from the simulation results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rnn_lyapunov import default_device, dmft, max_exponent as mx
from rnn_lyapunov.io import save_result


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the docstring layout and show defaults."""


G_A = 3.0          # gain of keys "a", "b"
D_B = 0.5          # time step of key "b"
G_GRID = [1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0,
          8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]


def tmax_for(g):
    """DMFT lag window: the correlation time grows as g -> 1."""
    return 640.0 if g < 1.25 else 320.0 if g < 1.5 else 160.0 if g < 2.25 else 80.0


def main(device, out):
    ga = dmft.GaussianAverages(device=device)
    sol_of = lambda g, delta, **kw: dmft.cached(g, delta, ga=ga, **kw)
    cd_inf = lambda sol: dmft.mean_gain(sol, ga) ** 2
    res = dict(kind="max_exponent_operator", G_A=G_A, D_B=D_B, quadrature=ga.params)

    # (a) continuous-time Schrodinger problem
    a, checks = {}, {}
    for dt, tmax, L in [(0.025, 80.0, 80.0), (0.025, 80.0, 160.0), (0.025, 80.0, 320.0), (0.025, 160.0, 160.0),
                        (0.0125, 80.0, 160.0)]:
        sol = sol_of(G_A, None, dt=dt, t_max=tmax)
        ri = cd_inf(sol)
        tau, cd = mx.cd_profile(sol, ri, L)
        V, E, psi = mx.schrodinger(G_A, tau, cd)
        Vinf = 1 - G_A**2 * ri
        checks[f"dt{dt:g}_tmax{tmax:g}_L{L:g}"] = dict(E=E[E < Vinf].tolist(), V_inf=Vinf,
                                                         s_star=mx.s_star_from_energy(E[0]),
                                                         continuous_edge=mx.continuous_edge(sol))
        if (dt, tmax, L) == (0.025, 80.0, 160.0):
            nb = int((E < Vinf).sum())
            keep = np.abs(tau) <= 12
            cx = np.r_[sol.c_x[:0:-1], sol.c_x]                      # C^x'(tau), normalized like the eigenfunctions
            cxp = np.gradient(cx, sol.dt)
            tx = (np.arange(len(cx)) - (len(sol.c_x) - 1)) * sol.dt
            cxp_n = cxp / np.sqrt(sol.dt * (cxp**2).sum())
            a = dict(g=G_A, dt=dt, t_max=tmax, L=L, tau=tau[keep].tolist(), V=V[keep].tolist(), V_inf=Vinf,
                     V_min=float(V.min()), E=E[:nb].tolist(), psi=[psi[keep, n].tolist() for n in range(nb)],
                     cxp_tau=tx[np.abs(tx) <= 12].tolist(), cxp=cxp_n[np.abs(tx) <= 12].tolist(),
                     overlap_psi1_cxp=float(abs(np.interp(tau, tx, cxp_n, left=0, right=0) @ psi[:, 1]) * dt),
                     q=sol.q, Cd0=float(sol.c_d[0]), r_inf=ri)
    a["checks"] = checks
    res["a"] = a
    print("(a) E =", a["E"], "V_inf =", a["V_inf"], "overlap(psi_1, Cx') =", a["overlap_psi1_cxp"], flush=True)

    # (b) spectrum of a^{-1} T_s versus s at delta = D_B
    sol = sol_of(G_A, D_B, t_max=80.0)
    ri = cd_inf(sol)
    tau, cd = mx.cd_profile(sol, ri, 160.0)
    s_u = np.log(1 - D_B) / D_B
    sst = mx.s_star(D_B, G_A, cd)
    ss = np.r_[np.linspace(s_u, s_u + 0.1, 41)[1:], np.linspace(s_u + 0.1, 1.2, 600)[1:]]
    lev, band = [], []
    for s in ss:
        d, o, al, c = mx.scaled_operator(D_B, G_A, s, cd)
        lev.append(mx.lowest(d, o, k=8).tolist())
        band.append([al + 1 / al - 2 - (c / al) * ri, al + 1 / al + 2 - (c / al) * ri])
    wins = {}
    for L in (40.0, 80.0, 160.0, 320.0):
        _, cd_L = mx.cd_profile(sol, ri, L)
        wins[f"L{L:g}"] = dict(s_star=mx.s_star(D_B, G_A, cd_L),
                               b_at_s0=mx.lowest(*mx.scaled_operator(D_B, G_A, 0.0, cd_L)[:2], k=3).tolist())
    res["b"] = dict(g=G_A, delta=D_B, s_u=s_u, s_star=sst, s=ss.tolist(), levels=lev, band=band, r_inf=ri,
                    edge_discrete_edge=mx.discrete_edge(sol), window_checks=wins)
    print("(b) s_u =", s_u, "s_* =", sst, "discrete_edge =", res["b"]["edge_discrete_edge"], flush=True)

    # (c) s_*(g); the second value is the theory pipeline's upper edge (DMFT window only)
    def point(g, delta, tm):
        if delta is None:
            tm = min(tm, 320.0)
            sol = sol_of(g, None, dt=0.025, t_max=tm)
            tau, cd = mx.cd_profile(sol, cd_inf(sol), 2 * tm)
            return mx.s_star_from_energy(mx.schrodinger(g, tau, cd, k=1)[1][0]), mx.continuous_edge(sol)
        if delta == 1.0:
            sol = sol_of(g, 1.0)
            return float(np.log(g * np.sqrt(sol.c_d[0]))), mx.discrete_edge(sol)
        sol = sol_of(g, delta, t_max=tm)
        _, cd = mx.cd_profile(sol, cd_inf(sol), 2 * tm)
        return mx.s_star(delta, g, cd), mx.discrete_edge(sol)

    c = {}
    for key, delta in (("0.25", 0.25), ("0.5", 0.5), ("1", 1.0), ("0", None)):
        rows = []
        for g in ([1.2] if delta == 1.0 else []) + G_GRID:          # delta = 1: closed form, down to the simulated 1.2
            tm = tmax_for(g)
            try:
                ours, lib = point(g, delta, tm)
            except RuntimeError as e:                       # DMFT Newton failure: record, do not plot
                print(f"(c) delta={key} g={g}: FAILED {e}", flush=True)
                continue
            rows.append(dict(g=g, t_max=tm if delta != 1.0 else None, s_star=ours, edge_lib=lib))
            print(f"(c) delta={key} g={g}: s_*={ours:.6f} edge={lib:.6f}", flush=True)
        c[key] = rows

    wc = []                                                 # DMFT lag window doubled; lag grid halved
    for g in (1.5, 3.0, 10.0):
        for delta in (0.25, 0.5, None):
            tm = tmax_for(g)
            vals = [mx.upper_edge(sol_of(g, delta, dt=0.025, t_max=t)) for t in (tm, 2 * tm)]
            wc.append(dict(g=g, delta=delta, t_max=[tm, 2 * tm], s_star=vals))
    for g in (1.5, 3.0, 10.0):
        v = [mx.continuous_edge(sol_of(g, None, dt=dt, t_max=tmax_for(g))) for dt in (0.025, 0.0125)]
        wc.append(dict(g=g, delta=None, dt=[0.025, 0.0125], s_star=v))
    res["c"], res["c_checks"] = c, wc
    print("wrote", save_result(out, res))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument("--out", default="results/max_exponent_operator")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    main(a.device or default_device(), Path(a.out))
