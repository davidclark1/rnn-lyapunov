"""Every plotted number of the paper, built from result files (``paper.data``) with the library's readouts.

    python -m paper.tables                       # readable summary of the bundled data
    python -m paper.tables --root results --json tables.json

  :func:`spectra_table`          Fig. spectra (1): theory F(s) averaged over scrambles, and N = 4096 network spectra
  :func:`dimension_entropy_table` Fig. dimension_entropy (2): D_KY/N, h_KS/N (theory, networks), PR^x/N, PR^phi/N
  :func:`max_exponent_table`     Fig. max_exponent_operator (4c): network maximum exponents
  :func:`grid_spacing_table`     delta -> 0 grid-spacing study at g = 20 (supports the extrapolation of Fig. 2)

Theory readouts (D_KY/N, h_KS/N) are recomputed from F(s) at the smallest regulator with
:func:`rnn_lyapunov.theory.dimension_entropy`; no simulated exponent enters the theory.  The dataset holds only the
files the paper uses, so the selection rules below are short; they are the paper's rules, kept as named constants.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from rnn_lyapunov import is_continuous, theory
from paper import data

# Fig. 1: two gains, four time steps; theory on the full-spectrum threshold grid, one N = 4096 network (seed 0).
SPECTRA_G = (3.0, 5.0)
SPECTRA_DELTAS = (0.05, 0.1, 0.25, 0.5)
SPECTRA_N = 4096
# Fig. 2: time steps (0.0 = delta -> 0).  Theory on the upper-spectrum grids ("top", "upper"; "full" is Fig. 1's).
DIMENSION_DELTAS = (0.0, 0.25, 0.5)
DIMENSION_RULES = ("top", "upper")
# delta -> 0 at g >= 8: each scramble is solved at m = 160 and 256 on T = 32 (spacings 0.2, 0.125) and extrapolated
# quadratically to zero spacing (theory.extrapolate_to_zero_spacing), then averaged over scrambles.
EXTRAPOLATED_G = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
EXTRAPOLATION_M = (160, 256)


def dimension_network_N(delta, g: float) -> int:
    """Fig. 2 network size: near onset (g <= 1.5) N = 4096 has too few positive exponents, so the map uses
    N = 16384 and the flow N = 8192 (RK4 costs ~4x per step); otherwise N = 4096.  Value = median over networks."""
    if g <= 1.5:
        return 8192 if is_continuous(delta) else 16384
    return 4096


def max_exponent_N(g: float, available) -> int:
    """Fig. 4c network size: N = 4096 for g >= 2, else the largest N available.  Value = median over networks."""
    return 4096 if g >= 2 and 4096 in available else max(available)


MAX_EXPONENT_KEYS = {"0.25": 0.25, "0.5": 0.5, "1": 1.0, "0": 0.0}


def _mean_sem(v) -> tuple[float, float]:
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def _key(delta, g) -> str:
    return f"delta{0 if is_continuous(delta) else delta:g}_g{g:g}"


# ---- Fig. 1 -------------------------------------------------------------------------------------------------------

def spectra_table(root=None) -> dict:
    """``{"g3_delta0.5": {...}, ...}`` for g in SPECTRA_G and delta in SPECTRA_DELTAS.  Each entry::

        g, delta
        theory:  s (thresholds), F (mean over scrambles of F(s) at the smallest regulator), F_all (s x eps, mean),
                 eps, edge (upper edge s_*, F(edge) = 1), n_scrambles, seeds, files
        network: N, seed, exponents (descending), ky_dimension, entropy_rate, file

    The theory curve lambda(i/N) of the figure is ``observables.exponents_vs_rank(s, F, edge)``.  ``theory`` or
    ``network`` is None where no result exists (e.g. a partial results directory).
    """
    out = {}
    for g in SPECTRA_G:
        for d in SPECTRA_DELTAS:
            th = data.theory_scrambles(d, g, root=root, rule="full")
            assert all(np.allclose(r["s"], th[0]["s"]) for r in th), "scrambles must share thresholds"
            nets = [r for r in data.networks(d, g, N=SPECTRA_N, root=root)
                    if r["params"]["full_spectrum"] and r["params"]["seed"] == 0]
            assert len(nets) <= 1, f"one full N={SPECTRA_N} spectrum expected at g={g}, delta={d}"
            entry = dict(g=g, delta=d, theory=None, network=None)
            if th:
                F_all = np.stack([np.asarray(r["F"], float) for r in th]).mean(0)
                entry["theory"] = dict(s=th[0]["s"], F=F_all[:, -1].tolist(), F_all=F_all.tolist(), eps=th[0]["eps"],
                                       edge=th[0]["edge"], n_scrambles=len(th), seeds=[r["config"]["seed"] for r in th],
                                       files=[r["_file"] for r in th])
            if nets:
                n = nets[0]
                entry["network"] = dict(N=n["params"]["N"], seed=0, exponents=sorted(n["exponents"], reverse=True),
                                        ky_dimension=n["summaries"]["ky_dimension"],
                                        entropy_rate=n["summaries"]["entropy_rate"], file=n["_file"])
            out[f"g{g:g}_delta{d:g}"] = entry
    return out


# ---- Fig. 2 -------------------------------------------------------------------------------------------------------

def _settings(r: dict) -> dict:
    c = r["config"]
    return dict(T=c["T"], m=c["m"] if is_continuous(c["delta"]) else round(c["T"] / c["delta"]), rule=r["grid"]["rule"],
                n_twist=c["n_twist"], n_paths=c["n_paths"], dmft_tmax=c["dmft_tmax"], eps=list(c["eps"]),
                n_thresholds=len(r["s"]))


def theory_point(delta, g: float, root=None) -> dict | None:
    """Theory D_KY/N and h_KS/N at ``(delta, g)``: mean and s.e.m. over Sobol scrambles (one setting per point).

    Returns ``ky_dimension``, ``ky_dimension_sem``, ``entropy_rate``, ``entropy_rate_sem``, ``n_scrambles``,
    ``seeds``, ``settings``, ``edge``, ``all_converged``, ``extrapolated`` and ``files``; extrapolated points also
    give the means at each grid (``ky_dimension_m160`` etc.) and ``unpaired_seeds`` (scramble files without their
    m partner, left out; empty for the bundled data).
    """
    rows = [r for r in data.theory_scrambles(delta, g, root=root) if r["grid"]["rule"] in DIMENSION_RULES]
    if not rows:
        return None
    readout = {id(r): theory.dimension_entropy(r) for r in rows}
    out = dict(edge=rows[0]["edge"], all_converged=all(r["all_converged"] for r in rows), files=[r["_file"] for r in rows])
    if is_continuous(delta) and g in EXTRAPOLATED_G:
        by_seed = {}
        for r in rows:
            by_seed.setdefault(r["config"]["seed"], {})[r["config"]["m"]] = r
        out["unpaired_seeds"] = sorted(k for k, v in by_seed.items() if set(v) != set(EXTRAPOLATION_M))
        by_seed = {k: v for k, v in by_seed.items() if set(v) == set(EXTRAPOLATION_M)}
        if not by_seed:
            return None
        coarse, fine = EXTRAPOLATION_M
        vals = {k: [] for k in ("ky_dimension", "entropy_rate")}
        per_m = {(k, m): [] for k in vals for m in EXTRAPOLATION_M}
        for seed, v in sorted(by_seed.items()):
            a, b = v[coarse], v[fine]
            for k in vals:
                vals[k].append(theory.extrapolate_to_zero_spacing(readout[id(a)][k], a["config"]["T"] / coarse,
                                                                  readout[id(b)][k], b["config"]["T"] / fine))
                for m in EXTRAPOLATION_M:
                    per_m[(k, m)].append(readout[id(v[m])][k])
        out.update(extrapolated=True, seeds=sorted(by_seed), n_scrambles=len(by_seed),
                   settings=dict(_settings(by_seed[min(by_seed)][fine]), m=list(EXTRAPOLATION_M)))
        for (k, m), v in per_m.items():
            out[f"{k}_m{m}"] = float(np.mean(v))
    else:
        settings = [_settings(r) for r in rows]
        assert all(s == settings[0] for s in settings), f"mixed theory settings at delta={delta}, g={g}"
        vals = {k: [readout[id(r)][k] for r in rows] for k in ("ky_dimension", "entropy_rate")}
        out.update(extrapolated=False, seeds=[r["config"]["seed"] for r in rows], n_scrambles=len(rows),
                   settings=settings[0])
    for k, v in vals.items():
        out[k], out[k + "_sem"] = _mean_sem(v)
    return out


def network_point(delta, g: float, root=None) -> dict | None:
    """Median D_KY/N, h_KS/N and maximum exponent over networks at N = :func:`dimension_network_N` (``J_ii``
    kept: zero-diagonal runs are excluded)."""
    N = dimension_network_N(delta, g)
    rows = [r for r in data.networks(delta, g, N=N, root=root) if not r.get("zero_diagonal")]
    if not rows:
        return None
    out = dict(N=N, n_seeds=len(rows), seeds=[r["params"]["seed"] for r in rows], files=[r["_file"] for r in rows])
    for k in ("ky_dimension", "entropy_rate", "max_exponent"):
        out[k] = float(np.median([r["summaries"][k] for r in rows]))
    return out


def pr_simulation_point(delta, g: float, root=None) -> dict | None:
    """Mean cross-block PR^x/N and PR^phi/N over networks at the largest N."""
    rows = data.pr_simulations(delta, g, root=root)
    if not rows:
        return None
    N = max(r["N"] for r in rows)
    rows = [r for r in rows if r["N"] == N]
    return dict(N=N, n_seeds=len(rows), seeds=[r["seed"] for r in rows], files=[r["_file"] for r in rows],
                pr_x=float(np.mean([r["x"]["pr_cross"] for r in rows])),
                pr_phi=float(np.mean([r["phi"]["pr_cross"] for r in rows])))


def dimension_entropy_table(root=None) -> dict:
    """Fig. 2 table.  Returns::

        points:          {"delta0.5_g3": point, ...}  for every (delta, g) with theory, delta in DIMENSION_DELTAS
                         point = dict(delta, g, theory=theory_point(...), network=network_point(...),
                                      pr_sim=pr_simulation_point(...), pr_theory={pr_x, pr_phi} at g or None)
        pr_theory_dense: {0.0 | 0.25 | 0.5: {g: [...], pr_x: [...], pr_phi: [...]}}    (the PR theory lines)

    ``delta = 0.0`` is delta -> 0.  Any entry may be None where no result exists.  Points are sorted by (delta, g).
    """
    points, dense = {}, {}
    for d in DIMENSION_DELTAS:
        rows = (data.pr_theory(d, root=root) or {}).get("rows", [])
        dense[d] = dict(g=[x["g"] for x in rows], pr_x=[x["pr_x"] for x in rows], pr_phi=[x["pr_phi"] for x in rows])
        gs = sorted({r["g"] for r in data.theory_scrambles(d, root=root) if r["grid"]["rule"] in DIMENSION_RULES})
        for g in gs:
            at_g = [x for x in rows if abs(x["g"] - g) < 1e-9]
            points[_key(d, g)] = dict(delta=d, g=g, theory=theory_point(d, g, root), network=network_point(d, g, root),
                                      pr_sim=pr_simulation_point(d, g, root),
                                      pr_theory=dict(pr_x=at_g[0]["pr_x"], pr_phi=at_g[0]["pr_phi"]) if at_g else None)
    return dict(points=points, pr_theory_dense=dense)


# ---- Fig. 4c ------------------------------------------------------------------------------------------------------

def max_exponent_table(root=None) -> dict:
    """``{"0.25" | "0.5" | "1" | "0": {g: dict(N, n, median, seeds, files)}}``, g ascending; ``"0"`` is delta -> 0
    (RK4).  Median maximum exponent over all networks at N = :func:`max_exponent_N`; zero-diagonal runs are excluded."""
    out = {}
    for key, d in MAX_EXPONENT_KEYS.items():
        runs = [r for r in data.networks(d, root=root) if not r.get("zero_diagonal")]
        out[key] = {}
        for g in sorted({r["g"] for r in runs}):
            at_g = [r for r in runs if r["g"] == g]
            N = max_exponent_N(g, {r["params"]["N"] for r in at_g})
            sel = [r for r in at_g if r["params"]["N"] == N]
            out[key][g] = dict(N=N, n=len(sel), median=float(np.median([r["summaries"]["max_exponent"] for r in sel])),
                               seeds=[r["params"]["seed"] for r in sel], files=[r["_file"] for r in sel])
    return out


# ---- checks -------------------------------------------------------------------------------------------------------

def grid_spacing_table(root=None) -> dict:
    """delta -> 0, g = 20, seed 0: D_KY/N and h_KS/N against the grid spacing T/m (``theory_checks/``), the value at
    spacing 0.1 predicted from 0.2 and 0.125 by the quadratic extrapolation, and the extrapolated value.

    Returns ``dict(rows=[{T, m, spacing, ky_dimension, entropy_rate, file}], ky_dimension_predicted_at_0.1,
    ky_dimension_extrapolated, entropy_rate_...)``, or None without the m = 160 and 256 runs."""
    rows = sorted((dict(T=r["config"]["T"], m=r["config"]["m"], spacing=r["config"]["T"] / r["config"]["m"],
                        **{k: v for k, v in theory.dimension_entropy(r).items() if k in ("ky_dimension", "entropy_rate")},
                        file=r["_file"])
                   for r in data.theory_scrambles(0, 20.0, root=root, subdir="theory_checks")),
                  key=lambda x: (x["T"], -x["spacing"]))
    at = {(x["T"], x["m"]): x for x in rows}
    if (32.0, 160) not in at or (32.0, 256) not in at:
        return None
    a, b = at[(32.0, 160)], at[(32.0, 256)]
    out = dict(rows=rows)
    for k in ("ky_dimension", "entropy_rate"):
        ext = lambda h: b[k] + (a[k] - b[k]) * (h**2 - 0.125**2) / (0.2**2 - 0.125**2)
        out[k + "_predicted_at_0.1"] = ext(0.1)
        out[k + "_extrapolated"] = theory.extrapolate_to_zero_spacing(a[k], 0.2, b[k], 0.125)
    return out


# ---- CLI ----------------------------------------------------------------------------------------------------------

def _print(spectra, dim, mx, grid):
    print("Fig. 1 (spectra): theory scrambles, edge, D_KY/N (theory | network N=4096)")
    for k, v in spectra.items():
        t = v["theory"]
        if t is None or v["network"] is None:
            print(f"  {k:16s} (missing)")
            continue
        D = theory.dimension_entropy(dict(s=t["s"], F=np.array(t["F_all"]), edge=t["edge"]))["ky_dimension"]
        print(f"  {k:16s} n={t['n_scrambles']}  edge={t['edge']:.4f}  D/N {D:.4f} | {v['network']['ky_dimension']:.4f}")
    print("Fig. 2 (dimension_entropy): D_KY/N, h_KS/N theory (+-sem) | network median;  PR^x/N, PR^phi/N theory | sim")
    nan = float("nan")
    for k, p in dim["points"].items():
        t, n, s, pt = p["theory"] or {}, p["network"] or {}, p["pr_sim"] or {}, p["pr_theory"] or {}
        print(f"  {k:16s} D {t.get('ky_dimension', nan):.4f}+-{t.get('ky_dimension_sem', nan):.4f} | "
              f"{n.get('ky_dimension', nan):.4f}  H {t.get('entropy_rate', nan):.5f} | {n.get('entropy_rate', nan):.5f}"
              f"  [N={n.get('N')} x{n.get('n_seeds')}, scr {t.get('n_scrambles')}{' extrap' if t.get('extrapolated') else ''}]"
              f"  PRx {pt.get('pr_x', nan):.4f} | {s.get('pr_x', nan):.4f}  PRphi {pt.get('pr_phi', nan):.4f} | "
              f"{s.get('pr_phi', nan):.4f}")
    print("Fig. 4c (max exponent): median over networks (N, n)")
    for key, rows in mx.items():
        print(f"  delta key {key:4s} " + "  ".join(f"g={g:g}:{v['median']:.4f}({v['N']},{v['n']})" for g, v in rows.items()))
    if grid is None:
        return
    print("delta -> 0 grid spacing, g = 20, seed 0:")
    for x in grid["rows"]:
        print(f"  T={x['T']:g} m={x['m']} spacing={x['spacing']:.3f}  D/N={x['ky_dimension']:.5f}  h/N={x['entropy_rate']:.5f}")
    print(f"  predicted at 0.1: {grid['ky_dimension_predicted_at_0.1']:.5f} / {grid['entropy_rate_predicted_at_0.1']:.5f};"
          f" extrapolated: {grid['ky_dimension_extrapolated']:.5f} / {grid['entropy_rate_extrapolated']:.5f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="results directory (default: the bundled data/)")
    p.add_argument("--json", default=None, help="also write all tables to this JSON file")
    a = p.parse_args(argv)
    tabs = dict(spectra=spectra_table(a.root), dimension_entropy=dimension_entropy_table(a.root),
                max_exponent=max_exponent_table(a.root), grid_spacing=grid_spacing_table(a.root))
    _print(tabs["spectra"], tabs["dimension_entropy"], tabs["max_exponent"], tabs["grid_spacing"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(tabs, f, indent=1, default=str)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
