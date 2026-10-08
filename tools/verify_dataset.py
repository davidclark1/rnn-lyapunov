"""Check every plotted number built by ``paper.tables`` from ``data/`` against the original research tree.

    CUDA_VISIBLE_DEVICES= python tools/verify_dataset.py          (needs ~/Lyapunov; reads JSON/npz only)

References (computed by the OLD code paths, from the OLD files, independently of tools/build_dataset.py):

  Fig. 1  ``theory_ensemble`` of old/src/plotting.py (mean F over the scramble files, as globbed there) and the
          N = 4096 exponents from the npz arrays on ceph
  Fig. 2  ``results/gsweep_dimension_entropy_pr.json`` (experiments/gsweep_consolidate.py): theory D_KY/N, h_KS/N
          and their s.e.m., network medians, PR simulation means, PR theory at each point and the dense PR rows
  Fig. 4  ``network_max_exponents`` of experiments/max_exponent_operator.py (verbatim below), and every number of
          ``results/max_exponent_operator.json`` against data/max_exponent_operator.json

Point sets, network sizes and seed counts must agree exactly; numbers to TOL (relative or absolute).  Prints the
largest differences per quantity and PASS/FAIL (exit code 1 on failure).
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # the repo root, for the paper package
from paper import data, tables  # noqa: E402

OLD = Path.home() / "Lyapunov"
TOL = 1e-12
diffs = defaultdict(lambda: [0.0, 0.0, 0])        # quantity -> [max abs, max rel, count]
failures = []


def compare(what: str, new, old):
    a, b = np.asarray(new, float), np.asarray(old, float)
    if a.shape != b.shape:
        failures.append(f"{what}: shape {a.shape} vs {b.shape}")
        return
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0.0, np.abs(a - b))
    r = np.where(both_nan, 0.0, d / np.maximum(np.abs(b), 1e-300))
    q = what.split(" ")[0]
    diffs[q][0] = max(diffs[q][0], float(np.nanmax(d, initial=0.0)))
    diffs[q][1] = max(diffs[q][1], float(np.nanmax(r, initial=0.0)))
    diffs[q][2] += a.size
    if np.isnan(d).any() or ((d > TOL) & (r > TOL)).any():
        failures.append(f"{what}: max abs diff {np.nanmax(d):.3g}")


def same(what: str, new, old):
    if new != old:
        failures.append(f"{what}: {new!r} vs {old!r}")


# ---- Fig. 1 -------------------------------------------------------------------------------------------------------

def old_theory_ensemble_F(g, d):
    """old/src/plotting.theory_ensemble(pattern)['F'] (run from ~/Lyapunov/old), eps_index = -1."""
    files = sorted(f for f in glob.glob(str(OLD / f"old/results/theory/production/disc_g{g:g}_h{d:g}_T*seed*sobol*.json"))
                   if "hto0" not in f and "dtstudy" not in f and "_T16_" not in f)
    Fall = np.stack([np.array(json.load(open(f))["F"]) for f in files])
    first = json.load(open(files[0]))
    return Fall.mean(0)[:, -1], np.array(first["s"]), first["edge"], len(files)


def check_fig1():
    for key, v in tables.spectra_table().items():
        g, d = v["g"], v["delta"]
        F, s, edge, n = old_theory_ensemble_F(g, d)
        compare(f"fig1_theory_F {key}", v["theory"]["F"], F)
        compare(f"fig1_theory_s {key}", v["theory"]["s"], s)
        compare(f"fig1_theory_edge {key}", v["theory"]["edge"], edge)
        same(f"fig1 n_scrambles {key}", v["theory"]["n_scrambles"], n)
        old = json.load(open(OLD / f"old/results/network/map_g{g:g}_h{d:g}_N4096_seed0.json"))
        lam = np.sort(np.load(old["arrays_file"])["exponents"])[::-1]
        compare(f"fig1_network_exponents {key}", v["network"]["exponents"], lam)


# ---- Fig. 2 -------------------------------------------------------------------------------------------------------

def check_fig2():
    old = json.load(open(OLD / "results/gsweep_dimension_entropy_pr.json"))
    new = tables.dimension_entropy_table()
    to_old = lambda p: f"h{p['delta']:g}_g{p['g']:g}"
    same("fig2 point set", sorted(to_old(p) for p in new["points"].values()), sorted(old["points"]))
    for p in new["points"].values():
        k = to_old(p)
        o = old["points"].get(k)
        if o is None:
            continue
        for q in ("ky_dimension", "ky_dimension_sem", "entropy_rate", "entropy_rate_sem"):
            compare(f"fig2_theory_{q} {k}", p["theory"][q], o["theory"][q])
        same(f"fig2 theory scrambles {k}", sorted(p["theory"]["seeds"]), sorted(o["theory"]["seeds"]))
        for q in ("ky_dimension", "entropy_rate", "max_exponent"):
            compare(f"fig2_network_{q} {k}", p["network"][q], o["network"][q])
        same(f"fig2 network N, n {k}", (p["network"]["N"], p["network"]["n_seeds"]), (o["network"]["N"], o["network"]["n_seeds"]))
        for q in ("pr_x", "pr_phi"):
            compare(f"fig2_pr_sim_{q} {k}", p["pr_sim"][q], o["pr_sim"][q])
            if o["pr_theory"] is not None:
                compare(f"fig2_pr_theory_{q} {k}", p["pr_theory"][q], o["pr_theory"][q])
        same(f"fig2 pr_sim N, n {k}", (p["pr_sim"]["N"], p["pr_sim"]["n_seeds"]), (o["pr_sim"]["N"], o["pr_sim"]["n_seeds"]))
        same(f"fig2 pr_theory present {k}", p["pr_theory"] is None, o["pr_theory"] is None)
    for d, rows in new["pr_theory_dense"].items():
        o = old["pr_theory_dense"][f"h{d:g}"]
        for q in ("g", "pr_x", "pr_phi"):
            compare(f"fig2_pr_theory_dense_{q} h{d:g}", rows[q], o[q])


# ---- Fig. 4 -------------------------------------------------------------------------------------------------------

def network_max_exponents(scheme, h_values):
    """experiments/max_exponent_operator.py, verbatim (ROOT = ~/Lyapunov)."""
    files = glob.glob(str(OLD / "results/network" / f"{scheme}_g*_h*_N*_seed*.json")) + \
        glob.glob(str(OLD / "old/results/network" / f"{scheme}_g*_h*_N*_seed*.json"))
    best = {}
    for f in files:
        m = re.search(r"_g([\d.]+)_h([\d.]+)_N(\d+)(?:_k\d+)?_seed(\d+)", Path(f).name)
        g, h, N, seed = float(m[1]), float(m[2]), int(m[3]), int(m[4])
        if h not in h_values:
            continue
        d = json.load(open(f))
        if d.get("zero_diagonal"):          # deviation from the original: excluded since 2026-10-06 (owner's decision)
            continue
        key = (g, N, seed)
        if key not in best or d["params"]["t_obs"] > best[key][0]:
            best[key] = (d["params"]["t_obs"], d["summaries"]["max_exponent"], h)
    out = {}
    for g in sorted({k[0] for k in best}):
        Ns = sorted({k[1] for k in best if k[0] == g})
        N = 4096 if g >= 2 and 4096 in Ns else Ns[-1]
        vals = [v[1] for k, v in best.items() if k[0] == g and k[1] == N]
        out[g] = dict(N=N, n=len(vals), median=float(np.median(vals)))
    return out


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(o, list) and o and isinstance(o[0], (dict, list)):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def check_fig4():
    old = {"0.25": network_max_exponents("map", {0.25}), "0.5": network_max_exponents("map", {0.5}),
           "1": network_max_exponents("map", {1.0}), "0": network_max_exponents("rk4", {0.05, 0.1})}
    new = tables.max_exponent_table()
    for key in old:
        same(f"fig4 g set {key}", list(new[key]), list(old[key]))
        for g, o in old[key].items():
            if g in new[key]:
                compare(f"fig4_network_median {key} g{g:g}", new[key][g]["median"], o["median"])
                same(f"fig4 N, n {key} g{g:g}", (new[key][g]["N"], new[key][g]["n"]), (o["N"], o["n"]))
    mo = json.load(open(OLD / "results/max_exponent_operator.json"))
    mn = json.loads(json.dumps(data.max_exponent_data()))
    old_g1 = {r["g"] for r in mo["c"]["1"]}
    mn["c"]["1"] = [r for r in mn["c"]["1"] if r["g"] in old_g1]       # delta = 1, g = 1.2 added 2026-10-06
    lo = {p: v for p, v in leaves({k: v for k, v in mo.items() if k != "provenance"})}
    ln = {p: v for p, v in leaves({k: v for k, v in mn.items() if k != "provenance"})}
    same("fig4 operator data keys", sorted(ln), sorted(lo))
    for p, v in lo.items():
        if p in ln and not isinstance(v, str):
            compare(f"fig4_operator_data {p}", np.array(ln[p], dtype=float), np.array(v, dtype=float))


def main():
    if not OLD.exists():
        sys.exit(f"{OLD} (the original research tree) is needed")
    check_fig1()
    check_fig2()
    check_fig4()
    print(f"{'quantity':38s} {'values':>7s} {'max abs diff':>13s} {'max rel diff':>13s}")
    for q, (d, r, n) in sorted(diffs.items()):
        print(f"{q:38s} {n:7d} {d:13.3g} {r:13.3g}")
    for f in failures:
        print("FAIL", f)
    print("PASS" if not failures else f"FAIL ({len(failures)})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
