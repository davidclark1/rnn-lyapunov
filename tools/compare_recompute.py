"""Compare a from-scratch recomputation (``results/``, written by the job lists in reproduce/jobs/) with ``data/``.

    python tools/compare_recompute.py [--results results] [--markdown out.md]

File level: every data file is matched by name in ``results/`` and compared (theory F(s, eps), network exponents,
participation ratios in units of their jackknife standard error, PR theory, maximum-exponent operator data).
Figure level: every plotted number of ``paper.tables`` built from both directories.  Theory is deterministic, so it
should agree to round-off; chaotic simulations agree bitwise only on identical hardware and otherwise within their
statistical errors (docs/NUMERICS.md, "Reproducibility").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from paper import tables  # noqa: E402


def _flat(x):
    return np.asarray(x, float).ravel()


def file_level(data: Path, results: Path) -> dict:
    out = {}
    for sub, kind in (("theory", "theory"), ("theory_checks", "theory"), ("network", "network"),
                      ("participation", "participation"), ("participation_theory", "pr_theory")):
        rows = []
        for f in sorted((data / sub).glob("*.json")):
            g = results / sub / f.name
            if not g.exists():
                rows.append(dict(file=f.name, missing=True))
                continue
            a, b = json.loads(f.read_text()), json.loads(g.read_text())
            r = dict(file=f.name, gpu_old=a["provenance"].get("gpu"), gpu_new=b["provenance"].get("gpu"))
            if kind == "theory":
                r["max_abs"] = float(np.max(np.abs(_flat(a["F"]) - _flat(b["F"]))))
            elif kind == "network":
                n = min(len(a["exponents"]), len(b["exponents"]))
                r["max_abs"] = float(np.max(np.abs(_flat(a["exponents"])[:n] - _flat(b["exponents"])[:n])))
                for q in ("ky_dimension", "entropy_rate", "max_exponent"):
                    r[q + "_rel"] = abs(b["summaries"][q] / a["summaries"][q] - 1) if a["summaries"][q] else 0.0
            elif kind == "participation":
                r["max_abs"] = max(abs(a[x]["pr_cross"] - b[x]["pr_cross"]) for x in ("x", "phi"))
                r["max_in_se"] = max(abs(a[x]["pr_cross"] - b[x]["pr_cross"]) / a[x]["pr_cross_se"] for x in ("x", "phi"))
            else:
                r["max_abs"] = float(np.max(np.abs(_flat([[x["pr_x"], x["pr_phi"]] for x in a["rows"]])
                                                   - _flat([[x["pr_x"], x["pr_phi"]] for x in b["rows"]]))))
            rows.append(r)
        out[sub] = rows
    a = json.loads((data / "max_exponent_operator.json").read_text())
    b = json.loads((results / "max_exponent_operator.json").read_text())
    d = [np.max(np.abs(_flat(a[k][q]) - _flat(b[k][q]))) for k, q in (("a", "E"), ("a", "psi"), ("b", "levels"))]
    d += [np.max(np.abs(_flat([r["s_star"] for r in a["c"][k]]) - _flat([r["s_star"] for r in b["c"][k]]))) for k in a["c"]]
    out["max_exponent_operator"] = [dict(file="max_exponent_operator.json", max_abs=float(max(d)))]
    return out


def figure_level(data: Path, results: Path) -> dict:
    A, B = tables.dimension_entropy_table(root=data), tables.dimension_entropy_table(root=results)
    rel = {}
    for key, p in A["points"].items():
        q = B["points"].get(key)
        for part, quantities in (("theory", ("ky_dimension", "entropy_rate")), ("network", ("ky_dimension", "entropy_rate")),
                                 ("pr_sim", ("pr_x", "pr_phi"))):
            if p.get(part) is None:
                continue
            for x in quantities:
                v = (q or {}).get(part) or {}
                rel.setdefault(f"{part}.{x}", []).append(abs(v.get(x, np.nan) / p[part][x] - 1) if p[part][x] else 0.0)
    S, T = tables.spectra_table(root=data), tables.spectra_table(root=results)
    rel["spectra.theory_F"] = [float(np.max(np.abs(_flat(S[k]["theory"]["F"]) - _flat(T[k]["theory"]["F"])))) for k in S]
    rel["spectra.network_exponents"] = [float(np.max(np.abs(_flat(S[k]["network"]["exponents"])
                                                            - _flat(T[k]["network"]["exponents"])))) for k in S]
    M, N = tables.max_exponent_table(root=data), tables.max_exponent_table(root=results)
    rel["max_exponent.median"] = [abs(N[k][g]["median"] / v["median"] - 1) for k in M for g, v in M[k].items()]
    return {k: dict(n=len(v), max=float(np.nanmax(v)), median=float(np.nanmedian(v)), n_missing=int(np.sum(np.isnan(v))))
            for k, v in rel.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(REPO / "results"))
    p.add_argument("--data", default=str(REPO / "data"))
    p.add_argument("--json", default=None, help="write the full comparison here")
    a = p.parse_args()
    fl = file_level(Path(a.data), Path(a.results))
    print("file level (max |difference|; participation also in standard errors):")
    for sub, rows in fl.items():
        ok = [r for r in rows if not r.get("missing")]
        line = f"  {sub:24s} {len(ok)}/{len(rows)} files"
        if ok:
            line += f"  max {max(r['max_abs'] for r in ok):.2e}  median {np.median([r['max_abs'] for r in ok]):.2e}"
            if sub == "participation":
                line += f"  max {max(r['max_in_se'] for r in ok):.2f} s.e."
            if sub == "network":
                line += "  rel. D_KY {:.1e}, h_KS {:.1e}, lambda_1 {:.1e} (max)".format(
                    *(max(r[q + "_rel"] for r in ok) for q in ("ky_dimension", "entropy_rate", "max_exponent")))
        print(line)
    fig = figure_level(Path(a.data), Path(a.results))
    print("figure level (relative difference of plotted values; spectra/F, exponents: absolute):")
    for k, v in fig.items():
        print(f"  {k:28s} n={v['n']:3d}  max {v['max']:.2e}  median {v['median']:.2e}  missing {v['n_missing']}")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(files=fl, figures=fig), indent=1, default=float))


if __name__ == "__main__":
    main()
