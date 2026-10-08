"""One-time importer: every stored result that enters a plotted number of the paper, from the original research tree
(``~/Lyapunov``) into the bundled dataset ``data/``, in the output formats of ``scripts/run_*.py``.

    CUDA_VISIBLE_DEVICES= python tools/build_dataset.py          (needs ~/Lyapunov and the ceph arrays; CPU, ~1 min)

Why: the paper's figures are made from these files only (``paper.tables``), and recomputed results written by the
scripts drop into the same pipeline (``root=results``).  Nothing is recomputed except cheap readouts, which are
recomputed with this library and checked against the stored ones: D_KY/N and h_KS/N per regulator (theory), the
spectrum summaries and the split-half noise (networks, from the exponents and QR block logs on ceph).

Which files (the old figure code's rules, applied to the old tree; ``paper.tables`` then needs no selection logic):

  Fig. 1  theory    ``old/results/theory/production/disc_g{3,5}_h{δ}_T*seed*sobol*.json`` (top level; the
                    ``theory_ensemble`` exclusions), δ = 0.05, 0.1, 0.25, 0.5
          networks  ``old/results/network/map_g{g}_h{δ}_N4096_seed0.json`` (full spectra; ``network_runs``)
  Fig. 2  theory    the ``files`` of every point of ``results/gsweep_dimension_entropy_pr.json`` (the
                    consolidation's production selection), plus the m = 160 partner of every δ -> 0 m = 256 file
                    (the extrapolation pairs; the table lists only the m = 256 file)
          networks  ``network_all`` of every point at the plotted N;  PR sims ``pr_sim_all`` at the plotted N
          PR theory ``results/participation_theory/pr_{h0.25,h0.5,flow}_dz02.json``
  Fig. 4  networks  ``network_max_exponents`` of ``experiments/max_exponent_operator.py`` (replicated below,
                    including its directory/glob order, which decides ties between equal observation times), except
                    that zero-diagonal runs are excluded (owner's decision 2026-10-06; no plotted median changes)
          theory    ``results/max_exponent_operator.json`` of this repo (scripts/run_max_exponent.py; every number of
                    the old results file reproduced exactly, plus the delta = 1, g = 1.2 point added 2026-10-06)
  checks  the δ -> 0 grid-spacing study at g = 20, seed 0 (spacings 0.25, 0.2, 0.125, 0.1; window T = 64)

Output: ``data/{theory,theory_checks,network,participation,participation_theory}/*.json``,
``data/max_exponent_operator.json`` and ``data/manifest.json`` (sha256, size, kind, GPU seconds and the plotted
points that use each file).  Every file's provenance is the original one plus ``imported_from`` (relative to
~/Lyapunov) and ``imported_sha256``.  Large arrays are written one per line (compact JSON).
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rnn_lyapunov import observables, theory

OLD = Path.home() / "Lyapunov"
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MAX_EXP_NEW = REPO / "results" / "max_exponent_operator.json"
FIG1_G, FIG1_DELTAS = (3.0, 5.0), (0.05, 0.1, 0.25, 0.5)
TABLE = OLD / "results" / "gsweep_dimension_entropy_pr.json"
# PR simulations store no run time: estimate from one calibration (RK4 step 0.1, N = 16384, t = 20300: 781 s for
# 812,000 matrix-vector products, results/logs/flow), i.e. seconds = MATVEC_S * (N/16384)^2 * products.
MATVEC_S = 781.0 / 812000


# ---- small helpers ------------------------------------------------------------------------------------------------

def rel(path) -> str:
    return str(Path(path).resolve().relative_to(OLD))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(x) -> str:
    """Shortest exact decimal of a number, as in file names (``0.5``, ``20``)."""
    return f"{x:g}" if float(f"{x:g}") == x else repr(float(x))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def write(path: Path, obj: dict):
    """One top-level key per line, values compact: readable and small."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = jsonable(obj)
    lines = [f" {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}" for k, v in obj.items()]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n")


def provenance(r: dict, src, **extra) -> dict:
    return dict(r.get("provenance", {}), **extra, imported_from=rel(src), imported_sha256=sha256(src))


# ---- selection (the old figure code's rules) ----------------------------------------------------------------------

def fig1_theory() -> dict:
    """{(g, delta): [files]} as ``old/src/plotting.theory_ensemble`` globbed them (run from ~/Lyapunov/old)."""
    out = {}
    for g in FIG1_G:
        for d in FIG1_DELTAS:
            fs = sorted(f for f in glob.glob(str(OLD / f"old/results/theory/production/disc_g{g:g}_h{d:g}_T*seed*sobol*.json"))
                        if "hto0" not in f and "dtstudy" not in f and "_T16_" not in f)
            assert fs and not any(re.search(r"_c\d+\.json$", f) for f in fs)
            out[(g, d)] = fs
    return out


def fig1_networks() -> dict:
    """{(g, delta): file}: the full N = 4096 spectrum of seed 0 (``plotting.network_runs("map", g, delta)``)."""
    out = {}
    for g in FIG1_G:
        for d in FIG1_DELTAS:
            fs = [f for f in glob.glob(str(OLD / f"old/results/network/map_g{g:g}_h{d:g}_N*_seed0.json"))
                  if (p := json.load(open(f))["params"])["full_spectrum"] and p["N"] == 4096]
            assert len(fs) == 1
            out[(g, d)] = fs[0]
    return out


def fig2_selection(table: dict):
    """Theory, network and PR-simulation files of every plotted point of the old Fig. 2 table."""
    theory_files, nets, prs = defaultdict(list), defaultdict(list), defaultdict(list)
    for key, p in table["points"].items():
        files = p["theory"]["files"]
        if p["theory"]["source"].startswith("quadratic extrapolation"):
            files = [f.replace("_T32_m256_", "_T32_m160_") for f in files] + files
        theory_files[key] = [str(OLD / f) for f in files]
        nets[key] = [str(OLD / x["file"]) for x in p["network_all"] if x["N"] == p["network"]["N"]]
        prs[key] = [str(OLD / x["file"]) for x in p["pr_sim_all"] if x["N"] == p["pr_sim"]["N"]]
        assert len(nets[key]) == p["network"]["n_seeds"] and len(prs[key]) == p["pr_sim"]["n_seeds"]
    return theory_files, nets, prs


def fig4_networks() -> dict:
    """{(key, g): [files]}: ``network_max_exponents`` of experiments/max_exponent_operator.py, verbatim in effect.

    Per (g, N, seed) the run with the longest observation time (first in glob order on ties); N = 4096 for g >= 2,
    else the largest N; all runs at that N.  Zero-diagonal runs (J_ii = 0, an older experiment) are excluded."""
    out = {}
    for key, scheme, steps in (("0.25", "map", {0.25}), ("0.5", "map", {0.5}), ("1", "map", {1.0}),
                               ("0", "rk4", {0.05, 0.1})):
        files = glob.glob(str(OLD / "results/network" / f"{scheme}_g*_h*_N*_seed*.json")) + \
            glob.glob(str(OLD / "old/results/network" / f"{scheme}_g*_h*_N*_seed*.json"))
        best = {}
        for f in files:
            m = re.search(r"_g([\d.]+)_h([\d.]+)_N(\d+)(?:_k\d+)?_seed(\d+)", Path(f).name)
            g, h, N, seed = float(m[1]), float(m[2]), int(m[3]), int(m[4])
            if h not in steps or json.load(open(f)).get("zero_diagonal"):
                continue
            t_obs = json.load(open(f))["params"]["t_obs"]
            if (g, N, seed) not in best or t_obs > best[(g, N, seed)][0]:
                best[(g, N, seed)] = (t_obs, f)
        for g in sorted({k[0] for k in best}):
            Ns = sorted({k[1] for k in best if k[0] == g})
            N = 4096 if g >= 2 and 4096 in Ns else Ns[-1]
            out[(key, g)] = sorted(v[1] for k, v in best.items() if k[0] == g and k[1] == N)
    return out


def check_files() -> list:
    """δ -> 0, g = 20, seed 0: spacings T/m = 0.25 (T = 64), 0.2, 0.125, 0.1 (T = 32); window T = 64 at 0.2."""
    fc, gl = OLD / "results/theory/flow_checks", OLD / "results/theory/gsweep_large"
    return [str(gl / "cont_g20_T64_m256_S4096_tw4_upper_seed0_gsweep.json")] + \
        [str(fc / f"cont_g20_T32_m{m}_S4096_tw4_upper_seed0_check_dt.json") for m in (160, 256, 320)] + \
        [str(fc / "cont_g20_T64_m320_S4096_tw4_upper_seed0_check_window.json")]


# ---- converters (old format -> scripts/run_*.py format) -----------------------------------------------------------

def convert_theory(src: str) -> tuple[str, dict]:
    r = json.load(open(src))
    c = dict(r["config"])
    c["delta"] = c.pop("h")
    c["eps"] = tuple(c["eps"])
    cfg = theory.TheoryConfig(**c)
    assert cfg.sobol and cfg.warm_start_in_s and cfg.sample_m is None and cfg.anderson == 6 and cfg.dmft_dt == 0.025
    s, edge = np.array(r["s"]), r["edge"]
    s_u = theory.uncoupled_exponent(cfg.delta)
    if "grid" in r:                                            # gsweep driver (archived convergence.py)
        grid = {k: r["grid"][k] for k in ("rule", "s_low", "n_below", "n_above", "spacing")}
        if "quadrature" in r:
            assert r["quadrature"]["dz"] == r["dmft"]["params"]["quadrature"]["dz"]
    elif "--s-top" in (a := r["provenance"]["argv"]):          # first sweep: old run_theory.py --s-top LO
        grid = dict(rule="top", s_low=float(a[a.index("--s-top") + 1]), n_below=12, n_above=12, spacing=0.05,
                    inferred="old run_theory.py --s-top (12 + 12 thresholds)")
    else:                                                      # Fig. 1 production: the automatic (full) grid
        grid = dict(rule="full", s_low=float(s[0]), n_below=12, n_above=12, spacing=0.05,
                    inferred="old run_theory.py automatic grid (= thresholds_full)")
    grid.update(s_u=s_u, n_thresholds=len(s))
    rebuilt = {"full": lambda: theory.thresholds_full(cfg, edge),
               "upper": lambda: theory.thresholds_upper(cfg, edge, n_above=grid["n_above"], spacing=grid["spacing"]),
               "top": lambda: theory.thresholds_top(cfg, grid["s_low"], edge, n_below=grid["n_below"],
                                                    n_above=grid["n_above"])}[grid["rule"]]()
    grid["reproduces_thresholds"] = bool(len(rebuilt) == len(s) and np.abs(rebuilt - s).max() < 1e-9)
    res = {k: r[k] for k in ("s", "eps", "thetas", "theta_weights", "F", "F_theta", "twisted", "readout_parts",
                             "iterations", "residual", "trace_identity_error", "edge", "dmft", "gain_sample",
                             "all_converged", "seconds")}
    res["n_thresholds_done"] = r.get("n_thresholds_done", int(np.isfinite(np.array(r["F"])[:, -1]).sum()))
    res["config"] = asdict(cfg)
    by_eps = [dict(eps=e, **theory.dimension_entropy(res, j)) for j, e in enumerate(cfg.eps)]
    for new, old in zip(by_eps, r.get("dimension_entropy_by_eps", [])):     # stored by the gsweep driver
        assert all(new[k] == old[k] or (np.isnan(new[k]) and np.isnan(old[k])) for k in ("ky_dimension", "entropy_rate")), src
    extra = {k: r[k] for k in ("driver", "merged_from") if k in r}
    out = dict(kind="theory_cdf", g=cfg.g, delta=cfg.delta, grid=grid, dimension_entropy_by_eps=by_eps,
               config=res.pop("config"), **res, provenance=provenance(r, src, **extra))
    d = "delta0" if cfg.continuous else f"delta{fmt(cfg.delta)}"
    return f"theory_{d}_g{fmt(cfg.g)}_T{fmt(cfg.T)}_m{cfg.grid()}_{grid['rule']}_seed{cfg.seed}.json", out


SUMMARY_KEYS = dict(ky_dimension="kaplan_yorke_fraction", entropy_rate="positive_sum", max_exponent="max_exponent",
                    unstable_fraction="unstable_fraction", mean_exponent="mean_exponent", split_half_rms="split_half_rms")


def convert_network(src: str, mismatches: list) -> tuple[str, dict]:
    r = json.load(open(src))
    p = r["params"]
    z = np.load(r["arrays_file"])
    lam = np.asarray(z["exponents"], np.float64)
    flow = p["scheme"] == "rk4"
    step = p["h"]
    params = dict(N=p["N"], k=p["k"], scheme="flow" if flow else "map", step=step, t_burn=p["t_burn"],
                  t_tangent_burn=p["t_tangent_burn"], t_obs=p["t_obs"], qr_every=p["qr_every"], seed=p["seed"],
                  full_spectrum=p["full_spectrum"], seconds=p["seconds"], device=p["device"])
    summ = observables.spectrum_summary(lam, N=p["N"])
    summ["split_half_rms"] = observables.split_half_rms(z["block_logs"], p["qr_every"] * step)
    for new, old in SUMMARY_KEYS.items():
        a, b = summ[new], r["summaries"][old]
        if not (a == b or (np.isnan(a) and np.isnan(b))):
            mismatches.append((rel(src), new, a, b))
    coupling = "iid N(0, g^2/N), torch CPU generator manual_seed(seed)"
    out = dict(kind="network_lyapunov", g=r["g"], delta=None if flow else step, rk4_dt=step if flow else None,
               params=params, stats={k: v for k, v in r["stats"].items() if k != "x_final"}, summaries=summ,
               exponents=lam, coupling=coupling)
    if r.get("zero_diagonal"):
        out.update(coupling=coupling + ", then J_ii = 0", zero_diagonal=True)
    out["provenance"] = provenance(r, src, arrays_file=r["arrays_file"])
    label = f"flow_g{fmt(r['g'])}_rk4dt{fmt(step)}" if flow else f"map_g{fmt(r['g'])}_delta{fmt(step)}"
    zd = "_zerodiag" if r.get("zero_diagonal") else ""
    return f"{label}_N{p['N']}" + (f"_k{p['k']}" if p["k"] < p["N"] else "") + f"_seed{p['seed']}{zd}.json", out


def convert_pr_sim(src: str) -> tuple[str, dict, float]:
    r = json.load(open(src))
    flow, step, every = r["scheme"] == "flow", r["h"], r["sample_every_steps"]
    assert not r["zero_diagonal"]
    assert r["samples_per_block"] == round(r["t_obs"] / (step * every) / r["blocks"])
    sample_dt = every * step
    assert max(1, round(sample_dt / step)) == every
    out = dict(kind="participation_simulation", g=r["g"], delta=None if flow else step, rk4_dt=step if flow else None,
               N=r["N"], seed=r["seed"], t_burn=r["t_burn"], t_obs=r["t_obs"], blocks=r["blocks"],
               sample_dt=sample_dt, x=r["x"], phi=r["phi"], sample_every_steps=every,
               samples_per_block=r["samples_per_block"],
               provenance=provenance(r, src, driver=r.get("driver"), seconds="not recorded"))
    label = f"flow_g{fmt(r['g'])}_rk4dt{fmt(step)}" if flow else f"map_g{fmt(r['g'])}_delta{fmt(step)}"
    products = round((r["t_burn"] + r["t_obs"]) / step) * (4 if flow else 1)
    return f"pr_{label}_N{r['N']}_seed{r['seed']}.json", out, MATVEC_S * (r["N"] / 16384) ** 2 * products


def convert_pr_theory(src: str) -> tuple[str, dict]:
    r = json.load(open(src))
    rename = dict(alpha="mean_gain", g_alpha="g_mean_gain")
    rows = [{rename.get(k, k): v for k, v in row.items()} for row in r["rows"]]
    out = dict(kind="participation_theory", delta=r["h"], min_points=r["min_points"], tmax_factor=r["tmax_factor"],
               quadrature=r["quadrature"], rows=rows,
               provenance=provenance(r, src, driver=r["driver"], reference=r["reference"],
                                     min_points_onset=r["min_points_onset"]))
    return f"pr_theory_delta{'0' if r['h'] is None else fmt(r['h'])}.json", out


# ---- build --------------------------------------------------------------------------------------------------------

def delta_label(delta) -> str:
    return "δ→0" if delta in (None, 0.0) else f"δ={fmt(delta)}"


def key_label(delta, g) -> str:
    return f"{delta_label(delta)} g={fmt(g)}"


def main():
    if not OLD.exists() or not TABLE.exists() or not MAX_EXP_NEW.exists():
        sys.exit(f"needs {OLD} (the original research tree) and {MAX_EXP_NEW}; the bundled data/ is already built")
    if DATA.exists():
        shutil.rmtree(DATA)
    manifest, written, mismatches = {}, {}, []

    def add(subdir, name, obj, src, use, seconds):
        path = f"{subdir}/{name}"
        if path in written:
            assert written[path] == src, f"name collision: {path} from {src} and {written[path]}"
        else:
            write(DATA / path, obj)
            written[path] = src
            manifest[path] = dict(kind=obj["kind"], source=rel(src), gpu_seconds=seconds, uses=[])
        manifest[path]["uses"].append(use)

    for (g, d), fs in fig1_theory().items():
        for f in fs:
            name, out = convert_theory(f)
            add("theory", name, out, f, f"Fig. 1 theory {key_label(d, g)}", out["seconds"])
    for (g, d), f in fig1_networks().items():
        name, out = convert_network(f, mismatches)
        add("network", name, out, f, f"Fig. 1 simulation {key_label(d, g)}", out["params"]["seconds"])

    table = json.load(open(TABLE))
    th, nets, prs = fig2_selection(table)
    for key, p in table["points"].items():
        d, g = (None if p["h"] == 0 else p["h"]), p["g"]
        for f in th[key]:
            name, out = convert_theory(f)
            m = f" (m={out['config']['m']}, extrapolation pair)" if p["theory"]["source"].startswith("quadratic") else ""
            add("theory", name, out, f, f"Fig. 2 theory D_KY/N, h_KS/N {key_label(d, g)}{m}", out["seconds"])
        for f in nets[key]:
            name, out = convert_network(f, mismatches)
            add("network", name, out, f, f"Fig. 2 simulation D_KY/N, h_KS/N {key_label(d, g)}", out["params"]["seconds"])
        for f in prs[key]:
            name, out, sec = convert_pr_sim(f)
            add("participation", name, out, f, f"Fig. 2 simulation PR^x/N, PR^phi/N {key_label(d, g)}", sec)
    for tag in ("h0.25", "h0.5", "flow"):
        f = str(OLD / f"results/participation_theory/pr_{tag}_dz02.json")
        name, out = convert_pr_theory(f)
        add("participation_theory", name, out, f, f"Fig. 2 theory PR^x/N, PR^phi/N (dense g) "
            f"{delta_label(out['delta'])}", sum(x["seconds"] + x["dmft_seconds"] for x in out["rows"]))

    for (key, g), fs in fig4_networks().items():
        for f in fs:
            name, out = convert_network(f, mismatches)
            add("network", name, out, f, f"Fig. 4c simulation max exponent δ key {key} g={fmt(g)}",
                out["params"]["seconds"])

    for f in check_files():
        name, out = convert_theory(f)
        add("theory_checks", name, out, f, "δ→0 grid-spacing study g=20 (supports the extrapolation)", out["seconds"])

    mx = json.load(open(MAX_EXP_NEW))
    assert mx["kind"] == "max_exponent_operator"
    write(DATA / "max_exponent_operator.json", mx)
    manifest["max_exponent_operator.json"] = dict(
        kind=mx["kind"], source=f"rnn-lyapunov {MAX_EXP_NEW.relative_to(REPO)} (scripts/run_max_exponent.py; "
        "every number of ~/Lyapunov/results/max_exponent_operator.json reproduced exactly; plus delta = 1, g = 1.2)", gpu_seconds=None,
        uses=["Fig. 4a,b operator spectra", "Fig. 4c theory s_*(g)"])

    for path, v in manifest.items():
        v.update(sha256=sha256(DATA / path), bytes=(DATA / path).stat().st_size)
    total = sum(v["bytes"] for v in manifest.values())
    man = dict(kind="manifest", description="Every result file that enters a plotted number of the paper (built by "
               "tools/build_dataset.py from ~/Lyapunov; verified by tools/verify_dataset.py)", total_bytes=total,
               files={k: manifest[k] for k in sorted(manifest)})
    (DATA / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(manifest)} files, {total / 1e6:.1f} MB in {DATA}")
    for k in ("theory", "theory_checks", "network", "participation", "participation_theory"):
        print(f"  {k}: {sum(p.startswith(k + '/') for p in manifest)}")
    print("theory files whose thresholds no grid rule reproduces:",
          [p for p, v in manifest.items() if v["kind"] == "theory_cdf"
           and not json.loads((DATA / p).read_text())["grid"]["reproduces_thresholds"]])
    print("network summary mismatches (new vs stored):", len(mismatches))
    for x in mismatches:
        print("  ", x)


if __name__ == "__main__":
    main()
