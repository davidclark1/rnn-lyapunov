"""Write ``reproduce/jobs/*.txt``: the commands that regenerate every file of ``data/`` from scratch, one per line.

    python tools/make_jobs.py                  (reads data/ only; no compute)

The commands are generated from the stored settings of each data file (``config``/``grid`` of theory files,
``params`` of network runs, ...), so running them writes the same file names under ``results/`` and
``python -m paper.tables --root results`` rebuilds the paper's tables from them.  Theory thresholds use the named
grid rule (``--grid full|upper|top``) only where the rule, evaluated with the stored upper edge, reproduces the stored
thresholds to 1e-9 (``grid.reproduces_thresholds``, checked at import); otherwise ``--s-values`` lists them.

Each data file appears once, under the first figure that uses it (Fig. 1, then 2, then 4); a header gives the
GPU-hours of the stored runs (participation simulations store no run time: estimated, see tools/build_dataset.py).
Pin one GPU per command (``CUDA_VISIBLE_DEVICES=<id>``); the commands are independent and can run in any order.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
JOBS = REPO / "reproduce" / "jobs"
THEORY_DEFAULTS = dict(twist_aT_max=12.0, tol=1e-7, max_iter=600, dmft_dt=0.025, chunk=512)


def fmt(x) -> str:
    return f"{x:g}" if float(f"{x:g}") == x else repr(float(x))


def theory_cmd(r: dict, out_dir: str) -> str:
    c, gr = r["config"], r["grid"]
    assert all(c[k] == v for k, v in THEORY_DEFAULTS.items()) and c["sobol"] and c["warm_start_in_s"]
    cmd = f"scripts/run_theory.py --g {fmt(c['g'])} --delta {fmt(c['delta'] or 0)} --T {fmt(c['T'])}"
    if c["delta"] is None:
        cmd += f" --m {c['m']}"
    if not gr["reproduces_thresholds"]:
        cmd += " --s-values " + " ".join(repr(float(s)) for s in r["s"])
    elif gr["rule"] == "top":
        cmd += f" --grid top --s-low {fmt(gr['s_low'])} --n-below {gr['n_below']} --n-above {gr['n_above']}"
    elif gr["rule"] == "upper":
        cmd += f" --grid upper --n-above {gr['n_above']} --spacing {fmt(gr['spacing'])}"
    else:
        cmd += " --grid full"
    cmd += f" --paths {c['n_paths']} --twists {c['n_twist']} --eps {' '.join(fmt(e) for e in c['eps'])}"
    return cmd + f" --dmft-tmax {fmt(c['dmft_tmax'])} --seed {c['seed']} --out-dir {out_dir}"


def network_cmd(r: dict) -> str:
    p = r["params"]
    step = f"--rk4-dt {fmt(p['step'])}" if p["scheme"] == "flow" else f"--delta {fmt(p['step'])}"
    cmd = (f"scripts/run_network.py --g {fmt(r['g'])} {step} --N {p['N']}" + (f" --k {p['k']}" if p["k"] < p["N"] else "")
           + f" --t-burn {fmt(p['t_burn'])} --t-tangent-burn {fmt(p['t_tangent_burn'])} --t-obs {fmt(p['t_obs'])}"
           f" --qr-every {p['qr_every']} --seed {p['seed']} --out-dir results/network")
    return cmd + (" --zero-diagonal" if r.get("zero_diagonal") else "")


def pr_sim_cmd(r: dict) -> str:
    step = f"--rk4-dt {fmt(r['rk4_dt'])}" if r["delta"] is None else f"--delta {fmt(r['delta'])}"
    return (f"scripts/run_participation.py --g {fmt(r['g'])} {step} --N {r['N']} --t-burn {fmt(r['t_burn'])}"
            f" --t-obs {fmt(r['t_obs'])} --blocks {r['blocks']} --sample-dt {fmt(r['sample_dt'])} --seed {r['seed']}"
            " --out-dir results/participation")


def pr_theory_cmd(r: dict) -> str:
    assert r["tmax_factor"] == 1.0
    return (f"scripts/run_pr_theory.py --delta {fmt(r['delta'] or 0)} --quad-dz {fmt(r['quadrature']['dz'])}"
            f" --min-points {r['min_points']} --out-dir results/participation_theory")


def natural(path: str) -> list:
    """Sort key with numbers compared as numbers (g2 before g10)."""
    return [float(t) if re.fullmatch(r"\d+(\.\d+)?", t) else t for t in re.split(r"(\d+(?:\.\d+)?)", path)]


def figure_of(uses: list) -> str:
    for tag in ("Fig. 1", "Fig. 2", "Fig. 4"):
        if any(u.startswith(tag) for u in uses):
            return tag[-1]
    return "checks"


def main():
    man = json.loads((DATA / "manifest.json").read_text())["files"]
    groups: dict[str, list] = {}
    for path, v in man.items():
        sub = path.split("/")[0]
        group = {"theory": f"fig{figure_of(v['uses'])}_theory", "network": f"fig{figure_of(v['uses'])}_network",
                 "participation": "fig2_participation", "participation_theory": "fig2_pr_theory",
                 "theory_checks": "checks_theory"}.get(sub, "fig4_network")
        groups.setdefault(group, []).append((path, v))
    JOBS.mkdir(parents=True, exist_ok=True)
    for old in JOBS.glob("*.txt"):
        old.unlink()
    for group, items in sorted(groups.items()):
        lines, seconds = [], 0.0
        for path, v in sorted(items, key=lambda x: natural(x[0])):
            if path == "max_exponent_operator.json":
                lines.append("scripts/run_max_exponent.py --out results/max_exponent_operator")
                continue
            r = json.loads((DATA / path).read_text())
            seconds += v["gpu_seconds"] or 0.0
            lines.append({"theory": lambda: theory_cmd(r, "results/theory"),
                          "theory_checks": lambda: theory_cmd(r, "results/theory_checks"),
                          "network": lambda: network_cmd(r), "participation": lambda: pr_sim_cmd(r),
                          "participation_theory": lambda: pr_theory_cmd(r)}[path.split("/")[0]]())
        note = {"fig2_participation": " (estimated: run times not recorded)",
                "fig4_network": "; Fig. 4c also uses the networks of fig1_network.txt and fig2_network.txt",
                "checks_theory": "; delta -> 0 grid-spacing study at g = 20 (spacings 0.2, 0.125 repeat two Fig. 2 runs, so that the list stands alone)"}
        mx = " + scripts/run_max_exponent.py (minutes)" if group == "fig4_network" else ""
        head = [f"# {group}: {len(items)} data files; GPU-hours {seconds / 3600:.1f}{mx}{note.get(group, '')}",
                "# run from the repository root, one GPU per command (CUDA_VISIBLE_DEVICES=<id>)"]
        name = "fig4.txt" if group == "fig4_network" else f"{group}.txt"
        (JOBS / name).write_text("\n".join(head + lines) + "\n")
        print(f"{name:26s} {len(lines):4d} commands  {seconds / 3600:7.1f} GPU-h")


if __name__ == "__main__":
    main()
