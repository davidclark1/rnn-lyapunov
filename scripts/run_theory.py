"""Single-site theory F(s) at one physical point (g, delta) and one Sobol scramble; writes one result JSON.

    python scripts/run_theory.py --g 3 --delta 0.5 --grid full --seed 0                   # paper Fig. spectra
    python scripts/run_theory.py --g 20 --delta 0.5 --T 64 --grid upper --seed 0          # paper Fig. dimension_entropy
    python scripts/run_theory.py --g 20 --delta 0 --m 160 --grid upper --seed 0           # delta -> 0 (continuous)

GPU strongly recommended (pin one with CUDA_VISIBLE_DEVICES=<id>).  The defaults are the paper's production settings
(T = 32, 4096 Sobol gain trajectories, 4 Bloch twists, regulators 0.1 ... 1.6e-4).  The paper's settings for every
point are listed in reproduce/jobs/*.txt.  No simulated exponent enters.

Output: <out-dir>/<name>.json with F(s) at every regulator and twist, the upper edge, convergence diagnostics,
D_KY/N and h_KS/N at every regulator, the settings, and provenance.  A partial file is rewritten after every
threshold in <out-dir>/partial/ and removed on completion.
"""
import argparse
from pathlib import Path

import numpy as np

from rnn_lyapunov import theory
from rnn_lyapunov.io import save_result


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the docstring layout and show defaults."""


p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
p.add_argument("--g", type=float, required=True, help="coupling strength")
p.add_argument("--delta", type=float, required=True, help="time step; 0 = continuous time (delta -> 0)")
p.add_argument("--T", type=float, default=32.0, help="periodic window (time units); m = T/delta for the map")
p.add_argument("--m", type=int, default=None, help="continuous time only: grid points on the window")
p.add_argument("--grid", choices=["full", "upper", "top"], default="full",
               help="thresholds: full spectrum / upper part, from just above s_u (strong coupling) / "
                    "n-below uniform on [s_low, 0) and n-above uniform on [0, edge)")
p.add_argument("--s-values", type=float, nargs="+", default=None, help="explicit thresholds (overrides --grid)")
p.add_argument("--s-low", type=float, default=None, help="top grid: lowest threshold")
p.add_argument("--n-below", type=int, default=12, help="top grid: thresholds on [s_low, 0)")
p.add_argument("--n-above", type=int, default=12, help="upper/top grid: thresholds on [0, edge)")
p.add_argument("--spacing", type=float, default=0.05, help="upper grid: spacing beyond the first offsets")
p.add_argument("--paths", type=int, default=4096, help="Sobol gain trajectories (power of two)")
p.add_argument("--seed", type=int, default=0, help="Sobol scramble")
p.add_argument("--eps", type=float, nargs="+", default=list(theory.PAPER_EPS), help="decreasing regulators")
p.add_argument("--twists", type=int, default=4, help="Bloch-phase nodes (0: periodic window only)")
p.add_argument("--twist-aT-max", type=float, default=12.0)
p.add_argument("--tol", type=float, default=1e-7)
p.add_argument("--max-iter", type=int, default=600)
p.add_argument("--chunk", type=int, default=512, help="gain trajectories per GPU batch (memory only)")
p.add_argument("--dmft-tmax", type=float, default=80.0, help="DMFT lag range (paper: 320 at g = 1.25, 160 at g = 1.5 and 2, else 80)")
p.add_argument("--dmft-dt", type=float, default=0.025, help="continuous time: DMFT lag grid")
p.add_argument("--tag", default="", help="appended to the file name")
p.add_argument("--out-dir", default="results/theory")
p.add_argument("--device", default=None)
a = p.parse_args()

if a.m is not None and a.delta != 0:
    p.error("--m is for continuous time only (--delta 0); the map uses m = T/delta")
cfg = theory.TheoryConfig(g=a.g, delta=a.delta, T=a.T, m=a.m, n_paths=a.paths, seed=a.seed, eps=tuple(a.eps),
                          n_twist=a.twists, twist_aT_max=a.twist_aT_max, tol=a.tol, max_iter=a.max_iter, chunk=a.chunk,
                          dmft_tmax=a.dmft_tmax, dmft_dt=a.dmft_dt)
edge = theory.edge_of(cfg, a.device)
if a.s_values is not None:
    s, a.grid = np.array(sorted(a.s_values)), "explicit"
elif a.grid == "full":
    s = theory.thresholds_full(cfg, edge)
elif a.grid == "upper":
    s = theory.thresholds_upper(cfg, edge, n_above=a.n_above, spacing=a.spacing)
else:
    assert a.s_low is not None, "--grid top needs --s-low"
    s = theory.thresholds_top(cfg, a.s_low, edge, n_below=a.n_below, n_above=a.n_above)
grid = dict(rule=a.grid, s_low=a.s_low if a.grid == "top" else None, n_below=a.n_below, n_above=a.n_above,
            spacing=a.spacing, s_u=theory.uncoupled_exponent(cfg.delta), n_thresholds=len(s))

d = "delta0" if cfg.continuous else f"delta{a.delta:g}"
name = f"theory_{d}_g{a.g:g}_T{a.T:g}_m{cfg.grid()}_{a.grid}_seed{a.seed}" + (f"_{a.tag}" if a.tag else "")
out_dir = Path(a.out_dir)
partial = out_dir / "partial" / name
res = theory.cdf_curve(cfg, s, device=a.device,
                       checkpoint=lambda r: save_result(partial, dict(kind="theory_cdf_partial", grid=grid, **r)))
by_eps = [dict(eps=e, **theory.dimension_entropy(res, j)) for j, e in enumerate(cfg.eps)]
out = save_result(out_dir / name, dict(kind="theory_cdf", g=a.g, delta=cfg.delta, grid=grid,
                                       dimension_entropy_by_eps=by_eps, **res))
(partial.parent / (partial.name + ".json")).unlink(missing_ok=True)
print(out, "edge", res["edge"], "converged", res["all_converged"], f"{res['seconds']:.0f}s", by_eps[-1])
