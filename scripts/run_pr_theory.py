"""Participation-ratio theory PR^x/N and PR^phi/N against g (two-site cavity theory); writes one result JSON.

    python scripts/run_pr_theory.py --delta 0.5          # Euler map, the paper's 30 values of g
    python scripts/run_pr_theory.py --delta 0            # delta -> 0

Inputs are the stationary DMFT two-point functions only.  Paper settings: Gaussian quadrature step dz = 0.02 (the
default here), frequency grid 16384, DMFT lag range 320 (g < 1.4), 160 (g <= 2), 80 (otherwise).  GPU recommended.
"""
import argparse
import time
from pathlib import Path

from rnn_lyapunov import default_device, dmft, is_continuous
from rnn_lyapunov.io import save_result
from rnn_lyapunov.participation import pr_theory

G_PAPER = [1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def dmft_tmax(g: float) -> float:
    """DMFT lag range: longer near onset, where correlations decay slowly."""
    return 320.0 if g < 1.4 else 160.0 if g <= 2 else 80.0


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the docstring layout and show defaults."""


p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
p.add_argument("--delta", type=float, required=True, help="time step; 0 = continuous time")
p.add_argument("--g", type=float, nargs="+", default=G_PAPER)
p.add_argument("--min-points", type=int, default=16384, help="frequency grid size after zero padding")
p.add_argument("--quad-dz", type=float, default=0.02, help="Gaussian quadrature step (standard units)")
p.add_argument("--tmax-factor", type=float, default=1.0, help="multiply the DMFT lag range (convergence check)")
p.add_argument("--tag", default="")
p.add_argument("--out-dir", default="results/participation_theory")
p.add_argument("--device", default=None)
a = p.parse_args()

dev = a.device or default_device()
ga = dmft.GaussianAverages(dz=a.quad_dz, device=dev)
rows = []
for g in a.g:
    tmax = dmft_tmax(g) * a.tmax_factor
    t0 = time.monotonic()
    sol = dmft.cached(g, a.delta, t_max=tmax, dt=0.025, ga=ga)
    r = pr_theory(sol, dmft.mean_gain(sol, ga), min_points=a.min_points, device=dev)
    r.update(g=g, q=sol.q, dmft_tmax=tmax, dmft_residual=sol.residual, dmft_seconds=time.monotonic() - t0)
    rows.append(r)
    print(f"delta={a.delta:g} g={g:g}  PR^x/N={r['pr_x']:.6f}  PR^phi/N={r['pr_phi']:.6f}", flush=True)
d = "delta0" if is_continuous(a.delta) else f"delta{a.delta:g}"
out = save_result(Path(a.out_dir) / (f"pr_theory_{d}" + (f"_{a.tag}" if a.tag else "")),
                  dict(kind="participation_theory", delta=None if is_continuous(a.delta) else a.delta,
                       min_points=a.min_points, tmax_factor=a.tmax_factor, quadrature=ga.params, rows=rows))
print(out)
