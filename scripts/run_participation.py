"""Participation-ratio dimensions PR^x/N and PR^phi/N of one simulated network; writes one result JSON.

    python scripts/run_participation.py --g 3 --delta 0.5 --N 16384 --seed 0         # Euler map
    python scripts/run_participation.py --g 3 --rk4-dt 0.1 --N 16384 --seed 0        # delta -> 0 (RK4)

Covariances of x and phi(x) are accumulated in --blocks disjoint time blocks (a sample every --sample-dt time
units); the plotted estimator is the cross-block one (unbiased for independent blocks).  Paper settings: N = 16384,
t_obs = 20000, 8 blocks, 2 networks (seeds 0, 1).  GPU recommended (N = 16384 needs ~20 GB).
"""
import argparse
from pathlib import Path

from rnn_lyapunov import default_device, network as net
from rnn_lyapunov.io import save_result
from rnn_lyapunov.participation import pr_simulation


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the docstring layout and show defaults."""


p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
p.add_argument("--g", type=float, required=True)
step = p.add_mutually_exclusive_group(required=True)
step.add_argument("--delta", type=float, help="Euler map time step")
step.add_argument("--rk4-dt", type=float, help="continuous time: RK4 step")
p.add_argument("--N", type=int, required=True)
p.add_argument("--t-burn", type=float, default=300.0)
p.add_argument("--t-obs", type=float, default=20000.0)
p.add_argument("--blocks", type=int, default=8)
p.add_argument("--sample-dt", type=float, default=0.5, help="time between stored samples (>= one update)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--out-dir", default="results/participation")
p.add_argument("--device", default=None)
a = p.parse_args()

J = net.make_coupling(a.N, a.g, a.seed, device=a.device or default_device())
r = pr_simulation(J, delta=a.delta, rk4_dt=a.rk4_dt, t_burn=a.t_burn, t_obs=a.t_obs, n_blocks=a.blocks,
                  sample_dt=a.sample_dt, seed=a.seed)
label = f"map_g{a.g:g}_delta{a.delta:g}" if a.delta is not None else f"flow_g{a.g:g}_rk4dt{a.rk4_dt:g}"
name = f"pr_{label}_N{a.N}_seed{a.seed}" + (f"_{a.tag}" if a.tag else "")
out = save_result(Path(a.out_dir) / name,
                  dict(kind="participation_simulation", g=a.g, delta=a.delta, rk4_dt=a.rk4_dt, N=a.N, seed=a.seed,
                       t_burn=a.t_burn, t_obs=a.t_obs, blocks=a.blocks, sample_dt=a.sample_dt, **r))
print(out, "PR^x/N", round(r["x"]["pr_cross"], 5), "PR^phi/N", round(r["phi"]["pr_cross"], 5))
