"""Lyapunov spectrum of one simulated network (one coupling matrix J) by the QR method; writes one result JSON.

    python scripts/run_network.py --g 3 --delta 0.5 --N 4096 --seed 0                       # full spectrum (Fig. spectra)
    python scripts/run_network.py --g 20 --delta 0.5 --N 4096 --k 1024 --t-obs 1500         # leading 1024 exponents
    python scripts/run_network.py --g 20 --rk4-dt 0.05 --N 4096 --k 1024 --qr-every 10      # delta -> 0 (RK4)

``--delta``: the Euler map at time step delta.  ``--rk4-dt``: the continuous-time network, integrated with RK4.
J_ij ~ N(0, g^2/N) from a CPU generator seeded with --seed (identical on every device).  GPU recommended.
The paper's settings for every run are listed in reproduce/jobs/*.txt.

Output: <out-dir>/<name>.json with the exponents, spectrum summaries (D_KY/N, h_KS/N, max exponent, split-half
noise), state statistics, settings and provenance; QR block logs go to an .npz only with --save-blocks.
"""
import argparse
from pathlib import Path

from rnn_lyapunov import network as net, observables
from rnn_lyapunov.io import save_result
from rnn_lyapunov.lyapunov import lyapunov_spectrum


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the docstring layout and show defaults."""


p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
p.add_argument("--g", type=float, required=True)
step = p.add_mutually_exclusive_group(required=True)
step.add_argument("--delta", type=float, help="Euler map time step")
step.add_argument("--rk4-dt", type=float, help="continuous time: RK4 step")
p.add_argument("--N", type=int, required=True)
p.add_argument("--k", type=int, default=None, help="leading exponents (default: all N)")
p.add_argument("--t-burn", type=float, default=300.0)
p.add_argument("--t-tangent-burn", type=float, default=200.0)
p.add_argument("--t-obs", type=float, default=2000.0)
p.add_argument("--qr-every", type=int, default=1, help="updates between QR re-orthonormalizations")
p.add_argument("--seed", type=int, default=0, help="J, initial state and initial frame")
p.add_argument("--zero-diagonal", action="store_true", help="set J_ii = 0 (two delta -> 0 runs of Fig. 4c at g = 3)")
p.add_argument("--save-blocks", action="store_true", help="also write the per-block QR logs (.npz)")
p.add_argument("--tag", default="")
p.add_argument("--out-dir", default="results/network")
p.add_argument("--device", default=None)
a = p.parse_args()

from rnn_lyapunov import default_device  # noqa: E402
dev = a.device or default_device()
J = net.make_coupling(a.N, a.g, a.seed, zero_diagonal=a.zero_diagonal, device=dev)
res = lyapunov_spectrum(J, delta=a.delta, rk4_dt=a.rk4_dt, k=a.k, t_burn=a.t_burn, t_tangent_burn=a.t_tangent_burn,
                        t_obs=a.t_obs, qr_every=a.qr_every, seed=a.seed, progress=True)
summ = observables.spectrum_summary(res.exponents, N=a.N)
summ["split_half_rms"] = observables.split_half_rms(res.block_logs, res.block_time)
stats = {k: v for k, v in res.stats.items() if k != "x_final"}
label = f"map_g{a.g:g}_delta{a.delta:g}" if a.delta is not None else f"flow_g{a.g:g}_rk4dt{a.rk4_dt:g}"
name = (f"{label}_N{a.N}" + (f"_k{a.k}" if a.k else "") + f"_seed{a.seed}" + ("_zerodiag" if a.zero_diagonal else "")
        + (f"_{a.tag}" if a.tag else ""))
out = save_result(Path(a.out_dir) / name,
                  dict(kind="network_lyapunov", g=a.g, delta=a.delta,
                       rk4_dt=a.rk4_dt, zero_diagonal=a.zero_diagonal, params=res.params, stats=stats, summaries=summ,
                       exponents=res.exponents,
                       coupling="iid N(0, g^2/N), torch CPU generator manual_seed(seed)"
                       + (", J_ii = 0" if a.zero_diagonal else "")),
                  arrays=dict(block_logs=res.block_logs) if a.save_blocks else None)
print(out, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in summ.items()})
