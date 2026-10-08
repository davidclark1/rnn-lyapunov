"""Quickstart: Lyapunov spectrum of a random network, simulated and from the single-site theory (CPU, ~2 minutes).

    python examples/quickstart.py

Reduced resolution throughout (N = 512, 256 gain trajectories, window T = 16), so F(s) agrees to about 0.01 and
D_KY/N, h_KS/N to about 10%; the paper's settings (reproduce/jobs/) agree closely and take GPU hours.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # CPU: multithreaded BLAS slows the small batched solves ~100x
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rnn_lyapunov import network, observables, theory  # noqa: E402
from rnn_lyapunov.lyapunov import lyapunov_spectrum  # noqa: E402

g, delta, N = 3.0, 0.5, 512
torch.set_num_threads(8)

# --- simulation: QR method on one network ---------------------------------------------------------------------
J = network.make_coupling(N, g, seed=0)
sim = lyapunov_spectrum(J, delta=delta, t_burn=200, t_tangent_burn=100, t_obs=400, seed=0)
summary = observables.spectrum_summary(sim.exponents)
print(f"simulation, N={N}:  lambda_1={summary['max_exponent']:.3f}  D_KY/N={summary['ky_dimension']:.4f}  "
      f"h_KS/N={summary['entropy_rate']:.4f}")

# --- theory: F(s) of the single-site theory -----------------------------------------------------------------------
cfg = theory.TheoryConfig(g=g, delta=delta, T=16.0, n_paths=256, n_twist=2, eps=(1e-1, 1e-2, 1e-3))
edge = theory.edge_of(cfg)
s = theory.thresholds_upper(cfg, edge, n_above=6, spacing=0.15)
res = theory.cdf_curve(cfg, s, verbose=False)
de = theory.dimension_entropy(res)
print(f"theory (reduced):   s_*={edge:.3f}  D_KY/N={de['ky_dimension']:.4f}  h_KS/N={de['entropy_rate']:.4f}")

# --- compare F(s) at the thresholds --------------------------------------------------------------------------------
F_sim = observables.empirical_cdf(sim.exponents, res["s"])
for si, ft, fs in zip(res["s"], res["F"][:, -1], F_sim):
    print(f"  s={si:+.3f}   F theory={ft:.4f}   F simulation={fs:.4f}")
print("max |F theory - F simulation| =", float(np.max(np.abs(res["F"][:, -1] - F_sim))))
