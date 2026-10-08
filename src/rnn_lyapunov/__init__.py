"""Lyapunov spectrum of random neural networks: dynamical mean-field (single-site) theory and direct simulation.

Model (paper Eq. x)::

    x(n+1) = (1 - delta) x(n) + delta J phi(x(n)),    phi = tanh,    J_ij ~ N(0, g^2/N) iid

with time step ``delta``; ``delta -> 0`` is the continuous-time network ``dx/dt = -x + J phi(x)``.
Exponents, entropy rates and thresholds ``s`` are per unit time (per ``delta`` steps of the map), so different
time steps are directly comparable.

Modules (see PAPER_MAP.md for paper equation -> code):

  network              the network: couplings, Euler map, RK4 flow, tangent (Jacobian) steps
  lyapunov             Lyapunov spectrum of a simulated network (QR method)
  observables          F(s), Kaplan-Yorke dimension D_KY/N and entropy rate h_KS/N, from exponents or from F(s)
  dmft                 stationary dynamic mean-field theory (single-site Gaussian process) at step delta or delta -> 0
  gain_paths           gain trajectories d_0(n) = phi'(x_0(n)) sampled from the DMFT process
  single_site          single-site (cavity) equations at fixed regulator eta: solver and F(s) readout
  theory               F(s) from physical parameters: thresholds, Bloch twists, eta continuation, grid extrapolation
  max_exponent         maximum Lyapunov exponent: operator T_s, Schrodinger limit, upper edge s_*
  trivial_fixed_point  exact circular-law F(s), its single-site readout, and the semicircle limit
  participation        participation-ratio dimensions PR^x/N, PR^phi/N: two-site cavity theory and simulation
  exact                exact finite-N objects of Parts 1-2: K, the regularized solution, B_eta, R^vv
  io                   result files with provenance
"""
import torch as _torch

if _torch.cuda.is_available():
    # Keep MAGMA out: its raw cudaMalloc/cudaFree calls block on the NVIDIA driver lock (minutes-long stalls) when
    # other processes are busy on the node.  The solvers rely only on batched cuSOLVER/cuBLAS kernels.
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _torch.backends.cuda.preferred_linalg_library("cusolver")

__version__ = "1.0.0"


def default_device() -> str:
    """``"cuda:0"`` if a GPU is visible, else ``"cpu"``.  Pin a GPU with ``CUDA_VISIBLE_DEVICES=<id>``."""
    return "cuda:0" if _torch.cuda.is_available() else "cpu"


def is_continuous(delta) -> bool:
    """``delta`` is ``None`` or ``0``: the continuous-time limit ``delta -> 0``."""
    return delta is None or delta == 0
