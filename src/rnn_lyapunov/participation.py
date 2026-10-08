"""Participation-ratio dimensions ``PR^x/N`` and ``PR^phi/N``: two-site cavity theory and simulation estimators.

Definition (paper [eq:PRdef] and Sec. [sec:dimension]), for ``a in {x, phi}`` with ``C^a_ij`` the equal-time covariance::

    PR^a = (sum_i C^a_ii)^2 / sum_ij (C^a_ij)^2,        PR^a / N -> C^a(0)^2 / (C^a(0)^2 + psi^a(0,0))

THEORY (paper Appendix "Participation-ratio dimension at finite time step"; Clark, Abbott & Litwin-Kumar, PRL 2023,
Eqs. 3, 4, 16), with the plain transform ``f(theta) = sum_n f(n) e^{-i theta n}`` on the Brillouin zone::

    psi^phi(t1,t2) = (M - 1) C^phi(t1) C^phi(t2),          M = 1 / |1 - g^2 S^phi(t1) S^phi(t2)|^2     [eq:psiphi]
    psi^x(t1,t2)   = |U|^2 C^phi(t1) C^phi(t2) + 2 Re[U] C^{x phi}(t1) C^{x phi}(t2)                    [eq:psix]
                     U = g^2 S^x(t1) S^x(t2) / (1 - g^2 S^phi(t1) S^phi(t2)),   C^{x phi} = <phi'> C^x
    S^x(theta) = delta / (e^{i theta} - (1 - delta)),   S^phi = <phi'> S^x
    psi^a(0,0) = int_{-pi}^{pi} dt1 dt2 / (2 pi)^2  psi^a(t1, t2)                                        [eq:psi00]

Continuous time (PRL Appendix C): ``S^x(w) = 1/(1 + i w)``, ``C(w) = int dtau C(tau) e^{-i w tau}``, measure
``dw/(2 pi)``.  At ``delta = 1`` the activity is white: ``PR^phi/N = 1 - (g <phi'>)^4``,
``PR^x/N = (1 - (g <phi'>)^4) / (2 - (g <phi'>)^4)``.

Numerics: the DMFT lag functions are zero-padded to ``min_points`` (they have decayed), so the DFT samples the
exact transform on a fine grid; the double integral is a periodic trapezoid sum, chunked over ``theta_1`` (GPU).
The grid must resolve the near-critical factor ``M`` (width ~ ``1 - (g <phi'>)^2`` near onset).

SIMULATION: :func:`covariance_blocks` accumulates second moments of ``x`` and ``phi(x)`` in disjoint time blocks;
:func:`pr_from_blocks` returns the naive estimator (biased down by estimation noise ~ ``N^2 tau_c / T``) and the
cross-block estimator ``Tr(C_a C_b), a != b`` (unbiased for independent blocks; used in the paper), with a jackknife
standard error.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import default_device
from . import network as net
from .dmft import DMFTSolution, GaussianAverages, mean_gain


# ---- theory -------------------------------------------------------------------------------------------------------

def pr_unit_step_closed_form(g: float, mean_gain_value: float) -> dict:
    """``delta = 1``: ``PR^phi/N = 1 - r``, ``PR^x/N = (1 - r)/(2 - r)``, ``r = (g <phi'>)^4``."""
    r = (g * mean_gain_value) ** 4
    return dict(pr_phi=1.0 - r, pr_x=(1.0 - r) / (2.0 - r), r=r)


def spectra(sol: DMFTSolution, min_points: int = 16384):
    """Frequency grid, measure (``dtheta/2pi`` or ``dw/2pi``), ``C^x``, ``C^phi`` and ``S^x`` on that grid (NumPy)."""
    pad = max(0, min_points // 2 - len(sol.c_x))
    cx, cp = np.r_[sol.c_x, np.zeros(pad)], np.r_[sol.c_phi, np.zeros(pad)]
    Cx = np.r_[cx, 0.0, cx[:0:-1]]                                # two-sided, even, lag 0 first
    Cp = np.r_[cp, 0.0, cp[:0:-1]]
    M = len(Cx)
    if sol.continuous:                                            # C(w) = dt * DFT, w_k = 2 pi k / (M dt)
        w = 2 * np.pi * np.fft.fftfreq(M, sol.dt)
        return w, 1.0 / (M * sol.dt), sol.dt * np.fft.fft(Cx).real, sol.dt * np.fft.fft(Cp).real, 1.0 / (1.0 + 1j * w)
    th = 2 * np.pi * np.fft.fftfreq(M)                            # Brillouin zone, dtheta / 2 pi = 1 / M
    delta = sol.time_step
    return th, 1.0 / M, np.fft.fft(Cx).real, np.fft.fft(Cp).real, delta / (np.exp(1j * th) - (1.0 - delta))


@torch.no_grad()
def pr_theory(sol: DMFTSolution, mean_gain_value: float | None = None, min_points: int = 16384, chunk: int = 1024,
              device: str | None = None) -> dict:
    """``PR^x/N`` and ``PR^phi/N`` from the DMFT two-point functions ``sol`` and ``<phi'>`` (default: from ``sol``)."""
    t0 = time.monotonic()
    dev = torch.device(device or default_device())
    mg = mean_gain(sol, GaussianAverages(device=str(dev))) if mean_gain_value is None else mean_gain_value
    g = sol.g
    _, measure, Cxw, Cpw, Sx = spectra(sol, min_points)
    Cxw = torch.as_tensor(Cxw, dtype=torch.float64, device=dev)
    Cpw = torch.as_tensor(Cpw, dtype=torch.float64, device=dev)
    Sx = torch.as_tensor(Sx, dtype=torch.complex128, device=dev)
    Sp = mg * Sx                                                  # S^phi = <phi'> S^x
    Cxp = mg * Cxw                                             # C^{x phi} = <phi'> C^x (x Gaussian)
    psi_phi = torch.zeros((), dtype=torch.float64, device=dev)
    psi_x = torch.zeros((), dtype=torch.float64, device=dev)
    for i0 in range(0, len(Sx), chunk):                           # double integral, chunked over theta_1
        sl = slice(i0, i0 + chunk)
        den = 1.0 - (g * g) * Sp[sl, None] * Sp[None, :]
        Mfac = 1.0 / den.abs() ** 2
        U = (g * g) * Sx[sl, None] * Sx[None, :] / den
        psi_phi += (Cpw[sl] * ((Mfac - 1.0) @ Cpw)).sum()
        psi_x += (Cpw[sl] * ((U.abs() ** 2) @ Cpw)).sum() + 2.0 * (Cxp[sl] * (U.real @ Cxp)).sum()
    psi_phi = float(psi_phi) * measure**2
    psi_x = float(psi_x) * measure**2
    Cx0, Cp0 = float(sol.c_x[0]), float(sol.c_phi[0])
    return dict(pr_x=Cx0**2 / (Cx0**2 + psi_x), pr_phi=Cp0**2 / (Cp0**2 + psi_phi), psi_x=psi_x, psi_phi=psi_phi,
                mean_gain=float(mg), g_mean_gain=float(g * mg), n_freq=len(Cxw),
                seconds=time.monotonic() - t0)


# ---- simulation ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def covariance_blocks(J: torch.Tensor, *, delta: float | None = None, rk4_dt: float | None = None,
                      x0: torch.Tensor | None = None, t_burn: float = 300.0, t_obs: float = 8000.0,
                      n_blocks: int = 8, sample_every: int = 1, seed: int = 0, batch: int = 256) -> dict:
    """Second-moment matrices and means of ``x`` and ``tanh x`` in ``n_blocks`` disjoint time blocks.

    ``delta``: the Euler map; ``rk4_dt``: the continuous-time network (RK4).  A state is stored every
    ``sample_every`` updates; ``n_per_block = round(t_obs / (step * sample_every) / n_blocks)``.
    """
    dyn = net.dynamics(delta=delta, rk4_dt=rk4_dt)
    dt = dyn.step_size
    N, dev = J.shape[0], J.device
    x = net.initial_state(N, 1.0, seed, dev) if x0 is None else x0.clone().to(dev)
    for _ in range(round(t_burn / dt)):
        x = dyn.step(x, J, dt)
    n_samples = round(t_obs / (dt * sample_every) / n_blocks)
    Sx = torch.zeros(n_blocks, N, N, dtype=torch.float64, device=dev)
    Sp = torch.zeros_like(Sx)
    mx = torch.zeros(n_blocks, N, dtype=torch.float64, device=dev)
    mp = torch.zeros_like(mx)
    buf = torch.empty(batch, N, dtype=torch.float64, device=dev)
    for b in range(n_blocks):
        done = 0
        while done < n_samples:
            k = min(batch, n_samples - done)
            for r in range(k):
                for _ in range(sample_every):
                    x = dyn.step(x, J, dt)
                buf[r] = x
            X = buf[:k]
            P = torch.tanh(X)
            Sx[b] += X.T @ X
            Sp[b] += P.T @ P
            mx[b] += X.sum(0)
            mp[b] += P.sum(0)
            done += k
    return dict(Sx=Sx, Sp=Sp, mx=mx, mp=mp, n_per_block=n_samples, x_final=x)


def pr_from_blocks(S: torch.Tensor, m: torch.Tensor, n: int, subtract_mean: bool = True) -> dict:
    """``PR/N`` from block second moments ``S`` (``B x N x N``) and sums ``m`` (``B x N``) of ``n`` samples each:
    naive (biased down), cross-block (unbiased for independent blocks) with jackknife s.e., and the variance."""
    B, N = S.shape[0], S.shape[1]
    mean = m.sum(0) / (B * n)
    C = S / n - (torch.outer(mean, mean)[None] if subtract_mean else 0.0)          # (B, N, N)
    tr = torch.einsum("bii->b", C)
    Cbar = C.mean(0)
    naive = float(tr.mean() ** 2 / (N * (Cbar * Cbar).sum()))
    flat = C.reshape(B, -1)
    G = flat @ flat.T                                                               # G[a, b] = Tr(C_a C_b)
    off = (G.sum() - torch.diagonal(G).sum()) / (B * (B - 1))
    cross = float(tr.mean() ** 2 / (N * off))
    jk = []
    for a in range(B):
        keep = [b for b in range(B) if b != a]
        Gk = G[keep][:, keep]
        offk = (Gk.sum() - torch.diagonal(Gk).sum()) / ((B - 1) * (B - 2))
        jk.append(float(tr[keep].mean() ** 2 / (N * offk)))
    jk = np.array(jk)
    return dict(pr_naive=naive, pr_cross=cross, pr_cross_se=float(np.sqrt((B - 1) / B * np.sum((jk - jk.mean()) ** 2))),
                variance=float(tr.mean() / N))


def pr_simulation(J: torch.Tensor, *, delta: float | None = None, rk4_dt: float | None = None, t_burn: float = 300.0,
                  t_obs: float = 20000.0, n_blocks: int = 8, sample_dt: float = 0.5, seed: int = 0) -> dict:
    """``PR^x/N`` and ``PR^phi/N`` of one simulated network (paper settings: ``t_obs = 20000``, 8 blocks, a sample
    every 0.5 time units).  Returns ``{"x": pr_from_blocks(...), "phi": pr_from_blocks(...), ...}``."""
    step = delta if delta is not None else rk4_dt
    every = max(1, round(sample_dt / step))
    blk = covariance_blocks(J, delta=delta, rk4_dt=rk4_dt, t_burn=t_burn, t_obs=t_obs, n_blocks=n_blocks,
                            sample_every=every, seed=seed)
    return dict(x=pr_from_blocks(blk["Sx"], blk["mx"], blk["n_per_block"]),
                phi=pr_from_blocks(blk["Sp"], blk["mp"], blk["n_per_block"]),
                sample_every_steps=every, samples_per_block=blk["n_per_block"])
