"""Gain trajectories ``d_0(n) = phi'(x_0(n))`` of the single-site process, on a periodic window of ``m`` times.

``x_0`` is the stationary Gaussian DMFT process (paper [eq:dmft]).  On a ring of ``m`` points spaced ``stride``
DMFT lags apart, its covariance is the periodized DMFT covariance ``sum_b C^x(|n + b m|)``; its FFT is the
(non-negative) ring spectrum.  Negative FFT values, which can arise only from truncation or periodization error,
are reported (``info["min_rel_ring_spectrum"]``) and clipped.

Sampling: plain pseudo-random normals (torch generator) or scrambled-Sobol normals used as Fourier coordinates (a
quadrature choice that lowers the path-sampling variance of averages over gain trajectories; it involves no
fitting).  The paper uses Sobol paths throughout; ``seed`` is then the Sobol scramble.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dmft import DMFTSolution


@dataclass
class GainPaths:
    d: torch.Tensor          # (paths, m) gains d_0(n) = phi'(x_0(n)) = sech^2 x_0(n)
    x: torch.Tensor          # (paths, m) Gaussian activity paths
    info: dict


def ring_covariance(sol: DMFTSolution, m: int, stride: int = 1) -> np.ndarray:
    """Periodized covariance on a ring of ``m`` points spaced ``stride`` DMFT lags apart."""
    L = len(sol.c_x)
    cov = np.zeros(m)
    n = np.arange(m)
    reps = L // (m * stride) + 2
    for b in range(-reps, reps + 1):
        lag = np.abs(n + b * m) * stride
        ok = lag < L
        cov[ok] += sol.c_x[lag[ok]]
    return cov


def sample_gain_paths(sol: DMFTSolution, m: int, n_paths: int, seed: int = 0, *, stride: int = 1,
                      sobol: bool = False, device: str | None = None) -> GainPaths:
    """``n_paths`` stationary gain trajectories on a ring of ``m`` points spaced ``stride`` DMFT lags apart.

    For Sobol sampling, ``n_paths`` must be a power of two and ``seed`` is the scramble.
    """
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cov = ring_covariance(sol, m, stride)
    spec = np.fft.fft(cov).real
    neg = float(spec.min() / spec.max())
    spec = np.clip(spec, 0.0, None)
    amp = torch.as_tensor(np.sqrt(spec), dtype=torch.float64, device=device)
    if sobol:
        # Scrambled-Sobol normals used as *Fourier* coordinates, lowest frequencies first: the best-distributed
        # Sobol dimensions then carry the modes with the most variance.  Real-FFT synthesis:
        # c_0 = a_0 u_0, c_k = a_k (u_{2k-1} + i u_{2k})/sqrt 2, c_{m/2} = a_{m/2} u_{m-1};  x = irfft(c, norm="ortho").
        from scipy.special import ndtri
        from scipy.stats import qmc
        assert n_paths & (n_paths - 1) == 0, "Sobol path count must be a power of two"
        u = qmc.Sobol(d=m, scramble=True, seed=seed).random_base2(int(np.log2(n_paths)))
        u = torch.as_tensor(ndtri(np.clip(u, 1e-16, 1 - 1e-16)), dtype=torch.float64, device=device)
        nf = m // 2 + 1
        c = torch.zeros(n_paths, nf, dtype=torch.complex128, device=device)
        c[:, 0] = u[:, 0]
        last = nf - 1 if m % 2 == 0 else nf
        kk = torch.arange(1, last, device=device)
        c[:, 1:last] = (u[:, 2 * kk - 1] + 1j * u[:, 2 * kk]) / np.sqrt(2.0)
        if m % 2 == 0:
            c[:, -1] = u[:, m - 1]
        x = torch.fft.irfft(c * amp[:nf][None, :], n=m, dim=1, norm="ortho")
    else:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(n_paths, m, generator=gen, dtype=torch.float64).to(device)
        # filter white noise with sqrt(spectrum); real even spectrum -> real output
        x = torch.fft.ifft(torch.fft.fft(z, dim=1) * amp[None, :], dim=1).real
    d = 1.0 / torch.cosh(x) ** 2
    info = dict(m=m, n_paths=n_paths, seed=seed, stride=stride, sobol=sobol, min_rel_ring_spectrum=neg,
                sample_variance=float((x * x).mean()), target_variance=sol.q,
                sample_mean_gain=float(d.mean()), sample_mean_sq_gain=float((d * d).mean()), target_mean_sq_gain=float(sol.c_d[0]))
    return GainPaths(d=d, x=x, info=info)
