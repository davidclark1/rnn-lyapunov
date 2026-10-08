"""Stationary dynamic mean-field theory (DMFT) of the network at time step ``delta`` and in continuous time.

Single-site process (paper [eq:x0lim], [eq:dmft])::

    x_0(n+1) = (1-delta) x_0(n) + delta xi_0(n),     <xi_0(n) xi_0(n')> = g^2 <phi(x_0(n)) phi(x_0(n'))>

With ``Delta(tau) = <x_0(n) x_0(n+tau)>`` and ``b = 1-delta``, multiplying the recursion at two times gives the
lag-space equation (discrete analogue of ``Delta - Delta'' = g^2 C_phi``)::

    (1+b^2) Delta(tau) - b [Delta(tau+1) + Delta(tau-1)] = delta^2 g^2 C_phi(Delta(tau); Delta(0))

Both this and the continuous-time equation (second-order central differences on a lag grid ``dt``) are instances
of ``c0 Delta(tau) - c1 [Delta(tau+1)+Delta(tau-1)] = kappa C_phi``::

    time step delta :  c0 = 1+b^2,       c1 = b,        kappa = delta^2 g^2
    continuous time :  c0 = 1+2/dt^2,    c1 = 1/dt^2,   kappa = g^2       (O(dt^2) discretization)

Boundary conditions: ``Delta(-1) = Delta(1)`` (stationarity) and ``Delta(n_lags) = 0`` (decaying, chaotic
solution).  The lag equation alone also admits non-physical solutions with a negative power spectrum, so the solve
starts from a positivity-preserving Fourier iteration, polishes with Newton, and rejects any result that is not a
valid covariance.  Newton uses the exact Jacobian (Price's theorem): ``dC/dc = E[phi'(x) phi'(y)] = C^d(tau)`` and
``dC/dq|_c = E[phi''(x) phi(y)]``.

Gaussian averages use the trapezoid rule on a uniform grid in the standard-normal variable, which converges
exponentially for integrands analytic in a strip (tanh has poles at distance ``pi / (2 sqrt q)``; Gauss-Hermite is
*not* accurate enough at large ``q``).

Main entry point: :func:`solve` (or :func:`cached`, which stores solutions on disk).  The result,
:class:`DMFTSolution`, carries ``C^x(tau) = Delta(tau)`` (``c_x``), ``C^phi(tau)`` (``c_phi``) and the gain autocorrelation
``C^d(tau) = <d_0(n) d_0(n')>`` (``c_d``) on the lag grid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq

_DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


class GaussianAverages:
    """``E[f(x)]`` and ``E[f(x) k(y)]`` for ``(x, y)`` Gaussian with variance ``q`` and covariance ``c``."""

    def __init__(self, zmax: float = 9.0, dz: float = 0.04, device: str = _DEV):
        n = int(round(2 * zmax / dz)) + 1
        self.z = torch.linspace(-zmax, zmax, n, dtype=torch.float64, device=device)
        w = torch.exp(-0.5 * self.z**2)
        self.w = w / w.sum()
        self.params = dict(zmax=zmax, dz=dz, rule="trapezoid")

    def one(self, f, q: float) -> float:
        return float((self.w * f(np.sqrt(q) * self.z)).sum())

    def pair(self, f, k, q: float, c: np.ndarray, chunk: int = 32) -> np.ndarray:
        r = torch.as_tensor(np.clip(np.asarray(c, float) / q, -1.0, 1.0), device=self.z.device)
        sq = float(np.sqrt(q))
        fx = self.w * f(sq * self.z)                                    # (nz,)
        out = []
        for rr in torch.split(r, chunk):
            y = sq * (rr[:, None, None] * self.z[None, :, None]
                      + torch.sqrt(1 - rr**2)[:, None, None] * self.z[None, None, :])
            inner = (k(y) * self.w[None, None, :]).sum(-1)              # E_z2 k(y | z1)
            out.append((inner * fx[None, :]).sum(-1))
        return torch.cat(out).cpu().numpy()


def _phi(x):
    return torch.tanh(x)


def _dphi(x):
    return 1.0 / torch.cosh(x) ** 2


def _ddphi(x):
    t = torch.tanh(x)
    return -2.0 * t * (1.0 - t * t)


@dataclass
class DMFTSolution:
    g: float
    time_step: float | None          # the network's time step delta; None for continuous time (delta -> 0)
    dt: float                        # lag spacing in time units (= delta for the map)
    c_x: np.ndarray                  # C^x(tau) = Delta(tau) = <x_0 x_0>(tau), tau = 0..n_lags-1 (zero beyond)
    c_phi: np.ndarray                # C^phi(tau) = <phi(x_0) phi(x_0)>(tau)
    c_d: np.ndarray                  # C^d(tau) = <d_0 d_0>(tau), d_0 = phi'(x_0): gain autocorrelation
    residual: float
    iterations: int
    params: dict = field(default_factory=dict)

    @property
    def continuous(self) -> bool:
        return self.time_step is None

    @property
    def q(self) -> float:
        return float(self.c_x[0])

    @property
    def lags(self) -> np.ndarray:
        return np.arange(len(self.c_x)) * self.dt


def stationary_variance_unit_step(g: float, ga: GaussianAverages | None = None) -> float:
    """``q = g^2 E tanh^2(sqrt(q) z)``: the variance at ``delta = 1``, where the activity is white."""
    ga = ga or GaussianAverages()
    return brentq(lambda q: q - g * g * ga.one(lambda x: _phi(x) ** 2, q), 1e-8, g * g)


def continuous_variance_energy_condition(g: float, ga: GaussianAverages | None = None) -> float:
    """Continuous-time variance from energy conservation of the Sompolinsky-Crisanti-Sommers 'particle':
    ``q^2/2 = g^2 [E Phi(x)^2 - (E Phi(x))^2]`` with ``Phi = log cosh``.  Independent of the lag solver."""
    ga = ga or GaussianAverages()
    Phi = lambda x: torch.log(torch.cosh(x))

    def f(q):
        return 0.5 * q * q - g * g * (ga.one(lambda x: Phi(x) ** 2, q) - ga.one(Phi, q) ** 2)
    return brentq(f, 1e-6, g * g)


def _psd_iteration(c0, c1, kappa, n_lags, q0, tau_c_steps, ga, n_iter, mix=0.5, tol=1e-12):
    """Positivity-preserving fixed-point iteration on a ring of ``2 n_lags`` points:
    ``Delta_hat <- kappa C_hat[Delta] / (c0 - 2 c1 cos w)``.  ``Delta -> C_phi(Delta)`` maps covariances to
    covariances (Schur product theorem applied to the Hermite expansion) and the kernel is positive, so the
    iterates stay inside the cone of valid covariances.  Linear convergence: used as Newton's starting point."""
    M = 2 * n_lags
    ell = c0 - 2 * c1 * np.cos(2 * np.pi * np.fft.fftfreq(M))
    lag = np.minimum(np.arange(M), M - np.arange(M))
    cx = q0 * np.exp(-(lag / tau_c_steps) ** 2 / 2)
    for _ in range(n_iter):
        half = cx[:n_lags + 1]
        C = ga.pair(_phi, _phi, half[0], half)
        new = np.fft.ifft(kappa * np.fft.fft(np.r_[C, C[-2:0:-1]]).real / ell).real
        change = np.max(np.abs(new - cx)) / new[0]
        cx = (1 - mix) * cx + mix * new
        if change < tol:
            break
    return cx[:n_lags]


def min_relative_spectrum(cx: np.ndarray) -> float:
    """min / max of the power spectrum of the two-sided covariance (must be >= 0 up to truncation ripple)."""
    spec = np.fft.fft(np.r_[cx, 0.0, cx[:0:-1]]).real
    return float(spec.min() / spec.max())


def _solve_lag_equation(c0, c1, kappa, n_lags, q0, tau_c_steps, ga, tol, max_iter, psd_iter=150):
    cx = _psd_iteration(c0, c1, kappa, n_lags, q0, tau_c_steps, ga, psd_iter)     # valid-covariance start
    res = np.inf
    for it in range(1, max_iter + 1):
        q = cx[0]
        C = ga.pair(_phi, _phi, q, cx)
        Cd = ga.pair(_dphi, _dphi, q, cx)
        dCdq = ga.pair(_ddphi, _phi, q, cx)
        up = np.r_[cx[1:], 0.0]
        dn = np.r_[cx[1], cx[:-1]]
        F = c0 * cx - c1 * (up + dn) - kappa * C
        res = float(np.max(np.abs(F)) / (kappa * C[0]))
        if res < tol:
            break
        Jm = np.diag(c0 - kappa * Cd) - c1 * (np.eye(n_lags, k=1) + np.eye(n_lags, k=-1))
        Jm[0, 1] -= c1                                                      # Delta(-1) = Delta(1)
        Jm[:, 0] -= kappa * dCdq                                            # q = Delta(0) enters every row
        step = np.linalg.solve(Jm, -F)
        lam = 1.0
        while lam > 1e-3 and (cx[0] + lam * step[0] <= 0 or np.max(np.abs(cx + lam * step)) > 1.0001 * (cx[0] + lam * step[0])):
            lam *= 0.5                                                      # keep |Delta(tau)| <= Delta(0) > 0
        cx = cx + lam * step
    # The lag equation also has solutions that are NOT covariances (negative power spectrum); Newton found
    # one at time step delta = 0.9 from a smooth guess.  Reject them.
    if not res < max(tol, 1e-9):
        raise RuntimeError(f"DMFT Newton iteration did not converge (residual {res:.2e})")
    msr = min_relative_spectrum(cx)
    if msr < -1e-6:
        raise RuntimeError(f"DMFT lag solution is not a valid covariance (min/max spectrum {msr:.2e})")
    C = ga.pair(_phi, _phi, cx[0], cx)
    Cd = ga.pair(_dphi, _dphi, cx[0], cx)
    return cx, C, Cd, res, it


def solve_discrete(g: float, delta: float, t_max: float = 80.0, tol: float = 1e-11, max_iter: int = 60,
                   ga: GaussianAverages | None = None) -> DMFTSolution:
    """Stationary chaotic DMFT (``g > 1``) of the Euler map at time step ``delta``, on lags ``0..t_max``."""
    ga = ga or GaussianAverages()
    if delta == 1.0:
        # lag-by-lag fixed point Delta = g^2 C_phi(Delta); Delta(tau != 0) = 0 is the attracting root.
        q = stationary_variance_unit_step(g, ga)
        n = 4
        Dl = np.r_[q, np.zeros(n - 1)]
        return DMFTSolution(g, delta, 1.0, Dl, ga.pair(_phi, _phi, q, Dl), ga.pair(_dphi, _dphi, q, Dl),
                            0.0, 0, dict(method="unit-step scalar", quadrature=ga.params))
    b = 1.0 - delta
    n_lags = int(round(t_max / delta))
    q0 = continuous_variance_energy_condition(g, ga)
    Dl, C, Cd, res, it = _solve_lag_equation(1 + b * b, b, (delta * g) ** 2, n_lags, q0, 2.0 / delta, ga, tol,
                                             max_iter)
    return DMFTSolution(g, delta, delta, Dl, C, Cd, res, it,
                        dict(method="newton", t_max=t_max, tol=tol, quadrature=ga.params))


def solve_continuous(g: float, dt: float = 0.025, t_max: float = 80.0, tol: float = 1e-11, max_iter: int = 60,
                     ga: GaussianAverages | None = None) -> DMFTSolution:
    """Stationary chaotic DMFT in continuous time (``delta -> 0``) on a lag grid ``dt`` (central differences)."""
    ga = ga or GaussianAverages()
    n_lags = int(round(t_max / dt))
    q0 = continuous_variance_energy_condition(g, ga)
    Dl, C, Cd, res, it = _solve_lag_equation(1 + 2 / dt**2, 1 / dt**2, g * g, n_lags, q0, 2.0 / dt, ga, tol,
                                             max_iter)
    return DMFTSolution(g, None, dt, Dl, C, Cd, res, it, dict(method="newton", t_max=t_max, tol=tol,
                                                               quadrature=ga.params))


def solve(g: float, delta: float | None, *, t_max: float = 80.0, dt: float = 0.025, tol: float = 1e-11,
          max_iter: int = 60, ga: GaussianAverages | None = None) -> DMFTSolution:
    """Stationary DMFT at time step ``delta`` (``None`` or ``0``: continuous time on the lag grid ``dt``)."""
    if delta is None or delta == 0:
        return solve_continuous(g, dt=dt, t_max=t_max, tol=tol, max_iter=max_iter, ga=ga)
    return solve_discrete(g, delta, t_max=t_max, tol=tol, max_iter=max_iter, ga=ga)


def cache_dir() -> Path:
    """Directory of :func:`cached`: ``$RNN_LYAPUNOV_CACHE`` or ``~/.cache/rnn_lyapunov``, subdirectory ``dmft``."""
    return Path(os.environ.get("RNN_LYAPUNOV_CACHE", Path.home() / ".cache" / "rnn_lyapunov")) / "dmft"


def cached(g: float, delta: float | None, *, t_max: float = 80.0, dt: float = 0.025, ga: GaussianAverages | None = None,
           directory: str | Path | None = None, **kw) -> DMFTSolution:
    """:func:`solve`, stored on disk.  The solution depends only on its arguments and the quadrature rule, which
    form the cache key (for continuous time, ``dt`` is part of the key; for the map, it is not used)."""
    import hashlib
    import json
    import pickle
    ga = ga or GaussianAverages()
    cont = delta is None or delta == 0
    directory = Path(directory) if directory else cache_dir()
    key = json.dumps(dict(g=float(g), delta=None if cont else float(delta), t_max=float(t_max),
                          dt=float(dt) if cont else None, kw=kw, quad=ga.params, v=1), sort_keys=True)
    f = directory / (hashlib.sha256(key.encode()).hexdigest()[:20] + ".pkl")
    if f.exists():
        return pickle.loads(f.read_bytes())
    sol = solve(g, delta, t_max=t_max, dt=dt, ga=ga, **kw)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(pickle.dumps(sol))
    tmp.replace(f)                      # atomic: concurrent workers never read a partial file
    return sol


def mean_gain(sol: DMFTSolution, ga: GaussianAverages | None = None) -> float:
    """``<phi'(x_0)>`` with ``x_0 ~ N(0, q)``."""
    ga = ga or GaussianAverages()
    return ga.one(_dphi, sol.q)
