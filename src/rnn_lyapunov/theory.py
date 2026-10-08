"""The single-site theory's ``F(s)`` from physical parameters, with every numerical choice explicit.

Pipeline (paper Sec. "Numerical solution and simulations"):

  1. stationary DMFT at ``(g, delta)``                        :mod:`rnn_lyapunov.dmft`
  2. upper edge ``s_*`` of the Lyapunov spectrum              :func:`rnn_lyapunov.max_exponent.upper_edge`
  3. gain trajectories on a periodic window of ``T``          :mod:`rnn_lyapunov.gain_paths`
  4. per threshold ``s``, Bloch twist ``theta`` and decreasing regulators ``eps``: the single-site fixed point,
     each solution warm-starting the next                     :mod:`rnn_lyapunov.single_site`
  5. readout ``F(s)`` at the smallest regulator, averaged over twists.

No simulated exponent enters; the thresholds are built from DMFT quantities only (the uncoupled exponent ``s_u``
and the upper edge).  :class:`TheoryConfig` defaults are the paper's production settings at strong coupling, except
``m`` for continuous time, which must be given (the paper's windows, grids and spacings for every point are listed in
docs/NUMERICS.md; at g >= 8 the continuous-time solution at m = 160 and 256 on T = 32 is extrapolated to zero
spacing with :func:`extrapolate_to_zero_spacing`).

Regulators are given in continuum units ``eps``; the map uses ``eta = delta * eps``, so that results at different
``delta`` are comparable at equal ``eps``.

Quick use::

    from rnn_lyapunov import theory
    cfg = theory.TheoryConfig(g=3.0, delta=0.5)
    res = theory.cdf_curve(cfg, theory.thresholds_full(cfg))       # GPU recommended; minutes
    theory.dimension_entropy(res)                                 # {'ky_dimension': ..., 'entropy_rate': ...}
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import default_device, dmft, gain_paths, is_continuous, single_site as ss
from .max_exponent import upper_edge
from .observables import dimension_entropy_from_cdf

PAPER_EPS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 4e-4, 1.6e-4)


@dataclass
class TheoryConfig:
    """A physical point ``(g, delta)`` and the numerical resolution of its single-site solution."""
    g: float
    delta: float | None              # time step; None (or 0) = continuous time (delta -> 0)
    T: float = 32.0                  # periodic window, time units
    m: int | None = None             # time points on the window: round(T/delta) for the map; required for continuous
    n_paths: int = 4096              # gain trajectories (a power of two for Sobol sampling)
    sample_m: int | None = None      # continuous only: sample trajectories on this finer grid and subsample to m
    sobol: bool = True               # scrambled-Sobol gain trajectories
    seed: int = 0                    # Sobol scramble (or pseudo-random seed)
    eps: tuple = PAPER_EPS           # decreasing regulators, continuum units (eta = delta * eps for the map)
    n_twist: int = 4                 # Bloch-phase nodes (Gauss-Legendre on [0, 1/2]); 0 = periodic window only
    twist_aT_max: float = 12.0       # twists are used only where |a_eff| T < this (artifact ~ exp(-|a_eff| T))
    warm_start_in_s: bool = True     # start each threshold from the previous one's solution at the same (eps, theta)
    tol: float = 1e-7                # fixed-point tolerance (max change of log(P, A - eta) per sweep)
    max_iter: int = 600
    anderson: int = 6
    chunk: int = 512                 # gain trajectories per GPU batch (memory only)
    dmft_dt: float = 0.025           # continuous-time DMFT lag grid
    dmft_tmax: float = 80.0          # DMFT lag range (time units); longer near the onset of chaos

    def __post_init__(self):
        if self.delta == 0:
            self.delta = None

    @property
    def continuous(self) -> bool:
        return self.delta is None

    def grid(self) -> int:
        """Number of time points ``m`` on the window."""
        if not self.continuous:
            return int(round(self.T / self.delta))
        if self.m is None:
            raise ValueError("continuous time needs m (T/m must be a multiple of dmft_dt)")
        return int(self.m)

    def spacing(self) -> float:
        """Time between grid points, ``T / m``."""
        return self.T / self.grid()


# ---- thresholds and twists ----------------------------------------------------------------------------------------

def uncoupled_exponent(delta: float | None) -> float:
    """``s_u = log(1 - delta) / delta`` (``-1`` in continuous time; ``-inf`` at ``delta = 1``): the exponent of the
    uncoupled network."""
    if is_continuous(delta):
        return -1.0
    return -np.inf if delta == 1.0 else float(np.log(1 - delta) / delta)


def a_eff(s: float, delta: float | None) -> float:
    """Distance from the uncoupled exponent in rate units: ``s + 1`` (continuous) or ``(1 - alpha_s)/delta``."""
    return s + 1.0 if is_continuous(delta) else (1.0 - (1.0 - delta) * np.exp(-s * delta)) / delta


def twist_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Bloch-phase quadrature.  ``F(theta) = F(-theta) = F(1-theta)`` (real gains), so integrate over ``[0, 1/2]``
    with ``n``-node Gauss-Legendre; weights sum to one.  ``n = 0``: the periodic window only."""
    if n == 0:
        return np.array([0.0]), np.array([1.0])
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 4, w / 2


def thresholds_full(cfg: TheoryConfig, edge: float | None = None, n_tail: int = 10) -> np.ndarray:
    """Thresholds for a whole Lyapunov spectrum (paper Fig. spectra): geometric offsets on both sides of ``s_u``,
    where ``F`` is steepest, and a uniform grid on ``[0, edge)`` for the positive exponents."""
    top = edge_of(cfg) if edge is None else edge
    if cfg.delta == 1.0:
        return np.unique(np.round(np.r_[np.linspace(-10, -1, 10), np.linspace(-0.8, 0, 5),
                                        np.linspace(0, top, n_tail + 1)[:-1]], 6))
    s_u = uncoupled_exponent(cfg.delta)
    off = np.r_[0.015, 0.03, 0.06, 0.1, 0.15, 0.2, np.arange(0.3, top - s_u + 0.35, 0.1)]
    s = np.r_[s_u - off, s_u + off[s_u + off < min(0.0, top)], np.linspace(0, top, n_tail + 1)[:-1]]
    return np.unique(np.round(s, 6))


def thresholds_upper(cfg: TheoryConfig, edge: float | None = None, n_above: int = 12,
                     spacing: float = 0.05) -> np.ndarray:
    """Thresholds for D_KY and h_KS at strong coupling (paper Fig. dimension_entropy, g >= 8 or 12): from just
    above ``s_u`` (offsets 0.015 ... 0.2, then every ``spacing``) up to 0, and ``n_above`` on ``[0, edge)``."""
    if cfg.delta == 1.0:
        raise ValueError("delta = 1 has no uncoupled exponent (s_u = -inf); use thresholds_full or thresholds_top")
    top = edge_of(cfg) if edge is None else edge
    s_u = uncoupled_exponent(cfg.delta)
    off = np.r_[0.015, 0.03, 0.06, 0.1, 0.15, 0.2, np.arange(0.25, -s_u, spacing)]
    return np.unique(np.round(np.r_[s_u + off[s_u + off < 0], np.linspace(0, max(top, 1e-3), n_above + 1)[:-1]], 6))


def thresholds_top(cfg: TheoryConfig, s_low: float, edge: float | None = None, n_below: int = 12,
                   n_above: int = 12) -> np.ndarray:
    """``n_below`` uniform thresholds on ``[s_low, 0)`` and ``n_above`` on ``[0, edge)`` (paper Fig.
    dimension_entropy for g <= 10; near onset ``s_low = -0.1`` with 16 + 12 thresholds)."""
    top = edge_of(cfg) if edge is None else edge
    return np.r_[np.linspace(s_low, 0, n_below + 1)[:-1], np.linspace(0, max(top, 1e-3), n_above + 1)[:-1]]


# ---- the computation ----------------------------------------------------------------------------------------------

def dmft_solution(cfg: TheoryConfig, device: str | None = None) -> dmft.DMFTSolution:
    """The (disk-cached) stationary DMFT of ``cfg``."""
    ga = dmft.GaussianAverages(device=device or default_device())
    return dmft.cached(cfg.g, cfg.delta, t_max=cfg.dmft_tmax, dt=cfg.dmft_dt, ga=ga)


def edge_of(cfg: TheoryConfig, device: str | None = None) -> float:
    """Upper edge ``s_*`` of the Lyapunov spectrum from the DMFT of ``cfg``."""
    return upper_edge(dmft_solution(cfg, device))


def prepare(cfg: TheoryConfig, device: str | None = None):
    """DMFT solution, upper edge, and sampled gain trajectories for ``cfg``."""
    device = device or default_device()
    sol = dmft_solution(cfg, device)
    top = upper_edge(sol)
    m = cfg.grid()
    if cfg.continuous:
        stride = cfg.T / m / cfg.dmft_dt
        assert abs(stride - round(stride)) < 1e-9, "T/m must be a multiple of dmft_dt"
        ms = cfg.sample_m or m
        assert ms % m == 0 and int(round(stride)) % (ms // m) == 0
        gp = gain_paths.sample_gain_paths(sol, ms, cfg.n_paths, cfg.seed, stride=int(round(stride)) // (ms // m),
                                          sobol=cfg.sobol, device=device)
        if ms != m:
            gp.d, gp.x = gp.d[:, ::ms // m].contiguous(), gp.x[:, ::ms // m].contiguous()
            gp.info.update(subsampled_to=m)
    else:
        gp = gain_paths.sample_gain_paths(sol, m, cfg.n_paths, cfg.seed, sobol=cfg.sobol, device=device)
    return sol, top, gp


def cdf_curve(cfg: TheoryConfig, s_values, device: str | None = None, verbose: bool = True,
              gain_trajectories: torch.Tensor | None = None, checkpoint=None) -> dict:
    """``F(s)`` at each threshold, for every regulator and twist.

    Returns a dict with ``s``, ``eps``, ``F[s, eps]`` (twist-averaged; ``F[:, -1]`` is the result at the smallest
    regulator), ``F_theta[s, eps, theta]``, ``edge``, convergence diagnostics (``iterations``, ``residual``,
    ``all_converged``), the DMFT and gain-sample summaries, ``config`` and ``seconds``.

    ``gain_trajectories`` (``paths x m``) replaces the DMFT-sampled gains (a diagnostic only).
    ``checkpoint(result)`` is called after every threshold with the results so far (NaN where not yet done).
    """
    device = device or default_device()
    start = time.monotonic()
    sol, top, gp = prepare(cfg, device)
    if gain_trajectories is not None:
        assert gain_trajectories.shape[1] == cfg.grid()
        gp = gain_paths.GainPaths(d=gain_trajectories.to(device), x=gp.x, info=dict(source="external gain paths"))
    m = cfg.grid()
    thetas, weights = twist_rule(cfg.n_twist)
    s_values = np.asarray(s_values, float)
    F_theta = np.full((len(s_values), len(cfg.eps), len(thetas)), np.nan)
    iters = np.zeros_like(F_theta)
    resid = np.zeros_like(F_theta)
    trace = np.zeros_like(F_theta)
    twisted = np.ones(len(s_values), bool) if cfg.n_twist > 0 else np.zeros(len(s_values), bool)
    parts = np.full(F_theta.shape + (3,), np.nan)
    warm: dict = {}

    def pack():
        done = ~np.isnan(F_theta[:, -1, 0])
        return dict(config=asdict(cfg), s=s_values, eps=np.array(cfg.eps), thetas=thetas, theta_weights=weights,
                    F=F_theta @ weights, F_theta=F_theta, twisted=twisted,
                    readout_parts=np.einsum("sekp,k->sep", parts, weights),
                    iterations=iters, residual=resid, trace_identity_error=trace,
                    edge=top, dmft=dict(q=sol.q, mean_sq_gain=float(sol.c_d[0]), residual=sol.residual,
                                        params=sol.params),
                    gain_sample=gp.info, all_converged=bool((resid[done] < cfg.tol).all()),
                    n_thresholds_done=int(done.sum()), seconds=time.monotonic() - start)

    for i, s in enumerate(s_values):
        use_twist = cfg.n_twist > 0 and abs(a_eff(s, cfg.delta)) * cfg.T < cfg.twist_aT_max
        for k, th in enumerate(thetas if use_twist else [0.0]):
            if cfg.continuous:
                make = lambda e: ss.continuous_problem(m, cfg.T, s, cfg.g, e, th)
            else:
                make = lambda e: ss.discrete_problem(m, s, cfg.delta, cfg.g, cfg.delta * e, th)
            if cfg.warm_start_in_s and (k, 0) in warm:
                sols = [ss.solve(gp.d, make(e), init=warm[(k, j)], tol=cfg.tol, max_iter=cfg.max_iter,
                                 anderson=cfg.anderson, chunk=cfg.chunk) for j, e in enumerate(cfg.eps)]
            else:
                sols = ss.continue_in_eta(gp.d, make, cfg.eps, tol=cfg.tol, max_iter=cfg.max_iter,
                                          anderson=cfg.anderson, chunk=cfg.chunk)
            for j, x in enumerate(sols):
                warm[(k, j)] = (x.P, np.maximum(x.A, make(cfg.eps[j]).eta * (1 + 1e-12)))
            F_theta[i, :, k] = [x.cdf for x in sols]
            iters[i, :, k] = [x.iterations for x in sols]
            resid[i, :, k] = [x.residual for x in sols]
            trace[i, :, k] = [x.trace_identity_error for x in sols]
            if not cfg.continuous:
                for j, (x, e) in enumerate(zip(sols, cfg.eps)):
                    pp = ss.discrete_readout_parts(make(e), x.A, x.V)
                    parts[i, j, k] = [pp["leak"], pp["contact"], pp["regulator"]]
        if not use_twist:                                     # theta = 0 stands for every node
            for arr in (F_theta, iters, resid, trace, parts):
                arr[i, :, 1:] = arr[i, :, :1]
            twisted[i] = False
        if checkpoint is not None:
            checkpoint(pack())
        if verbose:
            Fs = F_theta[i] @ weights
            print(f"  s={s:+.4f}  F(eps)=" + " ".join(f"{x:.5f}" for x in Fs)
                  + f"  it<= {int(iters[i].max())} res<= {resid[i].max():.1e}  [{time.monotonic() - start:.0f}s]",
                  flush=True)
    return pack()


# ---- readouts -----------------------------------------------------------------------------------------------------

def dimension_entropy(result: dict, eps_index: int = -1) -> dict:
    """``D_KY/N``, ``h_KS/N`` and ``s_KY`` of a :func:`cdf_curve` result, at regulator ``eps[eps_index]``."""
    return dimension_entropy_from_cdf(np.asarray(result["s"]), np.asarray(result["F"])[:, eps_index],
                                      result["edge"])


def extrapolate_to_zero_spacing(value_coarse: float, spacing_coarse: float, value_fine: float,
                                spacing_fine: float) -> float:
    """Quadratic extrapolation in the grid spacing to zero spacing (continuous-time theory at strong coupling).

    The error of the continuous-time solution on a time grid is quadratic in the spacing ``T/m`` (paper: ``m = 160``
    and ``256`` on ``T = 32``, i.e. spacings 0.2 and 0.125), so
    with spacings ``a = spacing_coarse``, ``b = spacing_fine``: ``v(0) = v_fine - (v_coarse - v_fine) b^2 / (a^2 - b^2)``.
    """
    w = spacing_fine**2 / (spacing_coarse**2 - spacing_fine**2)
    return value_fine - (value_coarse - value_fine) * w
