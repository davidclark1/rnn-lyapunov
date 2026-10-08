"""Single-site theory at fixed regulator ``eta``: the self-consistent kernels and the readout of ``F(s)``.

Paper [eq:Bsite]-[eq:Fsite].  On a periodic window of ``m`` times (approximating the infinite window), with
``D_0 = diag(d_0(n))`` for one gain trajectory and ``<.>`` the average over gain trajectories::

    R^vu_00 = [eta I + gamma_s D_0 P D_0 + L^T (eta I + A)^{-1} L]^{-1}          [eq:Rvu00]
    R^uv_00 = [eta I + A + L (eta I + gamma_s D_0 P D_0)^{-1} L^T]^{-1}          [eq:Ruv00]
    A = gamma_s <D_0 R^vu_00 D_0>,     P = <R^uv_00>                              [eq:closed]
    F(s) = lim_{eta -> 0+} <R^vv_00(k+1, k)>                                      [eq:Fsite]

Code names: ``A`` here is the paper's ``eta I + A`` (written A_cal below), ``Q_d = R^vu_00``, ``X_d = R^uv_00``
(computed in the manifestly positive Woodbury form), ``V = <R^vu_00>``.  ``alpha = alpha_s``, ``gamma = gamma_s``.

Representation
--------------
* The stationary kernels ``A_cal, P, V`` are real symbols on the frequency grid ``w_k = 2 pi (k + theta) / m``
  (FFT ordering); the average of a trajectory-dependent matrix is projected onto its stationary part,
  ``symbol_k = e_k^+ M e_k / m``.  Each gain trajectory needs dense ``m x m`` inverses (``D_0`` is not translation
  invariant); they are batched over trajectories on the GPU.
* Bloch twist ``theta``: with ``Phi = diag exp(2 pi i theta n / m)``, a twisted circulant is ``Phi C Phi^+`` with
  ``C`` the plain circulant of the symbol on the shifted grid.  ``Phi`` commutes with the diagonal gain matrices,
  so the twisted problem equals the plain one with shifted symbols (complex Hermitian matrices when the sampled
  symbols are not even in ``k``).  Averaging over ``theta`` removes the periodic-window artifact near the
  uncoupled exponent.
* Only ``|L|^2`` is needed: ``L = U |L|`` with ``U`` a unitary circulant, and unitary circulants drop out of the
  stationary projection.

One core serves both formulations (for continuous time, operators are matrices with kernel * dt)::

    time step delta :  |L|^2 = 1 + alpha^2 - 2 alpha cos w,  gamma = (delta g e^{-s delta})^2,  eta dimensionless
    continuous time :  |L|^2 = a^2 + W^2  (a = s+1, W = w m / T),  gamma = g^2,  eta = epsilon

Readout at finite regulator.  For time step ``delta``, ``F_eta = (1/m) sum_k V_k [eta + (1 - alpha cos w_k)/A_k]``:
the lag-one response ``<R^vv_00(k+1,k)>`` (symbol ``V (1 - alpha cos w)/A_cal``) plus ``eta <R^vu_00(k,k)>``, which
vanishes as ``eta -> 0`` (both terms are kept at finite ``eta``, as in all results of the paper).  For continuous
time, the readout is ``1/2 + int dW/2pi a V/A_cal`` (the ``1/2`` is the midpoint of the jump of the response
across the source).

Exposed numerical choices: ``m``, ``theta``, ``eta``, gain trajectories, ``tol``, ``mixing``, Anderson depth,
gauge fixing.  ``Tr P = Tr V`` holds at every fixed point with ``eta > 0``; imposing it after each sweep removes the
slow ``(A_cal, P, Q) -> (c A_cal, P/c, c Q)`` mode of the ``eta -> 0`` equations without moving the fixed point.
The un-gauged trace mismatch is reported (``trace_identity_error``).

Main entry points: :func:`discrete_problem` / :func:`continuous_problem` (one threshold ``s``, one ``eta``, one
twist), :func:`solve` (fixed point for given gain trajectories) and :func:`continue_in_eta` (decreasing ``eta``,
each solution warm-starting the next).  :mod:`rnn_lyapunov.theory` drives these over thresholds and twists.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

Tensor = torch.Tensor


# ---- symbols and readout weights ----------------------------------------------------------

def frequencies(m: int, theta: float = 0.0) -> np.ndarray:
    """``w_k = 2 pi (k + theta) / m`` in FFT ordering (radians per ring step)."""
    return 2 * np.pi * (np.fft.fftfreq(m, 1.0 / m) + theta) / m


@dataclass
class Problem:
    """Symbols defining one fixed-point problem and its CDF readout (all arrays over ``w_k``)."""
    ell2: np.ndarray          # |L|^2
    gamma: float
    eta: float
    theta: float
    readout_weight: np.ndarray   # r_k  in  F = offset + norm * sum_k [V_k (eta_r + r_k / A_k) - sub_k]
    readout_norm: float
    readout_eta: float
    readout_sub: np.ndarray
    readout_offset: float
    meta: dict = field(default_factory=dict)


def discrete_problem(m: int, s: float, delta: float, g: float, eta: float, theta: float = 0.0) -> Problem:
    """Problem at time step ``delta``, threshold ``s``, regulator ``eta`` on a ring of ``m`` steps.

    ``F_eta = (1/m) sum_k V_k [eta + (1 - alpha_s cos w_k) / A_k]``.  ``1 - alpha cos w = Re(e^{iw} conj L)`` gives
    the lag-one response ``<R^vv_00(k+1,k)>``; the ``eta V`` term is ``eta <R^vu_00(k,k)>``.
    """
    w = frequencies(m, theta)
    alpha, beta = (1 - delta) * math.exp(-s * delta), delta * math.exp(-s * delta)
    return Problem(ell2=1 + alpha**2 - 2 * alpha * np.cos(w), gamma=(beta * g) ** 2, eta=eta, theta=theta,
                   readout_weight=1 - alpha * np.cos(w), readout_norm=1.0 / m, readout_eta=eta,
                   readout_sub=np.zeros(m), readout_offset=0.0,
                   meta=dict(kind="discrete", m=m, s=s, delta=delta, g=g, eta=eta, theta=theta, alpha=alpha, beta=beta))


def continuous_problem(m: int, T: float, s: float, g: float, eps: float, theta: float = 0.0) -> Problem:
    """Continuous-time problem on a window ``T`` (time units) with ``m`` grid points (spectral derivative).

    ``F = 1/2 + int dW/2pi a V/A_cal``; the slowly convergent tail is handled by subtracting the
    uncoupled integrand ``a/(a^2+W^2)`` on the grid and adding back its exact sum over *all* Bloch
    frequencies of the ring, ``(1/2) sinh(aT) / (cosh(aT) - cos 2 pi theta)`` (theta-average: sign(a)/2).
    """
    a = s + 1.0
    if a == 0.0 and theta == 0.0:
        raise ValueError("s = -1 is the uncoupled exponent, where the periodic window (theta = 0) is singular; "
                         "use a threshold off s = -1 or a nonzero Bloch twist")
    W = frequencies(m, theta) * m / T
    ell2 = a * a + W * W
    bare = 0.5 * math.sinh(a * T) / (math.cosh(a * T) - math.cos(2 * math.pi * theta))
    return Problem(ell2=ell2, gamma=g * g, eta=eps, theta=theta,
                   readout_weight=np.full(m, a), readout_norm=1.0 / T, readout_eta=0.0,
                   readout_sub=a / ell2, readout_offset=0.5 + bare,
                   meta=dict(kind="continuous", m=m, T=T, s=s, a=a, g=g, eps=eps, theta=theta))


# ---- ring helpers ------------------------------------------------------------------------

class Ring:
    def __init__(self, m: int, complex_: bool, device):
        self.m, self.device = m, device
        self.dtype = torch.complex128 if complex_ else torch.float64
        n = torch.arange(m, device=device)
        self._diff = (n[:, None] - n[None, :]) % m
        self._idx = (n[None, :] + n[:, None]) % m                # [tau, n] -> n + tau
        self._n = n

    def circulant(self, symbol: Tensor) -> Tensor:
        """``C[n, n'] = (1/m) sum_k symbol_k exp(2 pi i k (n - n') / m)``."""
        col = torch.fft.ifft(symbol.to(torch.complex128))
        C = col[self._diff]
        return C if self.dtype == torch.complex128 else C.real

    def symbol_of(self, M: Tensor) -> Tensor:
        """Real symbol of the stationary projection of a Hermitian ``m x m`` matrix."""
        c = M[self._idx, self._n[None, :]].mean(1)               # c(tau) = mean_n M[n + tau, n]
        return torch.fft.fft(c.to(torch.complex128)).real


_TRSM_MAX = 512      # PyTorch sends batched triangular solves with n > 512 to MAGMA (hard-coded heuristic)


def _tri_inv(L: Tensor, max_block: int = _TRSM_MAX) -> Tensor:
    """Inverse of a batch of lower-triangular matrices using only cuBLAS-backed kernels.

    For ``n > 512`` ``torch.linalg.solve_triangular`` dispatches to MAGMA's recursive batched trsm, which calls
    raw ``cudaMalloc`` inside its recursion; every such call takes the NVIDIA driver's global lock and the
    process spends >95% of its time blocked (found with py-spy on m = 640 runs).
    Block recursion  [[A,0],[B,C]]^-1 = [[A^-1,0],[-C^-1 B A^-1, C^-1]]  keeps every solve at ``n <= max_block``.
    """
    n = L.shape[-1]
    if n <= max_block:
        eye = torch.eye(n, dtype=L.dtype, device=L.device).expand_as(L)
        return torch.linalg.solve_triangular(L, eye, upper=False)
    k = n // 2
    Ai, Ci = _tri_inv(L[..., :k, :k], max_block), _tri_inv(L[..., k:, k:], max_block)
    out = torch.zeros_like(L)
    out[..., :k, :k], out[..., k:, k:] = Ai, Ci
    out[..., k:, :k] = -Ci @ L[..., k:, :k] @ Ai
    return out


def _pd_inv(M: Tensor) -> Tensor:
    """Inverse of a Hermitian positive-definite batch: ``M = L L^+``, ``M^{-1} = L^{-+} L^{-1}``.

    Raises if ``M`` is not positive definite, so every solve also verifies positivity of ``C_d``,
    ``Q_d^{-1}`` and ``X_d^{-1}``.  Only truly batched cuSOLVER/cuBLAS kernels are used (batched Cholesky,
    batched triangular solves of size <= 512, bmm): ``cholesky_inverse``/``inv`` loop per matrix or go through
    MAGMA, whose raw cudaMalloc calls stall on the driver lock.
    """
    Linv = _tri_inv(torch.linalg.cholesky(0.5 * (M + M.conj().transpose(-1, -2))))
    return Linv.conj().transpose(-1, -2) @ Linv


def sweep(d: Tensor, ell2: Tensor, gamma: float, eta: float, P: Tensor, A: Tensor, ring: Ring,
          chunk: int = 512) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate the right-hand sides once: symbols ``(A_new, P_new, V_new)`` from ``(P, A)``."""
    m = d.shape[1]
    eye = torch.eye(m, dtype=ring.dtype, device=d.device)
    base = ring.circulant(ell2 / A)                              # L^+ A^{-1} L
    Pm, Am, absL = ring.circulant(P), ring.circulant(A), ring.circulant(torch.sqrt(ell2))
    SA = torch.zeros(m, m, dtype=ring.dtype, device=d.device)
    SP, SV = torch.zeros_like(SA), torch.zeros_like(SA)
    for dc in torch.split(d, chunk):
        dd = dc.to(ring.dtype)
        C = eta * eye + gamma * dd[:, :, None] * Pm[None] * dd[:, None, :]           # C_d
        Q = _pd_inv(C + base[None])                                                 # Q_d = R^{vu}[d]
        SV += Q.sum(0)
        SA += (dd[:, :, None] * Q * dd[:, None, :]).sum(0)
        Cinv = _pd_inv(C)
        SP += _pd_inv(Am[None] + absL[None] @ Cinv @ absL[None]).sum(0)   # X_d = R^{uv}[d]
    n = d.shape[0]
    return eta + gamma * ring.symbol_of(SA / n), ring.symbol_of(SP / n), ring.symbol_of(SV / n)


# ---- fixed point -------------------------------------------------------------------------

@dataclass
class OperatorSolution:
    A: np.ndarray            # symbol of A_cal
    P: np.ndarray
    V: np.ndarray
    cdf: float
    residual: float          # max |Delta log(P, A_cal - eta)| of the last sweep
    iterations: int
    converged: bool
    trace_identity_error: float      # |Tr P - Tr V| / Tr V before gauge fixing, last sweep
    seconds: float
    meta: dict = field(default_factory=dict)


def readout(prob: Problem, A: np.ndarray, V: np.ndarray) -> float:
    """``F_eta`` (paper [eq:Fsite] at finite eta) from the symbols of ``A_cal`` and ``V = <R^vu_00>``:
    ``offset + norm * sum_k [V_k (eta_r + w_k / A_k) - sub_k]`` with the weights of :func:`discrete_problem` /
    :func:`continuous_problem` (see the module docstring)."""
    terms = V * (prob.readout_eta + prob.readout_weight / A) - prob.readout_sub
    return float(prob.readout_offset + prob.readout_norm * terms.sum())


def discrete_readout_parts(prob: Problem, A: np.ndarray, V: np.ndarray) -> dict:
    """Split ``F_eta`` of a discrete problem using ``1 - alpha cos w = (1 - alpha) + alpha (1 - cos w)``:

    leak      (1/m) sum V (1-alpha)/A        -> int dW/2pi a V/A   (continuous integrand) as delta -> 0
    contact   (1/m) sum V alpha (1-cos w)/A  -> 1/2                (order-one frequencies)
    regulator eta (1/m) sum V                -> 0
    """
    alpha = prob.meta["alpha"]
    w = frequencies(len(A), prob.theta)
    return dict(leak=float(np.mean(V * (1 - alpha) / A)), contact=float(np.mean(V * alpha * (1 - np.cos(w)) / A)),
                regulator=float(prob.eta * np.mean(V)))


def uncoupled_kernels(prob: Problem) -> tuple[np.ndarray, np.ndarray]:
    """``gamma = 0`` solution ``A_cal = eta``, ``P = V = eta / (eta^2 + |L|^2)``: the large-regulator start."""
    return prob.eta / (prob.eta**2 + prob.ell2), np.full_like(prob.ell2, prob.eta)


def solve(d: Tensor, prob: Problem, *, init: tuple[np.ndarray, np.ndarray] | None = None, tol: float = 1e-8,
          max_iter: int = 1000, mixing: float = 1.0, anderson: int = 6, gauge_fix: bool = True,
          chunk: int = 512, verbose: bool = False) -> OperatorSolution:
    """Fixed point in the unknown symbols ``(P, A_cal)``.

    Iterates on ``x = log(P, A_cal - eta)`` (both positive on the branch inherited from the positive
    resolvent) with Anderson acceleration of depth ``anderson`` (0 = damped iteration, factor
    ``mixing``).  ``init=None`` starts from the uncoupled kernels plus one linear response of ``A``;
    regulator continuation is done by the caller through ``init``.
    """
    start = time.monotonic()
    dev, m = d.device, d.shape[1]
    eta, gamma = prob.eta, prob.gamma
    ell2 = torch.as_tensor(prob.ell2, dtype=torch.float64, device=dev)
    sym_even = np.allclose(prob.ell2, np.roll(prob.ell2[::-1], 1), rtol=1e-13, atol=0)
    ring = Ring(m, complex_=not sym_even, device=dev)
    if init is None:
        P0, _ = uncoupled_kernels(prob)
        P = torch.as_tensor(P0, dtype=torch.float64, device=dev)
        A = eta + gamma * float((d * d).mean()) * P
    else:
        P = torch.as_tensor(init[0], dtype=torch.float64, device=dev).clone()
        A = torch.as_tensor(init[1], dtype=torch.float64, device=dev).clone()
    floor = 1e-300
    xs: list[Tensor] = []
    fs: list[Tensor] = []
    res, it, tr_err, V, best = float("inf"), 0, float("nan"), P, float("inf")
    for it in range(1, max_iter + 1):
        A_new, P_new, V_new = sweep(d, ell2, gamma, eta, P, A, ring, chunk)
        tr_err = float(abs(P_new.sum() - V_new.sum()) / V_new.sum())
        if gauge_fix and gamma > 0:
            c = torch.sqrt(V_new.sum() / P_new.sum())            # impose Tr P = Tr V
            P_new, A_new = P_new * c, eta + (A_new - eta) / c
        x = torch.log(torch.cat([P, A - eta]).clamp_min(floor))
        fx = torch.log(torch.cat([P_new, A_new - eta]).clamp_min(floor)) - x
        res = float(fx.abs().max())
        if verbose and (it % 10 == 0 or res < tol):
            print(f"   it {it:4d}  res {res:.3e}  trace-id {tr_err:.2e}", flush=True)
        if res < tol:
            P, A, V = P_new, A_new, V_new
            break
        step = mixing * fx
        if anderson > 0 and res > 10 * best:                     # safeguard: history is misleading, restart
            xs, fs = [], []
        best = min(best, res)
        if anderson > 0:
            xs, fs = (xs + [x])[-(anderson + 1):], (fs + [fx])[-(anderson + 1):]
            if len(xs) > 1:
                dX = torch.stack([xs[i + 1] - xs[i] for i in range(len(xs) - 1)], 1)
                dF = torch.stack([fs[i + 1] - fs[i] for i in range(len(fs) - 1)], 1)
                # least squares via the small Gram matrix with a pseudo-inverse: the CUDA lstsq driver
                # assumes full rank, which fails for (near-)degenerate histories
                G = dF.T @ dF
                coef = torch.linalg.pinv(G, hermitian=True, rtol=1e-12) @ (dF.T @ fx)
                step = mixing * fx - (dX + mixing * dF) @ coef
        new = torch.exp(x + step.clamp(-2.0, 2.0))
        P, A, V = new[:m], eta + new[m:], V_new
    A_, P_, V_ = A.cpu().numpy(), P.cpu().numpy(), V.cpu().numpy()
    return OperatorSolution(A=A_, P=P_, V=V_, cdf=readout(prob, A_, V_), residual=res, iterations=it,
                            converged=bool(res < tol), trace_identity_error=tr_err,
                            seconds=time.monotonic() - start,
                            meta=dict(prob.meta, n_paths=int(d.shape[0]), tol=tol, mixing=mixing, anderson=anderson,
                                      gauge_fix=gauge_fix, complex_ring=not sym_even, init="uncoupled" if init is None else "continued"))


def continue_in_eta(d: Tensor, make_problem, etas, **kw) -> list[OperatorSolution]:
    """Regulator continuation: solve at decreasing ``etas`` (start large), warm-starting each from the last.

    ``make_problem(eta) -> Problem``.  The first solve starts from the uncoupled kernels, i.e. from the
    unique large-regulator solution, so the sequence follows the branch connected to it.
    """
    out, init = [], None
    for eta in etas:
        sol = solve(d, make_problem(eta), init=init, **kw)
        out.append(sol)
        init = (sol.P, sol.A)
    return out
