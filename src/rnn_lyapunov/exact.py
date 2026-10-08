"""Exact finite-N objects of the derivation (paper Parts 1-2), dense NumPy, for small networks and tests.

On a window of ``m`` steps (times ``0..m``) the shifted tangent dynamics driven by a source,
``v(n+1) = M_s(n) v(n) + I^v(n)`` (paper [eq:driven]), read ``K V = I^v`` (paper [eq:KV]) with the block-bidiagonal
``K`` (``mN x (m+1)N``, paper [eq:K]).  ``K`` leaves the tangent vector at one time free; the minimum-norm solution
``V = K^+ I^v`` removes this freedom.  Its response ``R^vv(n, n') = dv(n)/dI^v(n')`` (paper [eq:Rvvdef]) gives, deep
inside a long window,

    F(s) = (1/N) tr R^vv(k+1, k)                                                      (paper [eq:F])

the fraction of exponents below ``s`` (``R^vv(k+1, k)`` is the projector onto the stable subspace along the
unstable one).  The regularized solution ``V_eta = argmin ||K V - I^v||^2 + eta^2 ||V||^2`` (paper [eq:Veta]),
``V_eta = (K^T K + eta^2)^{-1} K^T I^v``, tends to the minimum-norm solution as ``eta -> 0+``; it is the forward
field of the forward-backward system ``B_eta (U, V) = (I^v, I^u)`` (paper [eq:B], [eq:Bdef]) at ``I^u = 0``, and
``||B_eta^{-1}||_op <= 1/eta``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

Array = np.ndarray


def tangent_matrices(J: Array, gains: Array, delta: float) -> list[Array]:
    """``M(n) = (1 - delta) I_N + delta J diag(d(n))`` for gains of shape ``(m, N)``."""
    eye = np.eye(J.shape[0])
    return [(1.0 - delta) * eye + delta * J * d[None, :] for d in gains]


def shifted(matrices: Sequence[Array], s: float, delta: float) -> list[Array]:
    """``M_s(n) = e^{-delta s} M(n)`` (paper [eq:Ms]): shifts every exponent down by ``s``."""
    f = np.exp(-s * delta)
    return [f * M for M in matrices]


def residual_matrix(Ms: Sequence[Array]) -> Array:
    """``K`` on a window of ``m`` steps: ``(K V)(n) = v(n+1) - M_s(n) v(n)``, shape ``(mN, (m+1)N)``."""
    m, N = len(Ms), Ms[0].shape[0]
    K = np.zeros((m * N, (m + 1) * N))
    for n, M in enumerate(Ms):
        K[n * N:(n + 1) * N, n * N:(n + 1) * N] = -M
        K[n * N:(n + 1) * N, (n + 1) * N:(n + 2) * N] = np.eye(N)
    return K


def forward_backward_matrix(K: Array, eta: float) -> Array:
    """``B_eta = [[eta I, K], [-K^T, eta I]]`` (paper [eq:Bdef]): ``eta`` times the identity plus an antisymmetric
    matrix, so ``||B_eta^{-1}||_op <= 1/eta``."""
    r, c = K.shape
    return np.block([[eta * np.eye(r), K], [-K.T, eta * np.eye(c)]])


def response_vv(K: Array, eta: float = 0.0) -> Array:
    """``R^vv`` of the regularized solution, ``(K^T K + eta^2)^{-1} K^T``; at ``eta = 0`` the minimum-norm solution's
    ``K^+``.  Shape ``((m+1)N, mN)``: block ``(n, n')`` is ``dv(n)/dI^v(n')``."""
    if eta == 0.0:
        return np.linalg.pinv(K)
    return np.linalg.solve(K.T @ K + eta**2 * np.eye(K.shape[1]), K.T)


def cdf_from_response(Ms: Sequence[Array], k: int | None = None, eta: float = 0.0) -> float:
    """``(1/N) tr R^vv(k+1, k)`` for the source at step ``k`` (default: the middle of the window)."""
    m, N = len(Ms), Ms[0].shape[0]
    k = m // 2 if k is None else k
    R = response_vv(residual_matrix(Ms), eta)
    return float(np.trace(R[(k + 1) * N:(k + 2) * N, k * N:(k + 1) * N]) / N)


def simulate_gains(J: Array, delta: float, m: int, burn: int = 2000, seed: int = 0) -> Array:
    """Gains ``d(n) = sech^2 x(n)`` along ``m`` steps of the Euler map after ``burn`` steps (NumPy, small N)."""
    x = np.random.default_rng(seed).normal(size=J.shape[0])
    for _ in range(burn):
        x = (1 - delta) * x + delta * J @ np.tanh(x)
    out = np.empty((m, J.shape[0]))
    for n in range(m):
        out[n] = 1.0 / np.cosh(x) ** 2
        x = (1 - delta) * x + delta * J @ np.tanh(x)
    return out
