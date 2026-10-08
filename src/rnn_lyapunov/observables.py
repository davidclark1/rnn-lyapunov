"""Observables of a Lyapunov spectrum: F(s), Kaplan-Yorke dimension, entropy rate.

From a list of exponents (simulation) or from the cumulative distribution ``F(s)`` (theory).  All per neuron:

    F(s)        fraction of exponents below s                                       (paper [eq:F])
    D_KY / N    Kaplan-Yorke dimension per neuron                                     (paper [eq:ky], [eq:dky_theory])
    h_KS / N    entropy rate per neuron = sum of the positive exponents / N           (paper [eq:hks_theory])

``N`` is always the full network size, so leading-subspace (partial) spectra are normalized correctly:
``F(s) = 1 - #{lambda_i > s} / N``, valid for ``s >= min(computed exponents)``.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


# ---- from exponents -----------------------------------------------------------------------------------------

def empirical_cdf(exponents: Array, s: Array, N: int | None = None) -> Array:
    """``F(s)`` on a grid; NaN below the smallest computed exponent of a partial spectrum."""
    lam = np.sort(np.asarray(exponents, float))
    N = len(lam) if N is None else N
    s = np.asarray(s, float)
    F = 1.0 - (len(lam) - np.searchsorted(lam, s, side="right")) / N
    if len(lam) < N:
        F = np.where(s < lam[0], np.nan, F)
    return F


def ky_dimension(exponents: Array, N: int | None = None) -> float:
    """Kaplan-Yorke dimension per neuron, ``(j + sum_{i<=j} lambda_i / |lambda_{j+1}|) / N``, with ``j`` the
    largest index at which the cumulative sum of the descending exponents is non-negative.  NaN if a partial
    spectrum does not reach the zero crossing of the cumulative sum."""
    lam = np.sort(np.asarray(exponents, float))[::-1]
    N = len(lam) if N is None else N
    cs = np.cumsum(lam)
    j = int(np.sum(cs >= 0))
    if j == 0:
        return 0.0
    if j < len(lam):
        return float((j + cs[j - 1] / abs(lam[j])) / N)
    return float("nan")


def entropy_rate(exponents: Array, N: int | None = None) -> float:
    """Sum of the positive exponents per neuron (``h_KS / N`` by Pesin's formula)."""
    lam = np.asarray(exponents, float)
    N = len(lam) if N is None else N
    return float(lam[lam > 0].sum() / N)


def spectrum_summary(exponents: Array, N: int | None = None) -> dict:
    """Scalar observables of a (possibly partial) spectrum, all per neuron."""
    lam = np.sort(np.asarray(exponents, float))[::-1]
    N = len(lam) if N is None else N
    full = len(lam) == N
    return dict(N=N, n_exponents=len(lam), full_spectrum=full, max_exponent=float(lam[0]),
                unstable_fraction=float(np.sum(lam > 0) / N), entropy_rate=entropy_rate(lam, N),
                ky_dimension=ky_dimension(lam, N), mean_exponent=float(lam.mean()) if full else np.nan)


def split_half_rms(block_logs: Array, block_time: float) -> float:
    """RMS difference between the exponents from the two halves of the observation window (a noise estimate)."""
    n = block_logs.shape[0] // 2
    a = block_logs[:n].astype(np.float64).mean(0) / block_time
    b = block_logs[n:2 * n].astype(np.float64).mean(0) / block_time
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---- from the cumulative distribution F(s) ----------------------------------------------------------------------

def dimension_entropy_from_cdf(s: Array, F: Array, edge: float, n_dense: int = 20001) -> dict:
    """Kaplan-Yorke dimension and entropy rate per neuron from ``F(s)`` known on thresholds ``s`` and the upper
    edge ``edge`` of the Lyapunov spectrum (``F(edge) = 1``).  Only the upper part of ``F`` is needed.

    With the rank fraction ``r(s) = 1 - F(s)`` and exponents ordered from the top (large-N forms of [eq:ky] and
    [eq:ruelle]; paper [eq:dky_theory], [eq:hks_theory])::

        h_KS / N = int_0^edge r(s) ds
        D_KY / N = r(s_KY),   where   C(s_KY) = s_KY r(s_KY) + int_{s_KY}^edge r(s) ds = 0

    ``C(s)`` is the sum of the exponents above ``s`` per neuron (integration by parts of ``int s dr``).  ``r`` is
    interpolated monotonically (PCHIP) on a dense grid.  ``D`` is NaN if the thresholds do not reach ``s_KY``.
    """
    from scipy.interpolate import PchipInterpolator
    s, F = np.asarray(s, float), np.asarray(F, float)
    keep = s < edge
    ss, rr = np.r_[s[keep], edge], np.r_[1.0 - F[keep], 0.0]
    order = np.argsort(ss)
    ss, rr = ss[order], np.maximum(rr[order], 0.0)
    r = PchipInterpolator(ss, rr)
    x = np.linspace(ss[0], edge, n_dense)
    rx = r(x)
    tail = np.r_[0.0, np.cumsum(0.5 * (rx[1:] + rx[:-1]) * np.diff(x))]      # int_{x0}^{x} r
    above = tail[-1] - tail                                                    # int_{x}^{edge} r
    C = x * rx + above
    H = float(np.interp(0.0, x, above)) if edge > 0 and x[0] <= 0 else (0.0 if edge <= 0 else np.nan)
    D = np.nan
    s_ky = np.nan
    if edge <= 0:
        D = 0.0
    else:
        neg = np.nonzero((C < 0) & (x < 0))[0]          # C increases with s for s < 0: a unique crossing s_KY < 0
        if len(neg) and neg[-1] + 1 < len(x):
            i = neg[-1]                                  # C < 0 at x[i], >= 0 at x[i+1]
            w = -C[i] / (C[i + 1] - C[i])
            D = float(rx[i] + w * (rx[i + 1] - rx[i]))
            s_ky = float(x[i] + w * (x[i + 1] - x[i]))
    return dict(entropy_rate=H, ky_dimension=D, s_ky=s_ky, s_ky_reached=bool(np.isfinite(D)))


def exponents_vs_rank(s: Array, F: Array, edge: float, n: int = 2000) -> tuple[Array, Array]:
    """The theory's Lyapunov spectrum as a curve: exponent ``s`` against rank fraction ``i/N = 1 - F(s)``.

    Monotone cubic (PCHIP) interpolation of ``s`` against ``1 - F(s)`` through the computed thresholds, closed at
    the upper edge (``F = 1``).  Returns ``(rank_fraction, exponent)`` on ``n`` points (paper Fig. spectra).
    """
    from scipy.interpolate import PchipInterpolator
    s, F = np.r_[np.asarray(s, float), edge], np.r_[np.asarray(F, float), 1.0]
    o = np.argsort(s)
    r, s = 1 - F[o][::-1], s[o][::-1]                       # increasing rank fraction, decreasing s
    keep = np.r_[True, np.diff(r) > 1e-12]
    rr = np.linspace(r[keep][0], r[keep][-1], n)
    return rr, PchipInterpolator(r[keep], s[keep])(rr)
