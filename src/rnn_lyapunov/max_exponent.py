"""Maximum Lyapunov exponent: the operator ``T_s``, its continuous-time (Schrodinger) limit, and the upper edge.

Paper Appendix "Maximum Lyapunov exponent" ([app:T] etc.).  With the gain autocorrelation
``C^d(tau) = <d_0(n) d_0(n')>`` at lag ``tau = (n - n') delta``::

    (T_s psi)(tau) = (1 + alpha_s^2) psi(tau) - alpha_s [psi(tau + delta) + psi(tau - delta)] - gamma_s C^d(tau) psi(tau)
    alpha_s = (1 - delta) e^{-delta s},     gamma_s = delta^2 g^2 e^{-2 delta s}

``lambda_1 = s_*``, the unique ``s`` at which the bottom of the spectrum of ``T_s`` is zero; ``s_*`` is also the upper
edge of the single-site theory's ``F(s)``.  Limits:

    delta -> 0 :  H = -d^2/dtau^2 + 1 - g^2 C^d(tau),   s_* = -1 + sqrt(1 - E_0)    (Sompolinsky-Crisanti-Sommers)
    delta = 1  :  s_* = log(g sqrt(C^d(0)))                                          (Molgedey-Schuchhardt-Schuster)

Two levels of detail:

* :func:`upper_edge` (used by the theory pipeline): the operator on the DMFT lag window, Dirichlet beyond it.
* :func:`cd_profile`, :func:`scaled_operator`, :func:`s_star`, :func:`schrodinger` (used for the paper's figure
  max_exponent_operator): the operator on a window ``|tau| <= L`` of any length, with ``C^d(tau)`` continued
  beyond the DMFT window by its large-lag value ``<phi'>^2``.  ``a_s^{-1} T_s`` is the symmetric tridiagonal matrix
  ``(a + 1/a - (c/a) C^d) I - (S + S^+)`` (``S`` the lag shift), whose low-lying spectrum is computed exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from .dmft import DMFTSolution


def _two_sided(r: np.ndarray) -> np.ndarray:
    return np.r_[r[:0:-1], r]


# ---- upper edge on the DMFT lag window --------------------------------------------------------------------------

def discrete_edge(sol: DMFTSolution) -> float:
    """``s_*`` at the DMFT's time step: the zero of the smallest eigenvalue of ``e^{2 s delta} T_s`` on the DMFT lag
    window (``e^{2 s delta} + (1-delta)^2 - (1-delta) e^{s delta} (S + S^+) - delta^2 g^2 C^d``)."""
    delta, g = sol.time_step, sol.g
    assert delta is not None
    cd = _two_sided(sol.c_d)
    if delta == 1.0:
        return float(0.5 * np.log(g * g * cd.max()))
    b = 1.0 - delta

    def smallest(s):
        e = np.exp(s * delta)
        diag = e * e + b * b - (delta * g) ** 2 * cd
        return eigh_tridiagonal(diag, np.full(len(cd) - 1, -b * e), select="i", select_range=(0, 0))[0][0]
    # At the uncoupled exponent (alpha_s = 1) the kinetic term has a zero mode and the attractive potential makes the
    # smallest eigenvalue negative; above it the eigenvalue increases with s.
    return float(brentq(smallest, np.log(b) / delta, 5.0 * g, xtol=1e-13))


def continuous_edge(sol: DMFTSolution) -> float:
    """``s_* = -1 + sqrt(1 - E_0)`` of the Schrodinger operator on the DMFT lag grid (central differences)."""
    assert sol.time_step is None
    dt = sol.dt
    pot = 1.0 - sol.g**2 * _two_sided(sol.c_d)
    E0 = eigh_tridiagonal(pot + 2 / dt**2, np.full(len(pot) - 1, -1 / dt**2), select="i", select_range=(0, 0))[0][0]
    return float(-1.0 + np.sqrt(1.0 - E0))


def upper_edge(sol: DMFTSolution) -> float:
    """Upper edge ``s_*`` of the Lyapunov spectrum (= maximum exponent) from a DMFT solution."""
    return continuous_edge(sol) if sol.continuous else discrete_edge(sol)


# ---- the operator on an arbitrary lag window ----------------------------------------------------------------------

def cd_profile(sol: DMFTSolution, c_d_inf: float, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided ``C^d(tau)`` on ``tau = k dt``, ``|k| <= L/dt``, continued by ``c_d_inf`` beyond the DMFT window.

    ``c_d_inf = <phi'(x_0)>^2`` (:func:`rnn_lyapunov.dmft.mean_gain` squared) is the large-lag value."""
    r = np.asarray(sol.c_d, float)
    K = max(int(round(L / sol.dt)), len(r) - 1)
    full = np.full(K + 1, c_d_inf)
    full[:len(r)] = r
    return np.arange(-K, K + 1) * sol.dt, np.r_[full[:0:-1], full]


def scaled_operator(delta: float, g: float, s: float, cd: np.ndarray):
    """Diagonal and off-diagonal of the tridiagonal ``alpha_s^{-1} T_s``, and ``alpha_s``, ``gamma_s``:
    ``(diag, off, alpha_s, gamma_s)``."""
    a = (1 - delta) * np.exp(-delta * s)
    c = (delta * g) ** 2 * np.exp(-2 * delta * s)
    return a + 1 / a - (c / a) * cd, -np.ones(len(cd) - 1), a, c


def lowest(diag: np.ndarray, off: np.ndarray, k: int = 1, vectors: bool = False):
    """The ``k`` lowest eigenvalues (and eigenvectors) of a symmetric tridiagonal matrix."""
    return eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1), eigvals_only=not vectors)


def s_star(delta: float, g: float, cd: np.ndarray) -> float:
    """``s_*``: the zero of the bottom of the spectrum of ``a_s^{-1} T_s`` on the given ``C^d`` profile.
    At ``delta = 1``: ``log(g sqrt(max C^d))``."""
    if delta == 1.0:
        return float(np.log(g * np.sqrt(cd.max())))
    s_u = np.log(1 - delta) / delta
    return float(brentq(lambda s: lowest(*scaled_operator(delta, g, s, cd)[:2])[0], s_u + 1e-10, 5 * g, xtol=1e-13))


def schrodinger(g: float, tau: np.ndarray, cd: np.ndarray, k: int = 6):
    """Lowest ``k`` eigenpairs of ``H = -d^2/dtau^2 + 1 - g^2 C^d(tau)`` (central differences, Dirichlet at the window
    ends).  Returns the potential ``V``, energies ``E`` and eigenfunctions normalized to ``int psi^2 dtau = 1``."""
    dt = tau[1] - tau[0]
    V = 1 - g * g * cd
    E, psi = lowest(V + 2 / dt**2, np.full(len(V) - 1, -1 / dt**2), k=k, vectors=True)
    psi = psi / np.sqrt(dt * (psi**2).sum(0))
    return V, E, psi


def s_star_from_energy(E0: float) -> float:
    """Continuous time: ``s_* = -1 + sqrt(1 - E_0)`` (the energy-to-exponent map ``E(s) = 1 - (1 + s)^2``)."""
    return float(-1 + np.sqrt(1 - E0))
