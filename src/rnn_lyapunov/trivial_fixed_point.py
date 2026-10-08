"""The trivial fixed point ``x = 0`` (``g < 1``): exact ``F(s)``, the single-site readout, and the ``delta -> 0`` limit.

Paper Appendix "Trivial fixed point".  Every gain is 1, so the Jacobian is the constant matrix
``(1 - delta) I_N + delta J`` and the exponents are ``lambda_i = log|1 - delta + delta mu_i| / delta`` with ``mu_i`` the
eigenvalues of ``J`` (paper [app:lam]).  As ``N -> infinity``, ``mu`` is uniform on the disk ``|mu| < g``, and
``lambda < s`` exactly when ``mu`` lies in the disk ``|mu - c_delta| < r_s``, ``c_delta = 1 - 1/delta``,
``r_s = e^{delta s} / delta``.  Hence (paper [app:Fexact])::

    F(s) = area(|mu| < g  and  |mu - c_delta| < r_s) / (pi g^2)

The single-site theory with ``d_0(n) = 1`` is translation invariant and gives the Fourier integral
:func:`single_site_cdf` (paper [app:Ffp]), equal to the exact result for every ``delta`` (Gauss's law).  As
``delta -> 0``, ``F'(s)`` becomes the semicircle law of radius ``g`` centered at ``-1`` (paper [app:semicircle]).
"""
from __future__ import annotations

import numpy as np


def exponents_from_eigenvalues(mu: np.ndarray, delta: float) -> np.ndarray:
    """``lambda_i = log|1 - delta + delta mu_i| / delta``: exponents of the constant Jacobian (paper [app:lam])."""
    return np.log(np.abs(1 - delta + delta * np.asarray(mu))) / delta


def overlap_fraction(g: float, delta: float, s: float) -> float:
    """Exact ``F(s)``: area of the overlap region of ``|mu| < g`` and ``|mu - c_delta| < r_s``, divided by ``pi g^2``."""
    a, c, b = g, 1 - 1 / delta, np.exp(delta * s) / delta
    dist = abs(c)
    if dist >= a + b:
        return 0.0
    if dist <= abs(b - a):
        return min(a, b) ** 2 / a**2
    t1 = a * a * np.arccos((dist**2 + a * a - b * b) / (2 * dist * a))
    t2 = b * b * np.arccos((dist**2 + b * b - a * a) / (2 * dist * b))
    t3 = 0.5 * np.sqrt((-dist + a + b) * (dist + a - b) * (dist - a + b) * (dist + a + b))
    return float((t1 + t2 - t3) / (np.pi * a * a))


def single_site_cdf(g: float, delta: float, s: float, n: int = 200001) -> float:
    """Single-site readout at the trivial fixed point (paper [app:Ffp]) on ``n`` frequencies ``theta``::

        F(s) = int dtheta/2pi  { (1 - alpha cos theta)/gamma           where |e^{i theta} - alpha|^2 < gamma
                               { Re 1/(1 - alpha e^{-i theta})         otherwise
    with ``alpha = alpha_s`` and ``gamma = gamma_s``.
    """
    al, be = (1 - delta) * np.exp(-delta * s), delta * np.exp(-delta * s)
    gam = (be * g) ** 2
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    l2 = np.abs(np.exp(1j * th) - al) ** 2
    f = np.where(l2 < gam, (1 - al * np.cos(th)) / gam, np.real(1 / (1 - al * np.exp(-1j * th))))
    return float(f.mean())


def semicircle_cdf(g: float, s):
    """``delta -> 0``: ``F(s)`` of the semicircle law of radius ``g`` centered at ``-1`` (paper [app:semicircle])."""
    c = np.clip((1 + np.asarray(s, float)) / g, -1, 1)
    return 1 - np.arccos(c) / np.pi + c * np.sqrt(1 - c * c) / np.pi
