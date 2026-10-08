"""Trivial fixed point (g < 1, paper Appendix "Trivial fixed point"): exact F(s) and its single-site readout.

* The single-site Fourier integral (paper [app:Ffp]) equals the circular-law overlap area (paper [app:Fexact]) for
  every delta (Gauss's law).
* At finite N, the fraction of exponents ``log|1 - delta + delta mu_i| / delta`` below ``s`` (``mu_i`` eigenvalues
  of J, paper [app:lam]) is close to the overlap fraction.
* As ``delta -> 0`` both approach the semicircle law (paper [app:semicircle]).
"""
import numpy as np
import pytest

from rnn_lyapunov import trivial_fixed_point as tfp


def _s_grid(g, delta, n=9):
    """Thresholds spanning the exponent range log|1-delta+delta mu|/delta for |mu| < g (both overlap regimes)."""
    lo = np.log(max(abs(1 - delta) - delta * g, 1e-3)) / delta if abs(1 - delta) > delta * g else -6.0
    hi = np.log(abs(1 - delta) + delta * g) / delta
    return np.linspace(lo - 0.1, hi + 0.1, n)


@pytest.mark.parametrize("delta", [0.1, 0.5, 0.8, 1.0, 1.5])
@pytest.mark.parametrize("g", [0.4, 0.9])
def test_single_site_readout_equals_overlap_fraction(g, delta):
    for s in _s_grid(g, delta):
        # periodic trapezoid sum with 2e5 nodes of an integrand with a kink: error ~1e-9
        assert tfp.single_site_cdf(g, delta, s) == pytest.approx(tfp.overlap_fraction(g, delta, s), abs=1e-6)


def test_overlap_fraction_limits():
    g, delta = 0.7, 0.5
    s_lo, s_hi = np.log(1 - delta - delta * g) / delta, np.log(1 - delta + delta * g) / delta
    assert tfp.overlap_fraction(g, delta, s_lo - 0.01) == 0.0
    assert tfp.overlap_fraction(g, delta, s_hi + 0.01) == 1.0
    vals = [tfp.overlap_fraction(g, delta, s) for s in np.linspace(s_lo, s_hi, 50)]
    assert np.all(np.diff(vals) >= 0)
    # delta = 1: c = 0, F = min(1, e^{2s}/g^2) (disk inside disk)
    for s in (-1.0, -0.5, np.log(g) + 0.1):
        assert tfp.overlap_fraction(g, 1.0, s) == pytest.approx(min(1.0, np.exp(2 * s) / g**2), abs=1e-14)


@pytest.mark.parametrize("delta", [0.3, 1.0])
def test_finite_N_eigenvalue_fraction(delta):
    """Circular law at N = 1000: fluctuations of the eigenvalue count are O(1/N) inside the disk, plus an O(N^-1/2)
    edge effect; 0.02 is a loose bound (observed <= 0.006)."""
    N, g = 1000, 0.8
    rng = np.random.default_rng(0)
    mu = np.linalg.eigvals(rng.normal(size=(N, N)) * g / np.sqrt(N))
    lam = tfp.exponents_from_eigenvalues(mu, delta)
    for s in np.quantile(lam, [0.1, 0.3, 0.5, 0.7, 0.9]):
        assert abs(np.mean(lam < s) - tfp.overlap_fraction(g, delta, s)) < 0.02


def test_exponents_from_eigenvalues_match_jacobian():
    """lambda_i = log|1 - delta + delta mu_i| / delta are log|eigenvalues| of M = (1-delta) I + delta J per step."""
    rng = np.random.default_rng(1)
    J = rng.normal(size=(40, 40)) * 0.8 / np.sqrt(40)
    for delta in (0.3, 1.0):
        lam = tfp.exponents_from_eigenvalues(np.linalg.eigvals(J), delta)
        direct = np.log(np.abs(np.linalg.eigvals((1 - delta) * np.eye(40) + delta * J))) / delta
        assert np.allclose(np.sort(lam), np.sort(direct), atol=1e-10)


@pytest.mark.parametrize("g", [0.5, 0.9])
def test_small_step_approaches_semicircle(g):
    s = np.linspace(-1 - g + 0.05, -1 + g - 0.05, 7)
    err = [max(abs(tfp.overlap_fraction(g, d, x) - tfp.semicircle_cdf(g, x)) for x in s) for d in (0.1, 0.01, 0.001)]
    # O(delta) convergence: each 10x smaller delta shrinks the error ~10x
    assert err[0] > err[1] > err[2] and err[2] < 2e-3 and err[1] / err[2] > 5
    assert np.allclose([tfp.single_site_cdf(g, 1e-3, x) for x in s], tfp.semicircle_cdf(g, s), atol=2e-3)


def test_semicircle_cdf_shape():
    g = 0.8
    assert tfp.semicircle_cdf(g, -1 - g - 0.1) == pytest.approx(0.0, abs=1e-15)
    assert tfp.semicircle_cdf(g, -1.0) == pytest.approx(0.5, abs=1e-15)
    assert tfp.semicircle_cdf(g, -1 + g + 0.1) == pytest.approx(1.0, abs=1e-15)
    # density = derivative: (2/(pi g^2)) sqrt(g^2 - (1+s)^2)
    x, b = -0.7, 1e-6
    dens = (tfp.semicircle_cdf(g, x + b) - tfp.semicircle_cdf(g, x - b)) / (2 * b)
    assert dens == pytest.approx(2 / (np.pi * g * g) * np.sqrt(g * g - (1 + x) ** 2), rel=1e-6)
