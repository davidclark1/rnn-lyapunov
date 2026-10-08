"""Observables of a Lyapunov spectrum (paper [eq:F], [eq:ky], [eq:dky_theory], [eq:hks_theory]).

* ``ky_dimension`` and ``entropy_rate`` on hand-computed examples, including partial spectra.
* ``empirical_cdf`` counts exponents below ``s`` per neuron; for a partial spectrum it is NaN below the smallest
  computed exponent.
* ``dimension_entropy_from_cdf`` (the large-N forms used for the theory) agrees with the exponent-list definitions
  for a large sample of exponents drawn from a smooth distribution.
* ``exponents_vs_rank`` inverts ``F``.
"""
import numpy as np
import pytest

from rnn_lyapunov import observables as ob


def test_ky_dimension_and_entropy_rate_hand_examples():
    lam = np.array([0.5, 0.2, -0.3, -1.0, -2.0])
    # cumulative sums 0.5, 0.7, 0.4, -0.6: j = 3, D = (3 + 0.4/1.0)/5
    assert ob.ky_dimension(lam) == pytest.approx(3.4 / 5, abs=1e-15)
    assert ob.entropy_rate(lam) == pytest.approx(0.7 / 5, abs=1e-15)
    # order does not matter; N normalizes a partial spectrum
    assert ob.ky_dimension(lam[::-1], N=50) == pytest.approx(3.4 / 50, abs=1e-15)
    assert ob.entropy_rate(lam[:3], N=50) == pytest.approx(0.7 / 50, abs=1e-15)
    # all negative: dimension zero, no entropy
    assert ob.ky_dimension(-np.arange(1.0, 4.0)) == 0.0 and ob.entropy_rate(-np.arange(1.0, 4.0)) == 0.0
    # partial spectrum not reaching the zero crossing of the cumulative sum: NaN
    assert np.isnan(ob.ky_dimension(np.array([0.5, 0.2, -0.3]), N=10))


def test_spectrum_summary():
    lam = np.array([-1.0, 0.3, -0.2, 0.1])
    s = ob.spectrum_summary(lam)
    assert s["max_exponent"] == 0.3 and s["unstable_fraction"] == 0.5 and s["full_spectrum"]
    assert s["mean_exponent"] == pytest.approx(-0.2) and s["entropy_rate"] == pytest.approx(0.1)
    # cumulative 0.3, 0.4, 0.2, -0.8: j = 3, D = (3 + 0.2/1)/4
    assert s["ky_dimension"] == pytest.approx(0.8)


def test_empirical_cdf_full_and_partial():
    lam = np.array([0.3, 0.1, -0.2])
    s = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    assert np.allclose(ob.empirical_cdf(lam, s), [0, 1 / 3, 1 / 3, 2 / 3, 1])
    F = ob.empirical_cdf(lam, s, N=10)
    assert np.isnan(F[0]) and np.allclose(F[1:], [0.8, 0.8, 0.9, 1.0])


def test_split_half_rms():
    logs = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    # halves: (1, 2) and (3, 2) per block time 0.5 -> (2, 4) and (6, 4); rms of (-4, 0) = sqrt(8)
    assert ob.split_half_rms(logs, 0.5) == pytest.approx(np.sqrt(8.0))


@pytest.fixture(scope="module")
def smooth_spectrum():
    """2e5 exponents from a smooth density on [-1.1, 0.3] with ~5% positive (beta-distributed)."""
    rng = np.random.default_rng(0)
    return np.sort(0.3 - 1.4 * rng.beta(2.0, 1.2, 200000))[::-1]


def test_dimension_entropy_from_cdf_matches_exponent_list(smooth_spectrum):
    lam = smooth_spectrum
    s = np.linspace(-1.2, 0.3, 40)
    out = ob.dimension_entropy_from_cdf(s, ob.empirical_cdf(lam, s), edge=0.3)
    # 40 thresholds and PCHIP interpolation of a smooth F: relative error ~1e-3 (observed 3e-4)
    assert out["entropy_rate"] == pytest.approx(ob.entropy_rate(lam), rel=2e-3)
    assert out["ky_dimension"] == pytest.approx(ob.ky_dimension(lam), rel=2e-3)
    assert out["s_ky_reached"] and -1.2 < out["s_ky"] < 0
    # D_KY/N = r(s_KY): the rank fraction at s_KY is the fraction of exponents above it
    assert out["ky_dimension"] == pytest.approx(np.mean(lam > out["s_ky"]), rel=5e-3)


def test_dimension_entropy_needs_thresholds_below_s_ky(smooth_spectrum):
    lam = smooth_spectrum
    hi = np.linspace(0.0, 0.3, 10)
    out = ob.dimension_entropy_from_cdf(hi, ob.empirical_cdf(lam, hi), edge=0.3)
    assert np.isnan(out["ky_dimension"]) and not out["s_ky_reached"]
    assert out["entropy_rate"] == pytest.approx(ob.entropy_rate(lam), rel=2e-3)   # h_KS needs only s >= 0


def test_dimension_entropy_non_chaotic_edge():
    s = np.linspace(-2, -0.5, 10)
    out = ob.dimension_entropy_from_cdf(s, np.linspace(0.1, 0.9, 10), edge=-0.4)
    assert out["ky_dimension"] == 0.0 and out["entropy_rate"] == 0.0


def test_dimension_entropy_uniform_closed_form():
    """F(s) uniform on [-1, 0.5]: r(s) = (0.5 - s)/1.5.  h = int_0^0.5 r = 1/12.  C(s) = s r + int_s^0.5 r
    = (0.25 - s^2)/3 = 0 at s_KY = -0.5, D = r(-0.5) = 2/3."""
    s = np.linspace(-1.0, 0.5, 31)[:-1]
    out = ob.dimension_entropy_from_cdf(s, (s + 1) / 1.5, edge=0.5)
    assert out["entropy_rate"] == pytest.approx(1 / 12, abs=1e-8)
    assert out["ky_dimension"] == pytest.approx(2 / 3, abs=1e-7)
    assert out["s_ky"] == pytest.approx(-0.5, abs=1e-7)


def test_exponents_vs_rank_inverts_F():
    """F(s) = 1 - ((0.5 - s)/1.5)^2 on [-1, 0.5]: the exponent at rank fraction r is 0.5 - 1.5 sqrt(r)."""
    edge = 0.5
    s = np.linspace(-1.0, edge, 61)[:-1]
    F = 1 - ((edge - s) / 1.5) ** 2
    r, lam = ob.exponents_vs_rank(s, F, edge, n=500)
    assert r[0] == pytest.approx(0.0) and r[-1] == pytest.approx(1.0) and np.all(np.diff(r) > 0)
    assert np.all(np.diff(lam) <= 0)                       # exponents decrease with rank
    # PCHIP on 61 nodes; the sqrt singularity at r = 0 limits accuracy near the top: check r > 0.01
    ok = r > 0.01
    assert np.allclose(lam[ok], edge - 1.5 * np.sqrt(r[ok]), atol=2e-3)
    assert lam[0] == pytest.approx(edge)
    # F evaluated at the returned exponents gives back 1 - r
    assert np.allclose(1 - ((edge - lam[ok]) / 1.5) ** 2, 1 - r[ok], atol=2e-3)
