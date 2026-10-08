"""Gain trajectories d_0(n) = phi'(x_0(n)) sampled from the stationary DMFT process on a ring.

* The ring covariance is the periodized DMFT covariance, with a non-negative spectrum.
* Sampled paths (pseudo-random and scrambled Sobol) reproduce the DMFT covariance and ``C^d(0)`` within sampling
  error; Sobol sampling needs a power-of-two path count.
"""
import numpy as np
import pytest

from rnn_lyapunov import gain_paths as gp


def test_ring_covariance_periodization(sol_half):
    m = 16
    cov = gp.ring_covariance(sol_half, m)
    L = len(sol_half.c_x)
    for n in (0, 3, 8):
        ref = sum(sol_half.c_x[abs(n + b * m)] for b in range(-L, L + 1) if abs(n + b * m) < L)
        assert cov[n] == pytest.approx(ref, abs=1e-14)
    assert np.allclose(cov[1:], cov[1:][::-1])                    # even on the ring
    spec = np.fft.fft(gp.ring_covariance(sol_half, 64)).real
    assert spec.min() > -1e-7 * spec.max()
    # stride: points spaced 2 lags apart
    assert gp.ring_covariance(sol_half, 64, stride=2)[1] == pytest.approx(gp.ring_covariance(sol_half, 128)[2], abs=1e-14)


@pytest.mark.parametrize("sobol", [False, True])
def test_sampled_paths_match_dmft(sol_half, sobol):
    m, n = 64, 4096
    out = gp.sample_gain_paths(sol_half, m, n, seed=0, sobol=sobol, device="cpu")
    assert out.d.shape == (n, m) and out.x.shape == (n, m)
    x = out.x.numpy()
    # n * m = 2.6e5 correlated samples (tau_c ~ 3 steps): s.e. of the lag covariance ~ 0.01 q; 0.03 q is ~3 s.e.
    for lag in (0, 2, 8):
        assert np.mean(x * np.roll(x, -lag, axis=1)) == pytest.approx(sol_half.c_x[lag], abs=0.03 * sol_half.q)
    assert np.allclose(out.d.numpy(), 1 / np.cosh(x) ** 2)
    assert out.info["sample_mean_sq_gain"] == pytest.approx(sol_half.c_d[0], abs=0.01)
    assert out.info["sobol"] is sobol and out.info["target_variance"] == sol_half.q


def test_sobol_needs_power_of_two(sol_half):
    with pytest.raises(AssertionError):
        gp.sample_gain_paths(sol_half, 16, 12, sobol=True, device="cpu")


def test_sampling_is_deterministic(sol_half):
    a = gp.sample_gain_paths(sol_half, 16, 32, seed=3, sobol=True, device="cpu")
    b = gp.sample_gain_paths(sol_half, 16, 32, seed=3, sobol=True, device="cpu")
    c = gp.sample_gain_paths(sol_half, 16, 32, seed=4, sobol=True, device="cpu")
    assert np.array_equal(a.d.numpy(), b.d.numpy()) and not np.array_equal(a.d.numpy(), c.d.numpy())
