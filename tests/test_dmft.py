"""Stationary DMFT (paper [eq:x0lim], [eq:dmft]) at time step delta and in continuous time.

* Gaussian averages agree with adaptive quadrature.
* ``delta = 1``: white activity, ``q = g^2 E tanh^2(sqrt(q) z)``.
* Continuous time: the lag solver's variance equals the variance from the energy condition of the
  Sompolinsky-Crisanti-Sommers particle (an independent route, no lag grid), up to the O(dt^2) grid error.
* The solution satisfies the lag equation ``(1+b^2) Delta(tau) - b [Delta(tau+1) + Delta(tau-1)] = delta^2 g^2 C^phi``,
  is a valid covariance (non-negative power spectrum), and its ``C^phi``, ``C^d`` are the Gaussian pair averages.
* As ``delta -> 0`` the discrete DMFT approaches the continuous one.
* ``cached`` stores and reloads solutions; ``solve(g, 0)`` is continuous time.
"""
import numpy as np
import pytest
import torch
from scipy.integrate import quad

from rnn_lyapunov import dmft, max_exponent as mx


def test_gaussian_averages_against_adaptive_quadrature(ga):
    q = 5.4
    ref = quad(lambda z: np.tanh(np.sqrt(q) * z) ** 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi), -12, 12,
               epsabs=1e-13)[0]
    # trapezoid rule, exponentially convergent: ~1e-9 error at dz = 0.08 for q ~ 5
    assert abs(ga.one(lambda x: torch.tanh(x) ** 2, q) - ref) < 1e-9
    pair = ga.pair(dmft._phi, dmft._phi, q, np.array([q, 0.0, -q]))
    assert abs(pair[0] - ref) < 1e-9 and abs(pair[1]) < 1e-12 and abs(pair[2] + ref) < 1e-9
    # Price's theorem: d/dc E[tanh x tanh y] = E[sech^2 x sech^2 y]
    c, b = 2.0, 1e-4
    fd = (ga.pair(dmft._phi, dmft._phi, q, [c + b])[0] - ga.pair(dmft._phi, dmft._phi, q, [c - b])[0]) / (2 * b)
    assert abs(fd - ga.pair(dmft._dphi, dmft._dphi, q, [c])[0]) < 1e-7


def test_unit_step_variance_equation(sol_unit, ga):
    g = 3.0
    q = dmft.stationary_variance_unit_step(g, ga)
    assert abs(q - g * g * ga.one(lambda x: torch.tanh(x) ** 2, q)) < 1e-10
    assert sol_unit.time_step == 1.0 and sol_unit.dt == 1.0 and not sol_unit.continuous
    assert sol_unit.q == pytest.approx(q, abs=1e-12)
    assert np.all(sol_unit.c_x[1:] == 0)                                    # white activity
    assert abs(sol_unit.q - g * g * sol_unit.c_phi[0]) < 1e-10
    assert np.allclose(sol_unit.c_phi[1:], 0, atol=1e-14)
    # C^d at nonzero lag is <phi'>^2 (independent samples)
    assert np.allclose(sol_unit.c_d[1:], dmft.mean_gain(sol_unit, ga) ** 2, atol=1e-12)


def test_continuous_variance_equals_energy_condition(sol_cont, ga):
    q_energy = dmft.continuous_variance_energy_condition(3.0, ga)
    assert sol_cont.continuous and sol_cont.time_step is None and sol_cont.dt == 0.1
    # central differences on the lag grid: O(dt^2) error, observed 9e-5 at dt = 0.1
    assert abs(sol_cont.q - q_energy) < 3e-4
    assert sol_cont.q < q_energy                                            # approached from below


@pytest.mark.parametrize("which", ["sol_half", "sol_quarter", "sol_cont"])
def test_lag_equation_residual_and_valid_covariance(which, request, ga):
    sol = request.getfixturevalue(which)
    g = sol.g
    assert sol.residual < 1e-10 and sol.iterations >= 1
    assert dmft.min_relative_spectrum(sol.c_x) > -1e-6
    assert abs(sol.c_x[-1]) < 1e-4 * sol.q                                 # decayed within the lag window
    assert np.all(np.abs(sol.c_x) <= sol.q * (1 + 1e-12))
    # C^phi and C^d are the Gaussian pair averages of the solution
    assert np.allclose(sol.c_phi, ga.pair(dmft._phi, dmft._phi, sol.q, sol.c_x), atol=1e-13)
    assert np.allclose(sol.c_d, ga.pair(dmft._dphi, dmft._dphi, sol.q, sol.c_x), atol=1e-13)
    D = sol.c_x
    if sol.continuous:
        c0, c1, kappa = 1 + 2 / sol.dt**2, 1 / sol.dt**2, g * g
    else:
        b = 1 - sol.time_step
        c0, c1, kappa = 1 + b * b, b, (sol.time_step * g) ** 2
    for tau in (0, 1, 5, 17, 40):
        lhs = c0 * D[tau] - c1 * (D[tau + 1] + D[abs(tau - 1)])
        assert abs(lhs - kappa * sol.c_phi[tau]) < 1e-9 * kappa * sol.c_phi[0]
    assert np.allclose(sol.lags, np.arange(len(D)) * sol.dt)


def test_large_step_dmft_is_a_valid_covariance(ga):
    """Regression: at delta = 0.9 plain Newton from a smooth guess converged to a non-covariance solution
    (q = 5.2458); the physical solution has q = 5.469 (network simulation: 5.463)."""
    sol = dmft.solve(3.0, 0.9, t_max=30.0, ga=ga)
    assert dmft.min_relative_spectrum(sol.c_x) > -1e-6
    assert abs(sol.q - 5.4693) < 2e-3 and sol.residual < 1e-10


def test_discrete_approaches_continuous_as_delta_decreases(sol_half, sol_quarter, sol_cont):
    """delta = 0.5 -> 0.25 -> continuous: the variance and the upper edge converge, linearly in delta."""
    dq = [abs(s.q - sol_cont.q) for s in (sol_half, sol_quarter)]
    de = [abs(mx.upper_edge(s) - mx.upper_edge(sol_cont)) for s in (sol_half, sol_quarter)]
    assert dq[1] < dq[0] and de[1] < de[0]
    # O(delta) convergence: halving delta roughly halves the edge error (observed ratio 1.9)
    assert 1.6 < de[0] / de[1] < 2.4
    assert de[0] < 0.03 and dq[0] < 0.01


def test_solve_dispatch(monkeypatch):
    """delta = None or 0: continuous time on the lag grid dt; otherwise the map (dt unused)."""
    monkeypatch.setattr(dmft, "solve_continuous", lambda g, **kw: ("continuous", g, kw["dt"]))
    monkeypatch.setattr(dmft, "solve_discrete", lambda g, delta, **kw: ("map", g, delta))
    assert dmft.solve(3.0, 0, dt=0.2) == ("continuous", 3.0, 0.2)
    assert dmft.solve(3.0, None, dt=0.2) == ("continuous", 3.0, 0.2)
    assert dmft.solve(3.0, 0.5, dt=0.2) == ("map", 3.0, 0.5)


def test_cached_round_trip(tmp_path, monkeypatch, ga):
    kw = dict(t_max=30.0, ga=ga, directory=tmp_path)
    a = dmft.cached(3.0, 0.5, **kw)
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    # a second call loads from disk instead of solving again
    monkeypatch.setattr(dmft, "solve", lambda *a, **k: pytest.fail("cache miss"))
    b = dmft.cached(3.0, 0.5, **kw)
    assert np.array_equal(a.c_x, b.c_x) and np.array_equal(a.c_d, b.c_d) and a.time_step == b.time_step
    assert a.params == b.params and a.residual == b.residual


def test_cache_key(tmp_path, monkeypatch, ga):
    """Key: g, delta (None and 0 are the same), t_max, quadrature, and dt for continuous time only."""
    calls = []

    def fake_solve(g, delta, **k):
        calls.append((g, delta, k["dt"], k["t_max"]))
        return dmft.DMFTSolution(g, delta or None, k["dt"], np.ones(2), np.ones(2), np.ones(2), 0.0, 0)
    monkeypatch.setattr(dmft, "solve", fake_solve)
    kw = dict(t_max=30.0, ga=ga, directory=tmp_path)
    dmft.cached(3.0, 0.5, dt=0.1, **kw)
    dmft.cached(3.0, 0.5, dt=0.3, **kw)                    # map: dt is not part of the key
    dmft.cached(3.0, None, dt=0.2, **kw)
    dmft.cached(3.0, 0, dt=0.2, **kw)                      # same key as delta = None
    dmft.cached(3.0, None, dt=0.15, **kw)                  # continuous: dt is part of the key
    dmft.cached(3.0, 0.5, t_max=40.0, ga=ga, directory=tmp_path)
    dmft.cached(3.0, 0.5, t_max=30.0, ga=dmft.GaussianAverages(device="cpu"), directory=tmp_path)
    assert len(calls) == 5 and len(list(tmp_path.glob("*.pkl"))) == 5
    assert not list(tmp_path.glob("*.tmp*"))


def test_too_short_lag_window_is_rejected(ga):
    """Truncating C^x where it has not decayed (C^x(10) ~ 0.05 q) gives a lag solution with a negative power
    spectrum; the solver must refuse it rather than return a non-covariance."""
    with pytest.raises(RuntimeError, match="not a valid covariance"):
        dmft.solve(3.0, None, t_max=10.0, dt=0.2, ga=ga)


def test_cache_dir_from_environment(tmp_path, monkeypatch, ga):
    monkeypatch.setenv("RNN_LYAPUNOV_CACHE", str(tmp_path))
    assert dmft.cache_dir() == tmp_path / "dmft"
    dmft.cached(3.0, 1.0, ga=ga)
    assert len(list((tmp_path / "dmft").glob("*.pkl"))) == 1


@pytest.mark.slow
def test_continuous_variance_richardson(ga):
    """Two lag grids: Richardson extrapolation of the O(dt^2) error recovers the energy-condition variance."""
    q_energy = dmft.continuous_variance_energy_condition(3.0, ga)
    q1 = dmft.solve(3.0, None, dt=0.1, t_max=40.0, ga=ga).q
    q2 = dmft.solve(3.0, None, dt=0.05, t_max=40.0, ga=ga).q
    assert abs((4 * q2 - q1) / 3 - q_energy) < 2e-6
    assert abs(q2 - q_energy) < 3e-5


@pytest.mark.slow
def test_small_step_dmft_converges_linearly(ga):
    """delta = 0.1, 0.05: the upper edge approaches the continuous edge linearly (Richardson to < 3e-4)."""
    e = [mx.upper_edge(dmft.solve(3.0, d, t_max=40.0, ga=ga)) for d in (0.1, 0.05)]
    ce = mx.continuous_edge(dmft.solve(3.0, None, dt=0.05, t_max=40.0, ga=ga))
    assert abs(2 * e[1] - e[0] - ce) < 3e-4
    assert abs(e[1] - ce) < 0.5 * abs(e[0] - ce) + 1e-4
