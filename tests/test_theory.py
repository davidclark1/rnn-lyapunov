"""The theory pipeline (paper Sec. "Numerical solution and simulations"): F(s) from physical parameters.

* ``TheoryConfig``: delta = 0 means continuous time, which needs an explicit grid ``m``.
* Threshold rules are built from DMFT quantities only (``s_u`` and the upper edge): sorted, unique, below the edge.
* Bloch-twist rule, ``a_eff``, ``s_u``; quadratic extrapolation to zero grid spacing is exact for quadratic data.
* A tiny ``cdf_curve`` runs on the CPU: F in [0, 1], monotone in s, converged, twist bookkeeping, readout parts.
* Slow: a moderate theory run agrees with a simulated network's empirical F(s).
"""
import functools

import numpy as np
import pytest
import torch

from rnn_lyapunov import dmft, max_exponent as mx, theory
from rnn_lyapunov import lyapunov as ly, network as net, observables as ob


@pytest.fixture
def coarse_quadrature(monkeypatch):
    """The pipeline builds its own Gaussian quadrature; use dz = 0.08 (identical results to ~1e-13 at g = 3,
    see conftest) to keep the DMFT step fast on a CPU."""
    monkeypatch.setattr(theory.dmft, "GaussianAverages",
                        functools.partial(dmft.GaussianAverages, dz=0.08))


def test_config():
    cfg = theory.TheoryConfig(g=3.0, delta=0.5)
    assert cfg.grid() == 64 and cfg.spacing() == 0.5 and not cfg.continuous
    c = theory.TheoryConfig(g=3.0, delta=0)
    assert c.delta is None and c.continuous
    with pytest.raises(ValueError):
        c.grid()
    c = theory.TheoryConfig(g=3.0, delta=None, m=160)
    assert c.grid() == 160 and c.spacing() == pytest.approx(0.2)
    assert theory.TheoryConfig(g=3.0, delta=0.5).eps == theory.PAPER_EPS


def test_uncoupled_exponent_and_a_eff():
    assert theory.uncoupled_exponent(None) == -1.0 and theory.uncoupled_exponent(0) == -1.0
    for delta in (0.25, 0.5, 0.9):
        s_u = theory.uncoupled_exponent(delta)
        assert s_u == pytest.approx(np.log(1 - delta) / delta)
        assert theory.a_eff(s_u, delta) == pytest.approx(0.0, abs=1e-14)
        # a_eff -> s + 1 as delta -> 0
    assert theory.a_eff(-0.3, 1e-6) == pytest.approx(0.7, abs=1e-5)
    assert theory.a_eff(-0.3, None) == pytest.approx(0.7)


def test_twist_rule():
    th, w = theory.twist_rule(0)
    assert np.array_equal(th, [0.0]) and np.array_equal(w, [1.0])
    th, w = theory.twist_rule(4)
    assert len(th) == 4 and np.all((th > 0) & (th < 0.5)) and w.sum() == pytest.approx(1.0)
    # Gauss-Legendre on [0, 1/2], normalized: exact for polynomials of degree 7 in theta (mean of theta^3 = 1/32)
    assert np.sum(w * th**3) == pytest.approx(1 / 32, abs=1e-14)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
@pytest.mark.parametrize("delta,edge", [(0.5, 0.216), (0.25, 0.225), (None, 0.235), (1.0, 0.31)])
def test_threshold_rules(delta, edge):
    cfg = theory.TheoryConfig(g=3.0, delta=delta, m=128)
    s_u = theory.uncoupled_exponent(delta)
    full = theory.thresholds_full(cfg, edge=edge)
    top = theory.thresholds_top(cfg, -0.4, edge=edge)
    if delta == 1.0:
        # s_u = -inf at delta = 1: thresholds_upper is undefined there (it raises; noted in the test report)
        with pytest.raises(ValueError):
            theory.thresholds_upper(cfg, edge=edge)
        upper = theory.thresholds_upper(theory.TheoryConfig(g=3.0, delta=0.5), edge=edge)
    else:
        upper = theory.thresholds_upper(cfg, edge=edge)
    for s in (full, upper, top):
        assert np.all(np.diff(s) > 0)                        # sorted, unique
        assert s.max() < edge
    assert np.sum(full >= 0) == 10 and full[full >= 0][0] == 0.0
    if delta != 1.0:
        assert full.min() < s_u < full.max()               # both sides of the steepest point of F
        assert np.all(upper > s_u)
        assert np.min(np.abs(full - s_u)) == pytest.approx(0.015, abs=1e-6)
    assert len(top) == 24 and top[0] == -0.4 and np.all(top >= -0.4)
    assert np.sum(upper >= 0) == 12


def test_extrapolate_to_zero_spacing_is_exact_for_quadratic_error():
    v = lambda h: 0.7 - 2.3 * h**2
    assert theory.extrapolate_to_zero_spacing(v(0.2), 0.2, v(0.125), 0.125) == pytest.approx(0.7, abs=1e-14)


S_TINY = [-3.0, -1.0, -0.5, 0.05]


@pytest.fixture
def tiny_result(coarse_quadrature):
    cfg = theory.TheoryConfig(g=3.0, delta=0.5, T=8.0, n_paths=16, eps=(1e-1, 1e-2), n_twist=2, dmft_tmax=40.0)
    calls = []
    res = theory.cdf_curve(cfg, S_TINY, device="cpu", verbose=False, checkpoint=lambda r: calls.append(r))
    return cfg, res, calls


def test_tiny_cdf_curve_on_cpu(tiny_result):
    cfg, res, calls = tiny_result
    F = res["F"]
    assert F.shape == (4, 2) and res["F_theta"].shape == (4, 2, 2)
    assert res["all_converged"] and res["n_thresholds_done"] == 4
    assert np.all((F >= 0) & (F <= 1))
    assert np.all(np.diff(F, axis=0) > 0)                    # monotone in s at every regulator
    # twists are used only where |a_eff| T < 12: not at s = -3 (a_eff = -2.5), at the other thresholds
    assert list(res["twisted"]) == [False, True, True, True]
    assert np.array_equal(res["F_theta"][0, :, 0], res["F_theta"][0, :, 1])
    # the readout splits into leak + contact + regulator (twist-averaged)
    assert np.allclose(res["readout_parts"].sum(-1), F, atol=1e-12)
    # the edge is the DMFT upper edge of the configuration; the DMFT summary is recorded
    assert res["edge"] == pytest.approx(mx.upper_edge(theory.dmft_solution(cfg, "cpu")), abs=1e-14)
    assert 0.2 < res["edge"] < 0.23 and res["dmft"]["residual"] < 1e-10
    assert res["config"]["g"] == 3.0 and res["gain_sample"]["n_paths"] == 16
    # one checkpoint per threshold, with the thresholds done so far
    assert [c["n_thresholds_done"] for c in calls] == [1, 2, 3, 4]
    assert np.isnan(calls[0]["F"][1:]).all()
    out = theory.dimension_entropy(res)
    assert set(out) >= {"ky_dimension", "entropy_rate", "s_ky"} and out["entropy_rate"] > 0


def test_cdf_curve_warm_start_does_not_change_the_fixed_point(tiny_result):
    cfg, res, _ = tiny_result
    cold = theory.TheoryConfig(**{**res["config"], "warm_start_in_s": False})
    out = theory.cdf_curve(cold, S_TINY, device="cpu", verbose=False)
    # both converge to tol = 1e-7 in log(P, A): F agrees to ~1e-7
    assert np.allclose(out["F"], res["F"], atol=1e-6)


def test_cdf_curve_with_external_gain_trajectories(coarse_quadrature):
    """Constant gains d = 1/3 at g = 3 behave like the trivial fixed point at g = 1 (gain g d = 1)."""
    from rnn_lyapunov import trivial_fixed_point as tfp
    cfg = theory.TheoryConfig(g=3.0, delta=0.5, T=32.0, n_paths=1, sobol=False, eps=(1e-1, 1e-2, 1e-3), n_twist=0,
                              dmft_tmax=40.0)
    d = torch.full((1, cfg.grid()), 1.0 / 3.0, dtype=torch.float64)
    res = theory.cdf_curve(cfg, [-1.0, -0.4], device="cpu", verbose=False, gain_trajectories=d)
    exact = [tfp.overlap_fraction(1.0, 0.5, s) for s in (-1.0, -0.4)]
    # eps = 1e-3, m = 64 ring: observed agreement to ~5e-4
    assert np.allclose(res["F"][:, -1], exact, atol=3e-3)
    assert res["gain_sample"] == {"source": "external gain paths"}


@pytest.mark.slow
def test_theory_agrees_with_simulated_network(coarse_quadrature):
    """g = 3, delta = 0.5: the single-site F(s) vs the full Lyapunov spectrum of one network (N = 512).
    Tolerance 0.02: finite-N (~1/N plus network-to-network fluctuations), finite observation time, finite
    regulator and 256 gain paths on T = 16 (each ~5e-3 or less)."""
    g, delta = 3.0, 0.5
    s = np.array([-2.0, -1.4, -0.8, -0.4, -0.1, 0.1])
    cfg = theory.TheoryConfig(g=g, delta=delta, T=16.0, n_paths=256, eps=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3), n_twist=2)
    res = theory.cdf_curve(cfg, s, device="cpu", verbose=False)
    assert res["all_converged"]
    N = 512
    J = net.make_coupling(N, g, seed=1)
    lam = ly.lyapunov_spectrum(J, delta=delta, t_burn=100, t_tangent_burn=50, t_obs=200, seed=1).exponents
    F_net = ob.empirical_cdf(lam, s)
    assert np.allclose(res["F"][:, -1], F_net, atol=0.02)
    assert abs(lam.max() - res["edge"]) < 0.05                # maximum exponent vs s_*


def test_cdf_curve_is_independent_of_batch_chunking(tiny_result):
    """Splitting the gain trajectories into GPU batches only changes the summation order."""
    cfg, res, _ = tiny_result
    out = theory.cdf_curve(theory.TheoryConfig(**{**res["config"], "chunk": 5}), S_TINY, device="cpu", verbose=False)
    assert np.allclose(out["F"], res["F"], atol=1e-12)


def test_thresholds_upper_rejects_unit_step():
    with pytest.raises(ValueError, match="uncoupled exponent"):
        theory.thresholds_upper(theory.TheoryConfig(g=3.0, delta=1.0), edge=0.31)
