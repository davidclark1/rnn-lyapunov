"""Participation-ratio dimensions PR^x/N and PR^phi/N (paper [eq:PRdef] and Sec. [sec:dimension], Appendix "Participation-ratio
dimension at finite time step").

* ``delta = 1``: the numerical two-site theory equals the closed form ``PR^phi/N = 1 - r``,
  ``PR^x/N = (1 - r)/(2 - r)``, ``r = (g <phi'>)^4``.
* Finite step: the U/M form of psi^x, psi^phi equals the independent X/nu form (PRL Eq. 25 at finite step).
* Estimators: for iid samples the cross-block estimator is unbiased (PR/N = 1 for white data), the naive one is
  biased down; ``covariance_blocks`` accumulates exactly the second moments of the simulated trajectory.
* Slow: small-delta theory approaches continuous time; simulated networks agree with the theory.
"""
import numpy as np
import pytest
import torch

from rnn_lyapunov import dmft, network as net, participation as pr


@pytest.mark.parametrize("g", [2.0, 3.0, 6.0])
def test_unit_step_theory_equals_closed_form(g, ga):
    sol = dmft.solve(g, 1.0, ga=ga)
    a = dmft.mean_gain(sol, ga)
    num = pr.pr_theory(sol, a, min_points=2048, device="cpu")
    cf = pr.pr_unit_step_closed_form(g, a)
    assert num["pr_phi"] == pytest.approx(cf["pr_phi"], abs=1e-10)
    assert num["pr_x"] == pytest.approx(cf["pr_x"], abs=1e-10)
    assert 0 < cf["pr_x"] < cf["pr_phi"] < 1
    assert cf["r"] == pytest.approx((g * a) ** 4)
    # the default <phi'> is computed from the solution (with the default quadrature dz = 0.04 instead of the
    # fixture's 0.08: agreement to ~1e-9 at q ~ 30)
    assert pr.pr_theory(sol, min_points=2048, device="cpu")["mean_gain"] == pytest.approx(a, abs=1e-7)


def test_finite_step_matches_X_nu_form(sol_half, ga):
    """psi^phi = int (|X/(X - nu)|^2 - 1) C^phi C^phi,  psi^x = int ((2|X|^2 - nu^2)/|X - nu|^2 - 1) C^x C^x,
    with X = L(t1) L(t2), L = (e^{i t} - 1 + delta)/delta and nu = (g <phi'>)^2: an independent NumPy evaluation."""
    g, h = 3.0, 0.5
    a = dmft.mean_gain(sol_half, ga)
    r = pr.pr_theory(sol_half, a, min_points=1024, device="cpu")
    th, meas, Cx, Cp, Sx = pr.spectra(sol_half, 1024)
    assert meas == pytest.approx(1 / len(th)) and np.allclose(Sx, h / (np.exp(1j * th) - (1 - h)))
    L = (np.exp(1j * th) - 1 + h) / h
    X = L[:, None] * L[None, :]
    nu = (g * a) ** 2
    psi_phi = np.sum((np.abs(X / (X - nu)) ** 2 - 1) * np.outer(Cp, Cp)) * meas**2
    psi_x = np.sum(((2 * np.abs(X) ** 2 - nu**2) / np.abs(X - nu) ** 2 - 1) * np.outer(Cx, Cx)) * meas**2
    assert psi_phi / r["psi_phi"] == pytest.approx(1, abs=1e-10)
    assert psi_x / r["psi_x"] == pytest.approx(1, abs=1e-10)
    assert 0 < r["pr_x"] < r["pr_phi"] < 1
    # the spectra integrate back to the equal-time covariances
    assert np.sum(Cx) * meas == pytest.approx(sol_half.q, rel=1e-12)


def test_continuous_spectra_normalization(sol_cont):
    w, meas, Cx, Cp, Sx = pr.spectra(sol_cont, 4096)
    assert np.sum(Cx) * meas == pytest.approx(sol_cont.q, rel=1e-12)
    assert np.allclose(Sx, 1 / (1 + 1j * w))


def _blocks(rng, lam, O, B, n):
    S, m = [], []
    for _ in range(B):
        X = (rng.normal(size=(n, len(lam))) * np.sqrt(lam)) @ O.T
        S.append(X.T @ X)
        m.append(X.sum(0))
    return torch.as_tensor(np.array(S)), torch.as_tensor(np.array(m))


def test_cross_block_estimator_white_data():
    """iid N(0, I) samples: PR/N = 1; few samples per block (n = N/2) bias the naive estimator strongly down."""
    N, B, n = 200, 8, 100
    rng = np.random.default_rng(0)
    out = pr.pr_from_blocks(*_blocks(rng, np.ones(N), np.eye(N), B, n), n)
    # naive: E Tr(C^2) ~ N^2 (1 + N/(B n)): PR/N ~ 1/(1 + 0.25) = 0.8
    assert out["pr_naive"] == pytest.approx(0.8, abs=0.03)
    # cross-block: unbiased; jackknife s.e. ~ 0.01 here
    assert out["pr_cross"] == pytest.approx(1.0, abs=4 * out["pr_cross_se"] + 0.01)
    assert out["variance"] == pytest.approx(1.0, abs=0.02)


def test_cross_block_estimator_is_unbiased_for_structured_covariance():
    N, B, n = 300, 8, 150
    lam = 1.0 / (1.0 + np.arange(N)) ** 0.8                 # known covariance spectrum
    true_pr = lam.sum() ** 2 / (N * (lam**2).sum())
    cross, naive = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        O = np.linalg.qr(rng.normal(size=(N, N)))[0]
        out = pr.pr_from_blocks(*_blocks(rng, lam, O, B, n), n)
        cross.append(out["pr_cross"])
        naive.append(out["pr_naive"])
    assert np.mean(cross) == pytest.approx(true_pr, rel=0.02)   # 8 trials: s.e. of the mean ~0.8%
    assert np.mean(naive) < 0.97 * true_pr                      # biased down (~ -4.5% here)


def test_covariance_blocks_are_second_moments_of_the_trajectory():
    N, delta, every = 12, 0.5, 2
    J = net.make_coupling(N, 3.0, seed=4)
    blk = pr.covariance_blocks(J, delta=delta, t_burn=5.0, t_obs=40.0, n_blocks=4, sample_every=every, seed=4, batch=3)
    n = blk["n_per_block"]
    assert n == 10
    assert blk["Sx"].shape == (4, N, N) and blk["Sp"].shape == (4, N, N) and blk["mx"].shape == (4, N)
    X = net.simulate(J, delta=delta, t=5.0 + 40.0, seed=4)                  # states after each update
    samples = X[10 - 1 + every::every][: 4 * n].reshape(4, n, N)            # after the 10 burn-in updates
    for b in range(4):
        assert torch.allclose(blk["Sx"][b], samples[b].T @ samples[b], atol=1e-11)
        assert torch.allclose(blk["Sp"][b], torch.tanh(samples[b]).T @ torch.tanh(samples[b]), atol=1e-11)
        assert torch.allclose(blk["mx"][b], samples[b].sum(0), atol=1e-12)
    assert torch.equal(blk["x_final"], samples[-1, -1])


def test_pr_simulation_structure():
    J = net.make_coupling(20, 3.0, seed=0)
    out = pr.pr_simulation(J, delta=0.5, t_burn=10.0, t_obs=80.0, n_blocks=4, sample_dt=1.0)
    assert out["sample_every_steps"] == 2 and out["samples_per_block"] == 20      # 80 / (0.5 * 2) / 4
    for a in ("x", "phi"):
        assert set(out[a]) == {"pr_naive", "pr_cross", "pr_cross_se", "variance"}
        assert 0 < out[a]["pr_naive"] <= 1


@pytest.mark.slow
def test_small_step_map_theory_approaches_flow(ga):
    flow = dmft.solve(3.0, None, dt=0.05, t_max=40.0, ga=ga)
    a = pr.pr_theory(flow, device="cpu")
    m = dmft.solve(3.0, 0.05, t_max=40.0, ga=ga)
    b = pr.pr_theory(m, device="cpu")
    # O(delta) difference between the map at delta = 0.05 and the flow
    assert a["pr_phi"] == pytest.approx(b["pr_phi"], rel=0.05) and a["pr_x"] == pytest.approx(b["pr_x"], rel=0.05)


@pytest.mark.slow
def test_simulated_networks_agree_with_theory(sol_half):
    """g = 3, delta = 0.5, four networks with N = 1000 (cross-block estimator, t_obs = 8000).  Network-to-network
    relative spread of PR/N is ~10% at this N (seeds 0-2: 0.036-0.045 vs theory 0.042), so the mean of four is
    compared with a 12% tolerance."""
    th = pr.pr_theory(sol_half, device="cpu")
    xs, ps = [], []
    for seed in range(4):
        J = net.make_coupling(1000, 3.0, seed=seed)
        out = pr.pr_simulation(J, delta=0.5, t_obs=8000.0, seed=seed)
        xs.append(out["x"]["pr_cross"])
        ps.append(out["phi"]["pr_cross"])
        assert out["x"]["variance"] == pytest.approx(sol_half.q, rel=0.05)
    assert np.mean(xs) == pytest.approx(th["pr_x"], rel=0.12)
    assert np.mean(ps) == pytest.approx(th["pr_phi"], rel=0.12)
