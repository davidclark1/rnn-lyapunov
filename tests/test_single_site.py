"""Single-site theory at fixed regulator (paper [eq:Bsite]-[eq:Fsite]): solver identities and solvable limits.

* The sweep's ``Q_d``, ``X_d`` are the vu and uv blocks of the single-site forward-backward matrix built with the
  true (complex, twisted) ``L``, projected onto their stationary parts.
* Uncoupled network (``g = 0``): ``A_cal = eta``, ``P = V = eta/(eta^2 + |L|^2)`` and the closed-form readout.
* Constant gains ``d_0(n) = d``: the full solver reproduces the circular-law ``F(s)`` of the trivial fixed point
  (paper [app:Fexact], via Gauss's law) as ``eta -> 0``, for the map, and the semicircle law in continuous time.
* ``Tr P = Tr V`` at every fixed point with ``eta > 0`` (gauge fixing does not move the fixed point).
* ``delta = 1``: the kernels are time-diagonal.
* ``_tri_inv`` / ``_pd_inv`` are exact, including the block recursion used for ``n > 512``.
* Twisted symbols lead to a complex ring; the discrete readout splits into leak + contact + regulator.
"""
import numpy as np
import pytest
import torch
from scipy.integrate import quad

from rnn_lyapunov import single_site as ss
from rnn_lyapunov import trivial_fixed_point as tfp

DEV = "cpu"


def _random_gains(paths, m, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return 0.05 + 0.95 * torch.rand(paths, m, generator=gen, dtype=torch.float64)


def _const_gains(m, d0=1.0):
    return torch.full((1, m), d0, dtype=torch.float64)


def test_frequencies():
    w = ss.frequencies(8)
    assert np.allclose(w, 2 * np.pi * np.fft.fftfreq(8))
    assert np.allclose(ss.frequencies(8, 0.25) - w, 2 * np.pi * 0.25 / 8)


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_sweep_blocks_are_single_site_responses(theta):
    """Q_d = R^vu_00, R^vv_00 = Q_d L^+ A^-1 and X_d = R^uv_00 (Woodbury) are the blocks of the inverse of
    [[A_cal, L], [-L^+, C_d]] built with the true L; the sweep (with |L|) reproduces their stationary projections."""
    m, eta, gam, delta, s = 12, 0.13, 0.8, 0.4, -0.2
    rng = np.random.default_rng(0)
    w = ss.frequencies(m, theta)
    alpha = (1 - delta) * np.exp(-s * delta)
    Lsym = np.exp(1j * w) - alpha
    P = rng.uniform(0.5, 1.5, m)
    A = eta + rng.uniform(0.5, 1.5, m)
    if theta == 0.0:                                   # real problem: symbols even in k
        P, A = 0.5 * (P + np.roll(P[::-1], 1)), 0.5 * (A + np.roll(A[::-1], 1))
    d = rng.uniform(0.1, 1.0, m)
    n = np.arange(m)
    F = np.exp(2j * np.pi * np.outer(n, n) / m) / np.sqrt(m)          # plain Fourier basis, columns e_k
    circ = lambda sym: F @ np.diag(sym) @ F.conj().T
    Lm, Am, Pm = circ(Lsym), circ(A), circ(P)
    Cd = eta * np.eye(m) + gam * d[:, None] * Pm * d[None, :]
    Minv = np.linalg.inv(np.block([[Am, Lm], [-Lm.conj().T, Cd]]))
    Ruv, Rvu, Rvv = Minv[:m, :m], Minv[m:, m:], Minv[m:, :m]
    Q = np.linalg.inv(Cd + Lm.conj().T @ np.linalg.inv(Am) @ Lm)
    assert np.allclose(Rvu, Q, atol=1e-10)
    assert np.allclose(Rvv, Q @ Lm.conj().T @ np.linalg.inv(Am), atol=1e-10)
    assert np.allclose(Ruv, np.linalg.inv(Am + Lm @ np.linalg.inv(Cd) @ Lm.conj().T), atol=1e-10)
    ring = ss.Ring(m, complex_=theta != 0.0, device=DEV)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    A_new, P_new, V_new = ss.sweep(t(d)[None], t(np.abs(Lsym) ** 2), gam, eta, t(P), t(A), ring)
    proj = lambda M: np.real(np.diag(F.conj().T @ M @ F))
    assert np.allclose(V_new.numpy(), proj(Rvu), atol=1e-10)
    assert np.allclose(P_new.numpy(), proj(Ruv), atol=1e-10)
    assert np.allclose(A_new.numpy(), eta + gam * proj(d[:, None] * Rvu * d[None, :]), atol=1e-10)


def test_ring_circulant_and_symbol_round_trip():
    m = 10
    rng = np.random.default_rng(2)
    sym = rng.uniform(0.5, 2.0, m)
    sym = 0.5 * (sym + np.roll(sym[::-1], 1))
    ring = ss.Ring(m, complex_=False, device=DEV)
    C = ring.circulant(torch.as_tensor(sym))
    assert C.dtype == torch.float64 and torch.allclose(C, C.T)
    assert np.allclose(ring.symbol_of(C).numpy(), sym, atol=1e-13)
    assert np.allclose(np.linalg.eigvalsh(C.numpy()), np.sort(sym), atol=1e-12)


@pytest.mark.parametrize("delta,s", [(0.3, -0.5), (0.5, -2.0), (1.0, 0.0)])
def test_uncoupled_closed_form(delta, s):
    """g = 0: the solver returns the uncoupled kernels and F_eta = int dw/2pi (eta^2 + 1 - alpha cos w)/(eta^2 + |L|^2)."""
    m, eta = 512, 0.19
    prob = ss.discrete_problem(m, s, delta, 0.0, eta)
    sol = ss.solve(_const_gains(m), prob)
    P0, A0 = ss.uncoupled_kernels(prob)
    assert sol.converged and np.allclose(sol.A, A0) and np.allclose(sol.P, P0, rtol=1e-12)
    assert np.allclose(sol.V, P0, rtol=1e-12)
    alpha = prob.meta["alpha"]
    f = lambda w: (eta**2 + 1 - alpha * np.cos(w)) / (eta**2 + 1 + alpha**2 - 2 * alpha * np.cos(w))
    ref = quad(f, -np.pi, np.pi, epsabs=1e-13)[0] / (2 * np.pi)
    # periodic trapezoid rule of an analytic periodic integrand: exponentially accurate at m = 512
    assert sol.cdf == pytest.approx(ref, abs=1e-12)
    # the gains do not matter when g = 0
    assert ss.solve(_random_gains(4, m), prob).cdf == pytest.approx(ref, abs=1e-12)


def test_uncoupled_step_at_the_uncoupled_exponent():
    """g = 0, eta -> 0: F jumps from 0 to 1 at s_u = log(1 - delta)/delta (alpha_s = 1)."""
    m, delta = 512, 0.3
    s_u = np.log(1 - delta) / delta
    for s, target in [(s_u - 0.3, 0.0), (s_u + 0.3, 1.0)]:
        sol = ss.solve(_const_gains(m), ss.discrete_problem(m, s, delta, 0.0, 1e-7))
        assert sol.cdf == pytest.approx(target, abs=1e-5)


EPS = [1e-1, 1e-2, 1e-3]


@pytest.mark.parametrize("g,d0,delta,s", [(0.7, 1.0, 0.5, -0.9), (0.7, 1.0, 0.5, -0.3), (0.7, 1.0, 1.5, -0.2),
                                          (0.9, 1.0, 0.2, -1.2), (2.0, 0.7, 0.3, -0.4), (2.0, 0.7, 1.0, 0.0)])
def test_constant_gain_solver_reproduces_circular_law(g, d0, delta, s):
    """Constant gains d_0(n) = d0: the full fixed-point solver, continued in eta, gives the exact circular-law F(s)
    of (1 - delta) I + delta d0 J (overlap of |mu| < g d0 and |mu - c_delta| < r_s).  With d0 = 1 and g < 1 this is
    the trivial fixed point: the solver reproduces the exact result of paper [app:Fexact]."""
    m = 64
    sols = ss.continue_in_eta(_const_gains(m, d0), lambda e: ss.discrete_problem(m, s, delta, g, delta * e), EPS,
                              tol=1e-8)
    assert all(x.converged for x in sols)
    exact = tfp.overlap_fraction(g * d0, delta, s)
    # finite eta = delta * 1e-3 and the m = 64 ring: observed errors <= 5e-4 (m = 256: same to 1e-4)
    assert sols[-1].cdf == pytest.approx(exact, abs=2e-3)
    # the eta sequence approaches the exact value
    assert abs(sols[-1].cdf - exact) < abs(sols[0].cdf - exact) + 1e-6
    if d0 == 1.0:
        # the translation-invariant single-site Fourier integral agrees as well (Gauss's law)
        assert tfp.single_site_cdf(g, delta, s) == pytest.approx(exact, abs=1e-6)


@pytest.mark.parametrize("s,theta", [(-1.5, 0.0), (-1.3, 0.25)])
def test_constant_gain_continuous_reproduces_semicircle(s, theta):
    """Continuous time, d_0 = 1, g = 0.8 < 1: the solver gives the semicircle law of radius g centered at -1."""
    m, T, g = 128, 40.0, 0.8
    sols = ss.continue_in_eta(_const_gains(m), lambda e: ss.continuous_problem(m, T, s, g, e, theta=theta),
                              EPS, tol=1e-8)
    assert all(x.converged for x in sols)
    assert sols[-1].meta["complex_ring"] is (theta != 0.0)
    # eps = 1e-3, grid spacing T/m = 0.31: observed errors <= 3e-4
    assert sols[-1].cdf == pytest.approx(tfp.semicircle_cdf(g, s), abs=2e-3)


def test_continuous_center_of_semicircle_is_one_half():
    """At s = -1 (a = 0) the twisted continuous readout gives F = 1/2 at every eta, by the symmetry of the
    semicircle about its center.  (theta = 0 is singular there: the uncoupled tail sum has a zero mode.)"""
    m, T = 64, 20.0
    sols = ss.continue_in_eta(_const_gains(m), lambda e: ss.continuous_problem(m, T, -1.0, 0.8, e, theta=0.25),
                              EPS, tol=1e-8)
    assert all(x.converged for x in sols)
    assert np.allclose([x.cdf for x in sols], 0.5, atol=1e-8)


def test_trace_identity_and_gauge_independence():
    d = _random_gains(16, 24)
    pr = ss.discrete_problem(24, -0.3, 0.5, 2.5, 0.05)
    a = ss.solve(d, pr, tol=1e-11, gauge_fix=True)
    b = ss.solve(d, pr, tol=1e-11, gauge_fix=False, max_iter=5000)
    assert a.converged and b.converged
    # Tr P = Tr V holds at the fixed point whether or not it is imposed
    assert a.trace_identity_error < 1e-9 and b.trace_identity_error < 1e-9
    assert abs(a.P.sum() - a.V.sum()) < 1e-9 * a.V.sum()
    assert a.cdf == pytest.approx(b.cdf, abs=1e-9) and np.allclose(a.P, b.P, rtol=1e-8)
    assert np.all(a.P > 0) and np.all(a.A > pr.eta)


def test_unit_step_kernels_are_time_diagonal():
    """delta = 1: L is the identity up to a phase (|L|^2 = 1), so A_cal and P are constant symbols."""
    d = _random_gains(32, 16, seed=3)
    sol = ss.solve(d, ss.discrete_problem(16, -0.2, 1.0, 3.0, 0.05), tol=1e-11)
    assert sol.converged
    assert np.ptp(sol.A) < 1e-8 and np.ptp(sol.P) < 1e-8


def test_discrete_readout_parts_sum_to_cdf():
    d = _random_gains(8, 32, seed=1)
    prob = ss.discrete_problem(32, -0.4, 0.25, 2.0, 0.02, theta=0.2)
    sol = ss.solve(d, prob, tol=1e-10)
    parts = ss.discrete_readout_parts(prob, sol.A, sol.V)
    assert parts["leak"] + parts["contact"] + parts["regulator"] == pytest.approx(sol.cdf, abs=1e-12)
    assert parts["regulator"] == pytest.approx(prob.eta * np.mean(sol.V), abs=1e-15)


@pytest.mark.parametrize("theta,cplx", [(0.0, False), (0.3, True), (0.5, True)])
def test_twisted_symbols_give_complex_ring(theta, cplx):
    d = _random_gains(4, 16, seed=2)
    sol = ss.solve(d, ss.discrete_problem(16, -0.3, 0.5, 2.0, 0.05, theta=theta), tol=1e-10)
    assert sol.converged and sol.meta["complex_ring"] is cplx
    assert sol.meta["theta"] == theta and np.all(np.isfinite(sol.A))


def test_continue_in_eta_warm_starts():
    d = _random_gains(8, 16, seed=4)
    make = lambda e: ss.discrete_problem(16, -0.3, 0.5, 2.0, e)
    sols = ss.continue_in_eta(d, make, [0.1, 0.01])
    assert [x.meta["eta"] for x in sols] == [0.1, 0.01]
    assert sols[0].meta["init"] == "uncoupled" and sols[1].meta["init"] == "continued"
    cold = ss.solve(d, make(0.01), tol=1e-8)
    assert cold.cdf == pytest.approx(sols[1].cdf, abs=1e-6)


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("n,max_block", [(23, 4), (40, 512)])
def test_blocked_triangular_inverse(cplx, n, max_block):
    """The block recursion used for n > max_block (to stay off MAGMA on the GPU) is exact."""
    gen = torch.Generator().manual_seed(0)
    A = torch.randn(3, n, n, generator=gen, dtype=torch.float64)
    if cplx:
        A = A + 1j * torch.randn(3, n, n, generator=gen, dtype=torch.float64)
    M = A @ A.conj().transpose(-1, -2) + n * torch.eye(n)
    L = torch.linalg.cholesky(M)
    Li = ss._tri_inv(L, max_block=max_block)
    eye = torch.eye(n, dtype=L.dtype).expand_as(L)
    assert torch.allclose(Li @ L, eye, atol=1e-12)
    assert torch.equal(Li, torch.tril(Li))
    assert torch.allclose(ss._pd_inv(M) @ M, eye, atol=1e-11)


def test_pd_inv_above_512_and_rejects_indefinite():
    """n = 600 > 512 exercises the default block split in _pd_inv; an indefinite matrix raises."""
    n = 600
    gen = torch.Generator().manual_seed(1)
    A = torch.randn(n, n, generator=gen, dtype=torch.float64)
    M = (A @ A.T / n + torch.eye(n))[None]
    assert torch.allclose(ss._pd_inv(M) @ M, torch.eye(n, dtype=torch.float64)[None], atol=1e-10)
    bad = torch.diag(torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64))[None]
    with pytest.raises(Exception):
        ss._pd_inv(bad)


def _unit_step_scalar_solution(d, g, s, eta, n_iter=20000):
    """delta = 1, time-diagonal kernels: per time step the single-site equations are scalar in the gain d,
        Q(d) = 1/(eta + gamma d^2 P + 1/A),  X(d) = 1/(A + 1/(eta + gamma d^2 P)),
        A = eta + gamma <d^2 Q>,  P = <X>,  V = <Q>,  F = V (eta + 1/A),   gamma = (g e^{-s})^2,
    solved here by damped iteration, independently of the matrix solver."""
    gam = (g * np.exp(-s)) ** 2
    P, A = 1.0, 1.0
    for _ in range(n_iter):
        Q = 1 / (eta + gam * d**2 * P + 1 / A)
        X = 1 / (A + 1 / (eta + gam * d**2 * P))
        A_new, P_new = eta + gam * np.mean(d**2 * Q), np.mean(X)
        if abs(A_new - A) + abs(P_new - P) < 1e-15:
            break
        A, P = 0.5 * (A + A_new), 0.5 * (P + P_new)
    V = np.mean(1 / (eta + gam * d**2 * P + 1 / A))
    return dict(A=A, P=P, V=V, F=V * (eta + 1 / A))


@pytest.mark.parametrize("s", [-1.0, -0.2, 0.2])
def test_unit_step_matrix_solver_equals_scalar_equations(s):
    """delta = 1, fluctuating gains, finite eta: the matrix fixed point equals the scalar equations evaluated on
    the same empirical gain law."""
    m, g, eta = 16, 3.0, 0.05
    d = _random_gains(32, m, seed=3)
    sol = ss.solve(d, ss.discrete_problem(m, s, 1.0, g, eta), tol=1e-12)
    ref = _unit_step_scalar_solution(d.numpy().ravel(), g, s, eta)
    assert sol.converged
    assert np.allclose(sol.A, ref["A"], rtol=1e-9) and np.allclose(sol.P, ref["P"], rtol=1e-9)
    assert sol.cdf == pytest.approx(ref["F"], abs=1e-10)


def test_continuous_problem_rejects_uncoupled_exponent_without_twist():
    with pytest.raises(ValueError, match="uncoupled exponent"):
        ss.continuous_problem(64, 20.0, -1.0, 0.8, 1e-2, theta=0.0)
    ss.continuous_problem(64, 20.0, -1.0, 0.8, 1e-2, theta=0.25)     # a twist regularizes it
