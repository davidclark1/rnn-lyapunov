"""Maximum Lyapunov exponent from the operator T_s (paper Appendix "Maximum Lyapunov exponent", [app:T]).

* ``delta = 1``: Molgedey-Schuchhardt-Schuster, ``s_* = log(g sqrt(C^d(0)))``.
* ``s_*`` on the DMFT lag window (``discrete_edge``) equals ``s_star`` of the scaled operator on the same ``C^d``
  profile; the bottom of the spectrum of ``a_s^{-1} T_s`` vanishes at ``s_*`` and changes sign there.
* Continuous time: ``s_* = -1 + sqrt(1 - E_0)`` from the Schrodinger ground state equals ``continuous_edge``.
* ``discrete_edge -> continuous_edge`` as ``delta -> 0``.
* Paper Fig. max_exponent_operator and the chaos criterion: ``C^x'(tau)`` is the zero-energy first excited state of
  ``H`` (``E_1 = 0``, ``E_0 < 0``, so ``s_* > 0``); at finite ``delta`` the first excited eigenvalue of
  ``alpha_0^{-1} T_0`` is ~2e-5 at ``delta = 0.5`` and ~1e-11 at ``delta = 0.25`` (g = 3).
"""
import numpy as np
import pytest

from rnn_lyapunov import dmft, max_exponent as mx


def test_unit_step_molgedey_formula(sol_unit, ga):
    g = 3.0
    cd0 = ga.one(lambda x: dmft._dphi(x) ** 2, sol_unit.q)                 # C^d(0) = E sech^4
    assert sol_unit.c_d[0] == pytest.approx(cd0, abs=1e-12)
    expected = np.log(g * np.sqrt(cd0))
    assert mx.discrete_edge(sol_unit) == pytest.approx(expected, abs=1e-12)
    assert mx.upper_edge(sol_unit) == pytest.approx(expected, abs=1e-12)
    _, cd = mx.cd_profile(sol_unit, dmft.mean_gain(sol_unit, ga) ** 2, 10.0)
    assert mx.s_star(1.0, g, cd) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("which,delta", [("sol_half", 0.5), ("sol_quarter", 0.25)])
def test_s_star_on_dmft_window_equals_discrete_edge(which, delta, request, ga):
    sol = request.getfixturevalue(which)
    edge = mx.discrete_edge(sol)
    assert mx.upper_edge(sol) == edge
    ri = dmft.mean_gain(sol, ga) ** 2
    tau, cd = mx.cd_profile(sol, ri, 0.0)                                    # L = 0: exactly the DMFT window
    assert len(cd) == 2 * len(sol.c_d) - 1 and np.allclose(tau[len(sol.c_d) - 1:], sol.lags)
    assert np.array_equal(cd[len(sol.c_d) - 1:], sol.c_d)
    s = mx.s_star(delta, 3.0, cd)
    assert s == pytest.approx(edge, abs=1e-11)                             # both brentq to xtol 1e-13
    # the ground state is localized: a longer window, continued by <phi'>^2, gives the same s_*
    _, cd_long = mx.cd_profile(sol, ri, 2 * sol.lags[-1])
    assert cd_long[0] == ri and len(cd_long) > len(cd)
    assert mx.s_star(delta, 3.0, cd_long) == pytest.approx(edge, abs=1e-10)
    # bottom of the spectrum of a_s^{-1} T_s: zero at s_*, negative below, positive above
    b = lambda x: mx.lowest(*mx.scaled_operator(delta, 3.0, x, cd)[:2])[0]
    assert abs(b(s)) < 1e-10
    assert b(s - 0.05) < 0 < b(s + 0.05)
    # a_s^{-1} T_s: diag a + 1/a - (c/a) C^d, off-diagonal -1, with a_s = (1-delta) e^{-delta s}, c_s = (delta g e^{-delta s})^2
    d, off, a, c = mx.scaled_operator(delta, 3.0, 0.1, cd)
    assert a == pytest.approx((1 - delta) * np.exp(-0.1 * delta))
    assert c == pytest.approx((delta * 3.0 * np.exp(-0.1 * delta)) ** 2)
    assert np.allclose(d, a + 1 / a - c / a * cd) and np.all(off == -1)


def test_continuous_edge_from_schrodinger_ground_state(sol_cont, ga):
    tau, cd = mx.cd_profile(sol_cont, dmft.mean_gain(sol_cont, ga) ** 2, 0.0)
    V, E, psi = mx.schrodinger(3.0, tau, cd, k=3)
    assert np.allclose(V, 1 - 9.0 * cd)
    assert mx.s_star_from_energy(E[0]) == pytest.approx(mx.continuous_edge(sol_cont), abs=1e-12)
    assert mx.upper_edge(sol_cont) == mx.continuous_edge(sol_cont)
    # eigenfunctions normalized to int psi^2 dtau = 1
    assert np.allclose((psi**2).sum(0) * sol_cont.dt, 1.0)
    # energy-to-exponent map E(s) = 1 - (1 + s)^2
    for s in (-0.5, 0.0, 0.3):
        assert mx.s_star_from_energy(1 - (1 + s) ** 2) == pytest.approx(s, abs=1e-14)


def test_discrete_edge_approaches_continuous_edge(sol_half, sol_quarter, sol_cont):
    ce = mx.continuous_edge(sol_cont)
    e1, e2 = mx.discrete_edge(sol_half), mx.discrete_edge(sol_quarter)
    assert e1 < e2 < ce                                                      # monotone approach from below (g = 3)
    # O(delta): linear (Richardson) extrapolation to delta = 0 removes most of the gap (0.019 -> < 2e-3)
    assert abs(2 * e2 - e1 - ce) < 2e-3
    assert abs(e1 - ce) < 0.03


def test_derivative_of_cx_is_the_zero_mode_continuous(sol_cont, ga):
    """delta -> 0: H C^x' = 0 (differentiate the DMFT equation and use Price's theorem); C^x' is odd with a
    single node, so it is the first excited state, E_1 = 0, and the ground state has E_0 < 0, i.e. s_* > 0."""
    ri = dmft.mean_gain(sol_cont, ga) ** 2
    tau, cd = mx.cd_profile(sol_cont, ri, 60.0)
    V, E, psi = mx.schrodinger(3.0, tau, cd, k=3)
    assert E[0] < 0 < mx.s_star_from_energy(E[0])
    # E_1 = 0 up to the O(dt^2) lag-grid error of the DMFT (observed 5e-8 at dt = 0.1); E_2 is order 0.1
    assert abs(E[1]) < 1e-6 and E[2] > 0.05
    assert E[1] < 1 - 9.0 * ri                                               # a bound state below the continuum
    cx = np.r_[sol_cont.c_x[:0:-1], sol_cont.c_x]
    tx = (np.arange(len(cx)) - (len(sol_cont.c_x) - 1)) * sol_cont.dt
    cxp = np.gradient(cx, sol_cont.dt)
    cxp /= np.sqrt(sol_cont.dt * (cxp**2).sum())
    overlap = abs(np.interp(tau, tx, cxp, left=0, right=0) @ psi[:, 1]) * sol_cont.dt
    assert overlap > 0.9999                                                  # observed 1 - 2e-7
    mid = len(tau) // 2
    assert abs(psi[mid, 1]) < 1e-10 * np.abs(psi[:, 1]).max()                 # node at tau = 0
    assert np.allclose(psi[:, 1], -psi[::-1, 1], atol=1e-8)                   # odd


@pytest.mark.parametrize("which,delta,lo,hi", [("sol_half", 0.5, 1e-5, 3e-5), ("sol_quarter", 0.25, 1e-12, 1e-10)])
def test_near_zero_mode_at_finite_step(which, delta, lo, hi, request, ga):
    """Paper: for g = 3 the first excited eigenvalue of alpha_0^{-1} T_0 is ~2e-5 at delta = 0.5 and of order
    1e-11 at delta = 0.25 (observed 1.73e-5 and 6.7e-12 on the window L = 160); the bottom is well below."""
    sol = request.getfixturevalue(which)
    _, cd = mx.cd_profile(sol, dmft.mean_gain(sol, ga) ** 2, 160.0)
    ev = mx.lowest(*mx.scaled_operator(delta, 3.0, 0.0, cd)[:2], k=2)
    assert lo < ev[1] < hi
    assert ev[0] < -0.01


def test_lowest_returns_vectors():
    d = np.array([2.0, 2.0, 2.0])
    off = -np.ones(2)
    E, V = mx.lowest(d, off, k=2, vectors=True)
    assert np.allclose(E, [2 - np.sqrt(2), 2.0]) and V.shape == (3, 2)


@pytest.mark.slow
@pytest.mark.parametrize("which,delta", [("sol_unit", 1.0), ("sol_half", 0.5)])
def test_network_maximum_exponent_equals_s_star(which, delta, request):
    """lambda_1 = s_*: leading exponent of one network with N = 1024 at g = 3 (finite-N and finite-time
    deviations ~0.01; tolerance 0.03)."""
    from rnn_lyapunov import lyapunov as ly, network as net
    sol = request.getfixturevalue(which)
    J = net.make_coupling(1024, 3.0, seed=11)
    res = ly.lyapunov_spectrum(J, delta=delta, k=2, t_burn=200, t_tangent_burn=50, t_obs=1000, seed=11)
    assert res.exponents[0] == pytest.approx(mx.upper_edge(sol), abs=0.03)
