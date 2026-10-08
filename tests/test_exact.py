"""Exact finite-N objects of Parts 1-2 (paper [eq:driven]-[eq:F], [eq:Veta], [eq:B], [eq:Bdef]).

* ``K`` encodes the driven shifted tangent dynamics ``v(n+1) = M_s(n) v(n) + I^v(n)``.
* ``||B_eta^{-1}||_op <= 1/eta`` (``B_eta`` is ``eta`` times the identity plus an antisymmetric matrix).
* The regularized solution ``(K^T K + eta^2)^{-1} K^T I^v`` is the forward field of ``B_eta (U, V) = (I^v, 0)``,
  and tends to the minimum-norm solution ``K^+ I^v`` as ``eta -> 0``.
* Constant non-normal step: ``R^vv(k+1, k)`` deep in the window is the projector onto the stable subspace along
  the unstable one, so its trace counts exponents below ``s``.
* Part 1 claim: for a small chaotic network, ``(1/N) tr R^vv(k+1, k)`` at the window center equals the fraction of
  Lyapunov exponents below ``s`` (from QR on the same matrices), for ``s`` away from the exponents.
"""
import numpy as np
import pytest

from rnn_lyapunov import exact as ex
from rnn_lyapunov import lyapunov as ly


def _random_steps(N=4, m=6, delta=0.4, g=1.5, seed=0):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(N, N)) * g / np.sqrt(N)
    return ex.tangent_matrices(J, rng.uniform(0.15, 1.0, (m, N)), delta), J


def _blk(M, i, j, N):
    return M[i * N:(i + 1) * N, j * N:(j + 1) * N]


def test_tangent_matrices_and_shift():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(4, 4))
    gains = rng.uniform(0.15, 1.0, (6, 4))
    Ms = ex.tangent_matrices(J, gains, 0.4)
    assert len(Ms) == 6
    assert np.allclose(Ms[1], 0.6 * np.eye(4) + 0.4 * J @ np.diag(gains[1]), atol=1e-15)
    Mss = ex.shifted(Ms, 0.3, 0.4)
    assert np.allclose(Mss[2], np.exp(-0.12) * Ms[2])
    # shifting by s lowers every exponent by s
    lam = ly.qr_exponents_from_matrices(Ms, 0.4)
    assert np.allclose(ly.qr_exponents_from_matrices(Mss, 0.4), lam - 0.3, atol=1e-12)


def test_residual_matrix_encodes_driven_dynamics():
    Ms, _ = _random_steps(seed=1)
    m, N = len(Ms), 4
    K = ex.residual_matrix(Ms)
    assert K.shape == (m * N, (m + 1) * N)
    rng = np.random.default_rng(2)
    V = rng.normal(size=(m + 1, N))
    KV = (K @ V.ravel()).reshape(m, N)
    for n in range(m):
        assert np.allclose(KV[n], V[n + 1] - Ms[n] @ V[n], atol=1e-14)


@pytest.mark.parametrize("eta", [1.0, 0.1, 0.01])
def test_forward_backward_norm_bound(eta):
    Ms, _ = _random_steps(seed=3)
    K = ex.residual_matrix(ex.shifted(Ms, -0.2, 0.4))
    B = ex.forward_backward_matrix(K, eta)
    r, c = K.shape
    assert B.shape == (r + c, r + c)
    assert np.allclose(B + B.T, 2 * eta * np.eye(r + c))          # eta I + antisymmetric
    assert np.linalg.norm(np.linalg.inv(B), 2) <= 1 / eta * (1 + 1e-12)


@pytest.mark.parametrize("eta", [0.3, 0.02])
def test_regularized_solution_is_forward_field(eta):
    Ms, _ = _random_steps(seed=4)
    K = ex.residual_matrix(ex.shifted(Ms, 0.1, 0.4))
    r, c = K.shape
    Iv = np.random.default_rng(5).normal(size=r)
    UV = np.linalg.solve(ex.forward_backward_matrix(K, eta), np.r_[Iv, np.zeros(c)])
    R = ex.response_vv(K, eta)
    assert R.shape == (c, r)
    assert np.allclose(UV[r:], R @ Iv, atol=1e-11)
    # V_eta minimizes ||K V - I^v||^2 + eta^2 ||V||^2: the gradient vanishes
    V = R @ Iv
    assert np.allclose(K.T @ (K @ V - Iv) + eta**2 * V, 0, atol=1e-11)


def test_regularized_response_tends_to_pseudo_inverse():
    Ms, _ = _random_steps(seed=6)
    K = ex.residual_matrix(ex.shifted(Ms, -0.1, 0.4))
    P = ex.response_vv(K, 0.0)
    assert np.allclose(P, np.linalg.pinv(K))
    errs = [np.abs(ex.response_vv(K, eta) - P).max() for eta in (1e-1, 1e-2, 1e-3)]
    # K has full row rank, so the regularized response differs from K^+ by O(eta^2)
    assert errs[0] > errs[1] > errs[2] and errs[1] / errs[2] > 50 and errs[2] < 1e-5


def test_constant_step_response_is_stable_projector():
    """Constant non-normal M with two growing and two decaying directions: deep inside the window,
    R^vv(k+1, k) = P_stable and R^vv(k, k) = -M^{-1} (I - P_stable); a slice-wise rotation keeps the trace."""
    rng = np.random.default_rng(0)
    V = rng.normal(size=(4, 4))
    M = V @ np.diag([1.8, 1.3, 0.6, 0.3]) @ np.linalg.inv(V)
    Ps = V @ np.diag([0, 0, 1.0, 1.0]) @ np.linalg.inv(V)
    m, N, k = 60, 4, 30
    R = ex.response_vv(ex.residual_matrix([M] * m))
    assert np.allclose(_blk(R, k + 1, k, N), Ps, atol=1e-8)
    assert np.allclose(_blk(R, k, k, N), -np.linalg.inv(M) @ (np.eye(N) - Ps), atol=1e-8)
    # boundary corrections decay geometrically with the distance k to the window ends (observed 4e-8)
    assert ex.cdf_from_response([M] * m) == pytest.approx(0.5, abs=1e-6)
    O = [np.linalg.qr(rng.normal(size=(N, N)))[0] for _ in range(m + 1)]
    rot = [O[n + 1] @ M @ O[n].T for n in range(m)]
    assert ex.cdf_from_response(rot) == pytest.approx(0.5, abs=1e-6)
    # shifting s across the exponent log 1.3 (or log 0.6) changes the count by one; thresholds midway between
    # exponents (gaps ~0.16 and ~0.35) and a longer window keep the boundary corrections below 1e-6
    assert ex.cdf_from_response(ex.shifted([M] * 200, 0.425, 1.0)) == pytest.approx(0.75, abs=1e-6)
    assert ex.cdf_from_response(ex.shifted([M] * 200, -0.86, 1.0)) == pytest.approx(0.25, abs=1e-6)


# A small chaotic network: N = 10, g = 6, seed 5, delta = 0.5 has exponents (window m = 200)
# 0.31, 0.04, -0.23, -0.52, -0.90, -1.33, -1.44, -1.60, -2.29, -3.34.  Thresholds lie in gaps between them.
N_CH, G_CH, SEED_CH, D_CH, M_CH = 10, 6.0, 5, 0.5, 200


@pytest.fixture(scope="module")
def chaotic_steps():
    J = np.random.default_rng(SEED_CH).normal(size=(N_CH, N_CH)) * G_CH / np.sqrt(N_CH)
    gains = ex.simulate_gains(J, D_CH, M_CH, burn=2000, seed=SEED_CH)
    Ms = ex.tangent_matrices(J, gains, D_CH)
    return Ms, ly.qr_exponents_from_matrices(Ms, D_CH)


def test_part1_trace_counts_exponents_fast(chaotic_steps):
    Ms, lam = chaotic_steps
    assert lam.max() > 0.1                                                   # chaotic
    for s in (0.19, -0.73):
        F = ex.cdf_from_response(ex.shifted(Ms, s, D_CH), eta=1e-6)
        # exact count up to boundary effects ~ exp(-gap * m delta / 2) and the O(eta) regulator (observed 1e-6)
        assert F == pytest.approx(np.mean(lam < s), abs=1e-4)


@pytest.mark.slow
def test_part1_trace_counts_exponents_min_norm(chaotic_steps):
    """The same with the minimum-norm solution (eta = 0, pseudo-inverse) at five thresholds."""
    Ms, lam = chaotic_steps
    for s in (0.19, -0.06, -0.36, -0.73, -1.9):
        F = ex.cdf_from_response(ex.shifted(Ms, s, D_CH))
        assert F == pytest.approx(np.mean(lam < s), abs=1e-4)
