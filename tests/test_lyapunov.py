"""Lyapunov spectrum by the QR method: agreement with a NumPy reference and exact volume identities.

* The torch QR run equals the NumPy reference ``qr_exponents_from_matrices`` on the same tangent matrices.
* For a constant matrix, the exponents are ``log|eigenvalues| / step``.
* The sum of the exponents is the time average of ``log|det M(n)|`` per unit time (exact, any N), with
  ``det M(n) = det((1-delta) I_N + delta J D(n))``; at ``delta = 1``, ``(1/N) sum lambda = (1/N) log|det J| + <log d>``.
* Continuous time with ``J_ii = 0``: ``tr(-I_N + J D) = -N``, so the mean exponent is ``-1``.
* Per-block logs reproduce the exponents; the leading-``k`` run agrees with the full run.
* The map at small ``delta`` approaches the flow.
"""
import numpy as np
import pytest
import torch

from rnn_lyapunov import lyapunov as ly
from rnn_lyapunov import network as net
from rnn_lyapunov.observables import spectrum_summary


def _trajectory_matrices(J, delta, n_burn, m, seed):
    """States and Jacobians M(n) = (1 - delta) I + delta J D(n) along the map (the same sequence the torch run uses)."""
    N = J.shape[0]
    x = net.initial_state(N, 1.0, seed)
    for _ in range(n_burn):
        x = net.map_step(x, J, delta)
    Ms = []
    for _ in range(m):
        _, M = net.map_tangent_step(x, torch.eye(N, dtype=torch.float64), J, delta)
        Ms.append(M.numpy())
        x = net.map_step(x, J, delta)
    return Ms


def test_torch_qr_equals_numpy_reference():
    N, delta, seed = 10, 0.5, 3
    J = net.make_coupling(N, 3.0, seed=seed)
    res = ly.lyapunov_spectrum(J, delta=delta, t_burn=20.0, t_tangent_burn=0.0, t_obs=100.0, seed=seed)
    Ms = _trajectory_matrices(J, delta, n_burn=40, m=200, seed=seed)
    # (i) started from the same initial frame, NumPy QR on the same matrices reproduces the torch run to rounding
    gen = torch.Generator().manual_seed(seed + 104729)
    Q = torch.linalg.qr(torch.randn(N, N, generator=gen, dtype=torch.float64))[0].numpy()
    acc = np.zeros(N)
    for M in Ms:
        Q, R = np.linalg.qr(M @ Q)
        acc += np.log(np.abs(np.diag(R)))
    assert np.allclose(acc / (len(Ms) * delta), res.exponents, atol=1e-12)
    # (ii) the reference function starts from a different random frame: agreement up to the frame transient,
    # which decays like exp(-gap * t); nearly degenerate pairs (gap ~ 1e-2) leave ~0.03 at t_obs = 100
    ref = ly.qr_exponents_from_matrices(Ms, delta)
    assert np.allclose(res.exponents, ref, atol=0.06)
    # the SUM of all N exponents is frame-independent: exact up to rounding
    assert abs(res.exponents.sum() - ref.sum()) < 1e-9
    # and equals the exact log-det identity
    logdet = sum(np.linalg.slogdet(M)[1] for M in Ms) / (len(Ms) * delta)
    assert abs(ref.sum() - logdet) < 1e-9


def test_qr_reference_constant_matrix_and_qr_every():
    rng = np.random.default_rng(0)
    eig = np.array([1.5, 0.9, 0.5, 0.2, 0.05])
    A = np.diag(eig) + np.triu(rng.normal(size=(5, 5)), 1)
    lam = ly.qr_exponents_from_matrices([A] * 600, step_size=0.5)
    # convergence is O(1/(m h)) from the frame transient: 0.02 at m h = 300
    assert np.allclose(lam, np.log(eig) / 0.5, atol=0.02)
    # re-orthonormalizing less often gives the same result (well conditioned for 3-step products)
    assert np.allclose(ly.qr_exponents_from_matrices([A] * 600, 0.5, qr_every=3), lam, atol=1e-10)
    # leading-k subspace (a different random initial frame: same transient tolerance)
    assert np.allclose(ly.qr_exponents_from_matrices([A] * 600, 0.5, k=2), np.log(eig[:2]) / 0.5, atol=0.02)


def test_sum_of_exponents_equals_mean_log_det():
    """sum_i lambda_i = (1/(m delta)) sum_n log|det M(n)| for an arbitrary gain sequence (exact)."""
    rng = np.random.default_rng(1)
    N, m, delta = 6, 50, 0.3
    J = rng.normal(size=(N, N)) * 2.0 / np.sqrt(N)
    Ms = [(1 - delta) * np.eye(N) + delta * J * rng.uniform(0.1, 1.0, N)[None, :] for _ in range(m)]
    lam = ly.qr_exponents_from_matrices(Ms, delta, qr_every=2)
    assert abs(lam.sum() - sum(np.linalg.slogdet(M)[1] for M in Ms) / (m * delta)) < 1e-11


def test_unit_step_mean_exponent_identity():
    """delta = 1: M(n) = J D(n), so (1/N) sum lambda = (1/N) log|det J| + <log d> exactly at finite N."""
    N, g = 30, 3.0
    J = net.make_coupling(N, g, seed=3)
    res = ly.lyapunov_spectrum(J, delta=1.0, t_burn=100, t_tangent_burn=10, t_obs=300, seed=3)
    expected = torch.linalg.slogdet(J)[1].item() / N + res.stats["mean_log_gain"]
    assert abs(res.exponents.mean() - expected) < 1e-9


def test_flow_volume_contraction_zero_diagonal():
    """Continuous time, J_ii = 0: div = tr(-I + J D) = -N, so the mean exponent is -1 (up to RK4 error)."""
    N = 16
    J = net.make_coupling(N, 3.0, seed=5, zero_diagonal=True)
    res = ly.lyapunov_spectrum(J, rk4_dt=0.05, t_burn=20, t_tangent_burn=5, t_obs=40, qr_every=4, seed=5)
    # RK4 local error O(dt^5) in the Jacobian determinant: observed ~1e-7
    assert abs(res.exponents.mean() + 1.0) < 1e-5


def test_block_logs_and_leading_subspace():
    N, delta = 24, 0.5
    J = net.make_coupling(N, 3.0, seed=8)
    kw = dict(delta=delta, t_burn=50, t_tangent_burn=50, t_obs=200, seed=8)
    full = ly.lyapunov_spectrum(J, **kw)
    n = full.block_logs.shape[0]
    assert full.block_logs.shape == (400, N) and full.block_time == delta
    # block logs are stored in float32: relative rounding 6e-8 per block, averaged
    assert np.allclose(full.exponents_from_blocks(slice(None)), full.exponents, atol=1e-6)
    halves = 0.5 * (full.exponents_from_blocks(slice(0, n // 2)) + full.exponents_from_blocks(slice(n // 2, n)))
    assert np.allclose(halves, full.exponents, atol=1e-6)
    assert np.all(np.diff(full.exponents) < 0.05)        # QR order is descending up to noise
    part = ly.lyapunov_spectrum(J, k=5, **kw)
    # identical trajectory; leading-k QR equals the leading block of the full QR up to the frame transient
    assert np.allclose(full.exponents[:5], part.exponents, atol=0.02)
    s = spectrum_summary(part.exponents, N=N)
    assert not s["full_spectrum"] and np.isnan(s["mean_exponent"])
    assert part.params["k"] == 5 and part.params["scheme"] == "map" and not part.params["full_spectrum"]


def test_validation():
    J = net.make_coupling(4, 2.0, seed=0)
    with pytest.raises(ValueError):
        ly.lyapunov_spectrum(J, t_obs=1.0)
    with pytest.raises(ValueError):
        ly.lyapunov_spectrum(J, delta=0.5, rk4_dt=0.05, t_obs=1.0)


def test_small_step_map_approaches_flow():
    """The map at delta = 0.02 and the flow (RK4) give nearly the same leading exponents (O(delta) difference)."""
    # small networks are often multistable or non-chaotic; N = 128, seed 2 is chaotic under both dynamics
    N, g, seed = 128, 3.0, 2
    J = net.make_coupling(N, g, seed=seed)
    kw = dict(k=3, t_burn=50, t_tangent_burn=20, t_obs=300, seed=seed, qr_every=5)
    mp = ly.lyapunov_spectrum(J, delta=0.02, **kw).exponents
    fl = ly.lyapunov_spectrum(J, rk4_dt=0.05, **kw).exponents
    # O(delta) systematic difference plus finite-time noise of two independent attractor samples
    # (observed differences <= 0.02); 0.05 is a loose bound
    assert np.allclose(mp, fl, atol=0.05)
    assert mp[0] > 0.03 and fl[0] > 0.03                  # chaotic
