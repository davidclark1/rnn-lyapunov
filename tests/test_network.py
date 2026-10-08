"""The network (paper [eq:x]): coupling statistics and determinism, and exactness of the tangent steps.

* ``J_ij ~ N(0, g^2/N)`` iid, reproducible from the seed alone.
* ``map_tangent_step`` applies ``M(n) = (1 - delta) I_N + delta J D(n)`` (paper [eq:Ms] at s = 0), the Jacobian of
  ``map_step`` at the old state; ``rk4_tangent_step`` applies the exact Jacobian of ``rk4_step``.
* ``dynamics()`` selects exactly one of the map and the flow; ``simulate`` returns the sampled trajectory.
"""
import numpy as np
import pytest
import torch

from rnn_lyapunov import network as net


def test_coupling_statistics():
    N, g = 600, 2.5
    J = net.make_coupling(N, g, seed=1)
    assert J.dtype == torch.float64 and J.shape == (N, N)
    # N^2 = 3.6e5 samples: the sample variance has relative s.e. sqrt(2/N^2) ~ 2.4e-3; 1.5% is ~6 s.e.
    assert abs(J.var().item() * N / g**2 - 1) < 0.015
    assert abs(J.mean().item()) < 5 * g / N**0.5 / N
    Jz = net.make_coupling(N, g, seed=1, zero_diagonal=True)
    assert torch.all(torch.diagonal(Jz) == 0)
    off = ~torch.eye(N, dtype=torch.bool)
    assert torch.equal(Jz[off], J[off])


def test_coupling_and_initial_state_are_deterministic():
    a, b = net.make_coupling(50, 3.0, seed=7), net.make_coupling(50, 3.0, seed=7)
    assert torch.equal(a, b)
    assert not torch.equal(a, net.make_coupling(50, 3.0, seed=8))
    # the gain only rescales the same normal draw
    assert torch.allclose(net.make_coupling(50, 1.0, seed=7) * 3.0, a, rtol=1e-15, atol=0)
    x1, x2 = net.initial_state(50, 1.0, 7), net.initial_state(50, 1.0, 7)
    assert torch.equal(x1, x2) and x1.dtype == torch.float64
    # x(0) is drawn independently of J for the same seed
    assert not torch.allclose(x1, a[0] * 50**0.5 / 3.0)


def test_gain_is_sech_squared_without_underflow():
    x = torch.tensor([0.0, 0.3, -2.0, 25.0], dtype=torch.float64)
    d = net.gain(x)
    assert torch.allclose(d[:3], 1 - torch.tanh(x[:3]) ** 2, rtol=1e-13)
    # 1 - tanh^2 underflows to exactly zero at |x| = 25; sech^2 = 1/cosh^2 does not
    assert (1 - torch.tanh(x[3]) ** 2).item() == 0.0 and d[3].item() > 0


def _fd_jacobian(step, x, J, h, eps=1e-6):
    cols = [(step(x + eps * e, J, h) - step(x - eps * e, J, h)) / (2 * eps)
            for e in torch.eye(len(x), dtype=torch.float64)]
    return torch.stack(cols, dim=1)


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0])
def test_map_tangent_step_is_jacobian_of_map_step(delta):
    N = 12
    J = net.make_coupling(N, 2.5, seed=0)
    x = net.initial_state(N, 1.5, seed=1)
    xn, M = net.map_tangent_step(x, torch.eye(N, dtype=torch.float64), J, delta)
    assert torch.equal(xn, net.map_step(x, J, delta))
    # closed form M(n) = (1 - delta) I + delta J diag(d(n))
    M_closed = (1 - delta) * torch.eye(N, dtype=torch.float64) + delta * J * net.gain(x)[None, :]
    assert torch.allclose(M, M_closed, atol=1e-14)
    # central differences with eps = 1e-6: truncation ~ eps^2 * |phi'''| ~ 1e-12, rounding ~ 1e-16/eps = 1e-10
    assert torch.allclose(M, _fd_jacobian(net.map_step, x, J, delta), atol=1e-8)


@pytest.mark.parametrize("dt", [0.05, 0.2])
def test_rk4_tangent_step_is_jacobian_of_rk4_step(dt):
    N = 10
    J = net.make_coupling(N, 3.0, seed=2)
    x = net.initial_state(N, 1.5, seed=3)
    xn, M = net.rk4_tangent_step(x, torch.eye(N, dtype=torch.float64), J, dt)
    assert torch.allclose(xn, net.rk4_step(x, J, dt), atol=1e-14)
    assert torch.allclose(M, _fd_jacobian(net.rk4_step, x, J, dt), atol=1e-8)


def test_tangent_step_is_linear_in_frame():
    N = 8
    J = net.make_coupling(N, 2.0, seed=4)
    x = net.initial_state(N, 1.0, seed=4)
    Q = torch.randn(N, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    for step in (lambda Q: net.map_tangent_step(x, Q, J, 0.3)[1], lambda Q: net.rk4_tangent_step(x, Q, J, 0.1)[1]):
        M = step(torch.eye(N, dtype=torch.float64))
        assert torch.allclose(step(Q), M @ Q, atol=1e-13)


def test_dynamics_argument_validation():
    with pytest.raises(ValueError):
        net.dynamics()
    with pytest.raises(ValueError):
        net.dynamics(delta=0.5, rk4_dt=0.05)
    d = net.dynamics(delta=0.5)
    assert d.scheme == "map" and d.step is net.map_step and d.tangent_step is net.map_tangent_step
    assert d.n_steps(10.0) == 20
    f = net.dynamics(rk4_dt=0.05)
    assert f.scheme == "flow" and f.step is net.rk4_step and f.tangent_step is net.rk4_tangent_step
    assert f.n_steps(1.0) == 20


def test_simulate_shape_and_consistency():
    N = 20
    J = net.make_coupling(N, 2.0, seed=5)
    X = net.simulate(J, delta=0.5, t=10.0, seed=5)
    assert X.shape == (20, N)
    Xs = net.simulate(J, delta=0.5, t=10.0, seed=5, sample_every=4)
    assert Xs.shape == (5, N) and torch.equal(Xs, X[3::4])
    # the trajectory is the iterated map from the default initial state
    x = net.initial_state(N, 1.0, 5)
    for _ in range(20):
        x = net.map_step(x, J, 0.5)
    assert torch.equal(X[-1], x)
    # explicit x0 and the RK4 flow
    x0 = torch.zeros(N, dtype=torch.float64)
    assert torch.all(net.simulate(J, rk4_dt=0.1, t=1.0, x0=x0) == 0)           # x = 0 is a fixed point
    assert net.simulate(J, rk4_dt=0.1, t=1.0, seed=1).shape == (10, N)
    assert np.isfinite(net.simulate(J, rk4_dt=0.1, t=1.0, seed=1).numpy()).all()
