"""The network: couplings, the Euler map at time step ``delta``, the continuous-time flow (RK4), tangent steps.

Euler map (paper [eq:x]) and its Jacobian M(n) (paper [eq:Ms] at s = 0)::

    x(n+1) = (1 - delta) x(n) + delta J phi(x(n))
    M(n)   = (1 - delta) I_N + delta J D(n),        D(n) = diag(d(n)),   d_i(n) = phi'(x_i(n))

Continuous time (``delta -> 0``)::

    dx/dt = -x + J phi(x),      dQ/dt = (-I_N + J D(t)) Q

integrated with classical RK4 at step ``dt``.  The tangent step is the exact derivative of the RK4 state update,
obtained by applying the same RK4 stages to the joint system ``(x, Q)``.

Everything is float64; ``phi = tanh`` and ``phi' = sech^2``, evaluated as ``1/cosh^2`` (``1 - tanh^2`` cancels
catastrophically and underflows to exactly 0 for ``|x| > 19``, which would make the tangent step singular).

Use :func:`dynamics` to select the map (``delta=...``) or the flow (``rk4_dt=...``) in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

Tensor = torch.Tensor


def make_coupling(N: int, g: float, seed: int, *, zero_diagonal: bool = False,
                  device: str | torch.device = "cpu") -> Tensor:
    """``J_ij ~ N(0, g^2/N)`` iid, drawn from a CPU generator seeded with ``seed`` (identical on every device)."""
    gen = torch.Generator().manual_seed(seed)
    J = torch.randn(N, N, generator=gen, dtype=torch.float64) * (g / N**0.5)
    if zero_diagonal:
        J.fill_diagonal_(0.0)
    return J.to(device)


def initial_state(N: int, scale: float, seed: int, device: str | torch.device = "cpu") -> Tensor:
    """``x(0) ~ N(0, scale^2)`` iid, from a CPU generator seeded with ``seed + 7919`` (independent of ``J``)."""
    gen = torch.Generator().manual_seed(seed + 7919)
    return (torch.randn(N, generator=gen, dtype=torch.float64) * scale).to(device)


def gain(x: Tensor) -> Tensor:
    """``d = phi'(x) = sech^2(x)``, computed as ``1/cosh^2(x)``."""
    return 1.0 / torch.cosh(x) ** 2


# ---- Euler map ----------------------------------------------------------------------------------------------

def map_step(x: Tensor, J: Tensor, delta: float) -> Tensor:
    """``x(n+1) = (1 - delta) x(n) + delta J tanh(x(n))``."""
    return (1.0 - delta) * x + delta * (J @ torch.tanh(x))


def map_tangent_step(x: Tensor, Q: Tensor, J: Tensor, delta: float) -> tuple[Tensor, Tensor]:
    """Advance the state and the tangent vectors (columns of ``Q``) together: ``Q <- M(n) Q``, with the gains
    evaluated at the old state ``x(n)``."""
    p = torch.tanh(x)
    d = gain(x)
    Qn = (1.0 - delta) * Q + delta * (J @ (d[:, None] * Q))
    xn = (1.0 - delta) * x + delta * (J @ p)
    return xn, Qn


# ---- continuous-time flow, RK4 --------------------------------------------------------------------------------

def _joint_rhs(x: Tensor, Q: Tensor, J: Tensor) -> tuple[Tensor, Tensor]:
    p = torch.tanh(x)
    d = gain(x)
    return J @ p - x, J @ (d[:, None] * Q) - Q


def rk4_step(x: Tensor, J: Tensor, dt: float) -> Tensor:
    """One classical RK4 step of ``dx/dt = -x + J tanh(x)``."""
    f = lambda y: J @ torch.tanh(y) - y
    k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_tangent_step(x: Tensor, Q: Tensor, J: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """RK4 on the joint (state, variational) system: the exact Jacobian of the RK4 state update applied to ``Q``."""
    k1x, k1q = _joint_rhs(x, Q, J)
    k2x, k2q = _joint_rhs(x + 0.5 * dt * k1x, Q + 0.5 * dt * k1q, J)
    k3x, k3q = _joint_rhs(x + 0.5 * dt * k2x, Q + 0.5 * dt * k2q, J)
    k4x, k4q = _joint_rhs(x + dt * k3x, Q + dt * k3q, J)
    return (x + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x),
            Q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q))


# ---- one switch for map vs flow ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Dynamics:
    """A discrete-time update of the network: the Euler map at step ``delta`` or RK4 for the flow at step ``dt``.

    ``step_size`` is the physical time per update, so ``round(t / step_size)`` updates cover a time ``t``.
    """
    scheme: str                                          # "map" or "flow"
    step_size: float
    step: Callable[[Tensor, Tensor, float], Tensor]
    tangent_step: Callable[[Tensor, Tensor, Tensor, float], tuple[Tensor, Tensor]]

    def n_steps(self, t: float) -> int:
        return round(t / self.step_size)


def dynamics(*, delta: float | None = None, rk4_dt: float | None = None) -> Dynamics:
    """``dynamics(delta=0.5)``: the Euler map.  ``dynamics(rk4_dt=0.05)``: the continuous-time flow (delta -> 0),
    integrated with RK4 at step ``rk4_dt``.  Give exactly one of the two."""
    if (delta is None) == (rk4_dt is None):
        raise ValueError("give exactly one of delta (Euler map) or rk4_dt (continuous-time flow, RK4)")
    if delta is not None:
        return Dynamics("map", float(delta), map_step, map_tangent_step)
    return Dynamics("flow", float(rk4_dt), rk4_step, rk4_tangent_step)


@torch.no_grad()
def simulate(J: Tensor, *, delta: float | None = None, rk4_dt: float | None = None, t: float,
             x0: Tensor | None = None, seed: int = 0, sample_every: int = 1) -> Tensor:
    """Trajectory of length ``t`` (time units) from ``x0`` (default: :func:`initial_state` with ``seed``).

    Returns the states after every ``sample_every`` updates, shape ``(n_samples, N)``.
    """
    dyn = dynamics(delta=delta, rk4_dt=rk4_dt)
    x = initial_state(J.shape[0], 1.0, seed, J.device) if x0 is None else x0.clone()
    out = []
    for k in range(1, dyn.n_steps(t) + 1):
        x = dyn.step(x, J, dyn.step_size)
        if k % sample_every == 0:
            out.append(x)
    return torch.stack(out) if out else x[None, :0]
