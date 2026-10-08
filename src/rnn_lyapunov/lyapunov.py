"""Lyapunov spectrum of a simulated network by the QR method (Benettin; discrete QR).

An orthonormal frame ``Q`` (``N x k``) is advanced with the exact tangent step and re-orthonormalized every
``qr_every`` updates; the exponents are the time averages of ``log|diag R|``.  ``k < N`` gives the leading ``k``
exponents (enough for D_KY and h_KS when ``k`` exceeds the Kaplan-Yorke index).  The per-block logs are kept, so
that observation-time variability can be estimated afterwards (split halves, batch means) without rerunning.

Exponents are per unit time (divided by the physical time ``n_updates * step_size``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import network as net

Tensor = torch.Tensor


@dataclass
class LyapunovResult:
    exponents: np.ndarray            # (k,) per unit time, in QR order (descending up to noise)
    block_logs: np.ndarray           # (n_blocks, k) float32: log|diag R| of each QR block
    block_time: float                # physical time per QR block
    stats: dict = field(default_factory=dict)    # time and neuron averages of the state; final state
    params: dict = field(default_factory=dict)   # every numerical setting of the run

    def exponents_from_blocks(self, blocks: slice) -> np.ndarray:
        """Exponents from a subset of the QR blocks (e.g. ``slice(0, n // 2)`` for the first half)."""
        return self.block_logs[blocks].astype(np.float64).mean(0) / self.block_time


@torch.no_grad()
def lyapunov_spectrum(J: Tensor, *, delta: float | None = None, rk4_dt: float | None = None, k: int | None = None,
                      t_burn: float = 200.0, t_tangent_burn: float = 100.0, t_obs: float = 1000.0,
                      qr_every: int = 1, seed: int = 0, x0: Tensor | None = None,
                      progress: bool = False) -> LyapunovResult:
    """Leading ``k`` Lyapunov exponents (default: all ``N``) of the network with couplings ``J``.

    ``delta``: the Euler map at time step ``delta``.  ``rk4_dt``: the continuous-time network, integrated with RK4
    at step ``rk4_dt`` (exact RK4 tangent).  Give exactly one.

    Times are physical: the state is burned in for ``t_burn``, then the frame for ``t_tangent_burn``, then the
    exponents are averaged over ``t_obs``.  ``seed`` sets the initial state and the initial frame.
    """
    dyn = net.dynamics(delta=delta, rk4_dt=rk4_dt)
    dt = dyn.step_size
    N = J.shape[0]
    k = N if k is None else k
    dev = J.device
    start = time.monotonic()

    x = net.initial_state(N, 1.0, seed, dev) if x0 is None else x0.clone()
    for _ in range(dyn.n_steps(t_burn)):
        x = dyn.step(x, J, dt)

    gen = torch.Generator().manual_seed(seed + 104729)
    Q = torch.linalg.qr(torch.randn(N, k, generator=gen, dtype=torch.float64).to(dev))[0]

    block_time = qr_every * dt
    n_warm = round(t_tangent_burn / block_time)
    n_blocks = round(t_obs / block_time)
    logs = torch.empty(n_blocks, k, dtype=torch.float32, device=dev)
    acc = torch.zeros(k, dtype=torch.float64, device=dev)
    sums = torch.zeros(4, dtype=torch.float64, device=dev)       # <d>, <d^2>, <log d>, <x^2>
    n_stat = 0
    for b in range(n_warm + n_blocks):
        for _ in range(qr_every):
            if b >= n_warm:
                d = net.gain(x)
                sums += torch.stack([d.mean(), (d * d).mean(), torch.log(d).mean(), (x * x).mean()])
                n_stat += 1
            x, Q = dyn.tangent_step(x, Q, J, dt)
        Q, R = torch.linalg.qr(Q)
        if b >= n_warm:
            lr = torch.log(torch.abs(torch.diagonal(R)))
            acc += lr
            logs[b - n_warm] = lr.to(torch.float32)
        if progress and (b + 1) % max(1, (n_warm + n_blocks) // 10) == 0:
            print(f"  block {b + 1}/{n_warm + n_blocks}  {time.monotonic() - start:.0f}s", flush=True)

    sums = (sums / max(n_stat, 1)).cpu().numpy()
    return LyapunovResult(
        exponents=(acc / (n_blocks * block_time)).cpu().numpy(),
        block_logs=logs.cpu().numpy(),
        block_time=block_time,
        stats=dict(mean_gain=float(sums[0]), mean_sq_gain=float(sums[1]), mean_log_gain=float(sums[2]),
                   variance=float(sums[3]), x_final=x.cpu().numpy()),
        params=dict(N=N, k=k, scheme=dyn.scheme, step=dt, t_burn=t_burn, t_tangent_burn=t_tangent_burn,
                    t_obs=t_obs, qr_every=qr_every, seed=seed, full_spectrum=(k == N),
                    seconds=time.monotonic() - start, device=str(dev)),
    )


def qr_exponents_from_matrices(matrices, step_size: float, k: int | None = None, qr_every: int = 1,
                               seed: int = 0) -> np.ndarray:
    """Reference NumPy QR method for an explicit sequence of tangent matrices ``M(0), M(1), ...`` (for tests).

    Exponents are per unit time: the sum of ``log|diag R|`` divided by ``len(matrices) * step_size``.
    """
    N = matrices[0].shape[0]
    k = N if k is None else k
    Q = np.linalg.qr(np.random.default_rng(seed).normal(size=(N, k)))[0]
    acc = np.zeros(k)
    for n, M in enumerate(matrices):
        Q = M @ Q
        if (n + 1) % qr_every == 0 or n + 1 == len(matrices):
            Q, R = np.linalg.qr(Q)
            acc += np.log(np.abs(np.diag(R)))
    return acc / (len(matrices) * step_size)
