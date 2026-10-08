"""GPU paths (marker ``gpu``; skipped without CUDA): device independence of couplings and of the theory pipeline.

Run with ``CUDA_VISIBLE_DEVICES=<one assigned GPU> python -m pytest -q -m gpu``.
"""
import functools

import numpy as np
import pytest
import torch

from rnn_lyapunov import dmft, lyapunov as ly, network as net, theory

pytestmark = pytest.mark.gpu
GPU = "cuda:0"


def test_coupling_is_device_independent():
    a = net.make_coupling(300, 3.0, seed=7, device="cpu")
    b = net.make_coupling(300, 3.0, seed=7, device=GPU)
    assert b.device.type == "cuda" and torch.equal(a, b.cpu())
    assert torch.equal(net.initial_state(300, 1.0, 7, "cpu"), net.initial_state(300, 1.0, 7, GPU).cpu())


def test_lyapunov_spectrum_gpu_equals_cpu():
    J = net.make_coupling(32, 3.0, seed=1)
    kw = dict(delta=0.5, t_burn=20, t_tangent_burn=10, t_obs=40, seed=1)
    a = ly.lyapunov_spectrum(J, **kw).exponents
    b = ly.lyapunov_spectrum(J.to(GPU), **kw).exponents
    # same float64 arithmetic up to reduction order; chaos amplifies rounding only over t_obs = 40
    assert np.allclose(a, b, atol=1e-6)


def test_dmft_gpu_equals_cpu():
    a = dmft.solve(3.0, 0.5, t_max=40.0, ga=dmft.GaussianAverages(device="cpu", dz=0.08))
    b = dmft.solve(3.0, 0.5, t_max=40.0, ga=dmft.GaussianAverages(device=GPU, dz=0.08))
    assert np.allclose(a.c_x, b.c_x, atol=1e-12) and np.allclose(a.c_d, b.c_d, atol=1e-12)


def test_small_cdf_curve_gpu_equals_cpu(monkeypatch):
    monkeypatch.setattr(theory.dmft, "GaussianAverages", functools.partial(dmft.GaussianAverages, dz=0.08))
    cfg = theory.TheoryConfig(g=3.0, delta=0.5, T=8.0, n_paths=16, eps=(1e-1, 1e-2), n_twist=2, dmft_tmax=40.0,
                              tol=1e-10)
    s = [-1.0, -0.5, 0.05]
    a = theory.cdf_curve(cfg, s, device="cpu", verbose=False)
    b = theory.cdf_curve(cfg, s, device=GPU, verbose=False)
    assert a["all_converged"] and b["all_converged"]
    # identical Sobol paths and float64 linear algebra; differences come only from reduction order and the
    # fixed-point tolerance 1e-10
    assert np.allclose(a["F"], b["F"], atol=1e-10)
    assert a["edge"] == pytest.approx(b["edge"], abs=1e-12)
