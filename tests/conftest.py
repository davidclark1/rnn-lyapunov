"""Shared fixtures and settings for the rnn_lyapunov test suite.

Everything runs on the CPU unless a test is marked ``gpu``.  The DMFT disk cache is redirected to a per-session
temporary directory, so no test reads or writes the user's cache.

CPU threading: torch's OpenBLAS backend, multi-threaded, is pathologically slow for the small batched triangular
solves of the single-site solver (a 64 x 64 ``solve_triangular`` takes ~20 ms instead of ~30 us on a 32-core
allocation).  ``OPENBLAS_NUM_THREADS=1`` (set before torch is imported, unless already set) removes this, and the
intra-op thread pool is capped for the many tiny tensor operations.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

from rnn_lyapunov import dmft  # noqa: E402

CPU = "cpu"
torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory):
    """Point ``RNN_LYAPUNOV_CACHE`` at a temporary directory for the whole session."""
    old = os.environ.get("RNN_LYAPUNOV_CACHE")
    os.environ["RNN_LYAPUNOV_CACHE"] = str(tmp_path_factory.mktemp("rnn_lyapunov_cache"))
    yield
    if old is None:
        os.environ.pop("RNN_LYAPUNOV_CACHE", None)
    else:
        os.environ["RNN_LYAPUNOV_CACHE"] = old


@pytest.fixture(scope="session")
def ga():
    """CPU Gaussian-average quadrature with step dz = 0.08 (the library default is 0.04).  The trapezoid rule
    converges exponentially: at g = 3 the DMFT variance and upper edge agree with dz = 0.04 to ~1e-13, at ~5x
    lower cost."""
    return dmft.GaussianAverages(device=CPU, dz=0.08)


# Shared DMFT solutions at g = 3.  The map uses the paper's lag window t_max = 80 (C^x(80) ~ 1e-12 q); the
# continuous-time solution uses a short window t_max = 30 (C^x(30) ~ 1e-5 q) and a coarse lag grid dt = 0.1 to
# keep the fast suite fast on a CPU.

@pytest.fixture(scope="session")
def sol_unit(ga):
    return dmft.solve(3.0, 1.0, ga=ga)


@pytest.fixture(scope="session")
def sol_half(ga):
    return dmft.solve(3.0, 0.5, t_max=80.0, ga=ga)


@pytest.fixture(scope="session")
def sol_quarter(ga):
    return dmft.solve(3.0, 0.25, t_max=80.0, ga=ga)


@pytest.fixture(scope="session")
def sol_cont(ga):
    return dmft.solve(3.0, None, t_max=30.0, dt=0.1, ga=ga)


def pytest_collection_modifyitems(config, items):
    """Skip ``gpu`` tests cleanly when no CUDA device is visible."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

