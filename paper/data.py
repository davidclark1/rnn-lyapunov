"""Loaders for the paper's result files: the bundled dataset ``data/`` or any results directory with the same layout.

Layout (each file in the output format of the script that writes it)::

    theory/*.json                    scripts/run_theory.py          single-site F(s), one Sobol scramble per file
    theory_checks/*.json             scripts/run_theory.py          delta -> 0 grid-spacing study at g = 20
    network/*.json                   scripts/run_network.py         Lyapunov exponents of one simulated network
    participation/*.json             scripts/run_participation.py   PR^x/N, PR^phi/N of one simulated network
    participation_theory/*.json      scripts/run_pr_theory.py       PR theory on a grid of g, one file per delta
    max_exponent_operator.json       scripts/run_max_exponent.py    operator T_s and s_*(g)

Every loader takes ``root=`` (default: ``<repo>/data``), so recomputed results plug in with ``root="results"``.
Time steps: ``delta`` is a float for the Euler map; ``0`` or ``None`` selects continuous time (theory with
``delta = null``, networks integrated with RK4).  Loaded dicts carry their file name under ``"_file"``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from rnn_lyapunov import is_continuous

DATA = Path(__file__).resolve().parents[1] / "data"


def _root(root) -> Path:
    return DATA if root is None else Path(root)


@lru_cache(maxsize=None)
def _load_dir(path: Path) -> tuple:
    out = []
    for f in sorted(path.glob("*.json")) if path.is_dir() else []:
        r = json.loads(f.read_text())
        r["_file"] = f"{path.name}/{f.name}"
        out.append(r)
    return tuple(out)


def load_dir(subdir: str, root=None, kind: str | None = None) -> list[dict]:
    """Every result in ``<root>/<subdir>/*.json`` (of ``kind`` if given; cached per directory; treat as read-only)."""
    return [r for r in _load_dir(_root(root).resolve() / subdir) if kind is None or r.get("kind") == kind]


def _same_delta(a, b) -> bool:
    return is_continuous(a) if is_continuous(b) else (not is_continuous(a) and abs(a - b) < 1e-12)


def theory_scrambles(delta, g: float | None = None, root=None, subdir: str = "theory", **settings) -> list[dict]:
    """Theory results at ``(delta, g)`` (every g if None), sorted by ``(g, scramble)``.

    ``settings`` filter on ``config`` fields (``T=64``, ``m=160``, ``n_twist=4``) or the grid rule
    (``rule="full"``, ``"upper"``, ``"top"``).
    """
    def ok(r):
        c = r["config"]
        return _same_delta(r["delta"], delta) and (g is None or abs(r["g"] - g) < 1e-12) and all(
            (r["grid"]["rule"] if k == "rule" else c[k]) == v for k, v in settings.items())
    return sorted((r for r in load_dir(subdir, root, "theory_cdf") if ok(r)), key=lambda r: (r["g"], r["config"]["seed"]))


def networks(delta, g: float | None = None, N: int | None = None, root=None) -> list[dict]:
    """Network runs at time step ``delta`` (``0``/``None``: RK4 runs of the flow), optionally at ``g`` and ``N``,
    sorted by ``(g, N, seed)``.  ``"zero_diagonal": true`` marks the two runs with ``J_ii = 0``."""
    def ok(r):
        d = None if r["params"]["scheme"] == "flow" else r["delta"]
        return _same_delta(d, delta) and (g is None or abs(r["g"] - g) < 1e-12) and \
            (N is None or r["params"]["N"] == N)
    return sorted((r for r in load_dir("network", root, "network_lyapunov") if ok(r)),
                  key=lambda r: (r["g"], r["params"]["N"], r["params"]["seed"]))


def pr_simulations(delta, g: float | None = None, N: int | None = None, root=None) -> list[dict]:
    """Participation-ratio simulations at ``delta`` (``0``/``None``: RK4), sorted by ``(g, N, seed)``."""
    def ok(r):
        return _same_delta(r["delta"], delta) and (g is None or abs(r["g"] - g) < 1e-12) and \
            (N is None or r["N"] == N)
    return sorted((r for r in load_dir("participation", root, "participation_simulation") if ok(r)), key=lambda r: (r["g"], r["N"], r["seed"]))


def pr_theory(delta, root=None) -> dict:
    """The participation-ratio theory file at ``delta`` (None if absent): ``rows``, one per g, with ``g``, ``pr_x``,
    ``pr_phi``, ``psi_x``, ``psi_phi``, ``mean_gain``, ``q``, ``dmft_tmax``, ..."""
    hits = [r for r in load_dir("participation_theory", root, "participation_theory") if _same_delta(r["delta"], delta)]
    assert len(hits) <= 1, f"{len(hits)} PR theory files at delta={delta}"
    return hits[0] if hits else None


def max_exponent_data(root=None) -> dict:
    """``max_exponent_operator.json`` (scripts/run_max_exponent.py).  Keys:

    ``a``  continuous time, g = G_A: ``tau``, potential ``V`` (|tau| <= 12), ``V_inf``, ``V_min``, bound-state
           energies ``E``, eigenfunctions ``psi`` (one list per bound state), ``cxp_tau``/``cxp`` (normalized
           C^x'(tau)), ``overlap_psi1_cxp``, ``checks`` (grid and window checks).
    ``b``  delta = D_B, g = G_A: thresholds ``s``, ``levels`` (8 lowest eigenvalues of alpha_s^{-1} T_s per s),
           ``band`` (continuum edges per s), ``s_u``, ``s_star``, ``window_checks``.
    ``c``  ``{"0.25" | "0.5" | "1" | "0": [{g, t_max, s_star, edge_lib}, ...]}`` on 30 values of g ("0": delta -> 0).
    ``c_checks``, ``G_A``, ``D_B``, ``quadrature``, ``provenance``.
    """
    return json.loads((_root(root) / "max_exponent_operator.json").read_text())


def manifest(root=None) -> dict:
    """``data/manifest.json``: sha256, size, kind, GPU seconds and plotted uses of every file."""
    return json.loads((_root(root) / "manifest.json").read_text())
