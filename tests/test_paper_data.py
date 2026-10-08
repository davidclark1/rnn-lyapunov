"""Bundled data and the paper layer: file integrity, and the tables behind every figure build from data/.

The tables' numbers were verified against the original results by tools/verify_dataset.py (max difference 0); here
a few published values are pinned so that an accidental change to data/ or paper/tables.py is caught.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from paper import data, tables

ROOT = Path(__file__).resolve().parents[1] / "data"


def test_manifest_sha256():
    m = json.loads((ROOT / "manifest.json").read_text())
    for name, info in m["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == info["sha256"], name


def test_every_theory_file_converged():
    for f in (ROOT / "theory").glob("*.json"):
        assert json.loads(f.read_text())["all_converged"], f.name


@pytest.fixture(scope="module")
def dim():
    return tables.dimension_entropy_table()


def test_dimension_entropy_points(dim):
    pts = dim["points"]
    assert {p["delta"] for p in pts.values()} == {0.0, 0.25, 0.5}
    p = pts["delta0.5_g20"]
    # theory and simulation agree closely at the strongest coupling (paper Fig. dimension_entropy)
    assert abs(p["theory"]["ky_dimension"] - p["network"]["ky_dimension"]) < 0.03 * p["network"]["ky_dimension"]
    q = pts["delta0_g20"]
    assert q["theory"]["extrapolated"] and q["network"]["N"] == 4096
    assert pts["delta0.25_g1.25"]["network"]["N"] == 16384 and pts["delta0_g1.25"]["network"]["N"] == 8192


def test_spectra_table():
    t = tables.spectra_table()
    assert len(t) == 8
    for v in t.values():
        assert v["network"]["N"] == 4096 and len(v["network"]["exponents"]) == 4096
        F = np.asarray(v["theory"]["F"])
        assert np.all((F >= 0) & (F <= 1))


def test_max_exponent_data_consistent():
    d = data.max_exponent_data()
    for rows in d["c"].values():
        for r in rows:            # the operator's s_* and the theory pipeline's upper edge agree
            assert abs(r["s_star"] - r["edge_lib"]) < 1e-3
    assert set(tables.max_exponent_table()) == {"0.25", "0.5", "1", "0"}
