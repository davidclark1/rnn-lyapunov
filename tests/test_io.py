"""Result files with provenance: save/load round trip, provenance fields, names containing dots."""
import json

import numpy as np

from rnn_lyapunov import __version__, io


def test_round_trip_with_provenance(tmp_path):
    summary = dict(g=3.0, delta=0.5, F=np.array([0.1, 0.5]), n=np.int64(4), ok=np.bool_(True),
                   nested=dict(eps=(1e-1, 1e-2)), where=tmp_path)
    arrays = dict(s=np.linspace(-1, 0, 5), F=np.arange(6.0).reshape(2, 3))
    p = io.save_result(tmp_path / "run", summary, arrays)
    assert p == tmp_path / "run.json" and p.exists() and (tmp_path / "run.npz").exists()
    out, arr = io.load_result(tmp_path / "run")
    assert out["g"] == 3.0 and out["F"] == [0.1, 0.5] and out["n"] == 4 and out["ok"] is True
    assert out["nested"]["eps"] == [0.1, 0.01] and out["where"] == str(tmp_path)
    prov = out["provenance"]
    assert prov["package"] == "rnn_lyapunov" and prov["version"] == __version__
    assert prov["code_sha"] == io.source_hash() and len(prov["code_sha"]) == 16
    for key in ("created_utc", "argv", "python", "numpy", "scipy", "torch", "host"):
        assert key in prov
    assert set(arr) == {"s", "F"} and np.array_equal(arr["F"], arrays["F"])


def test_names_with_dots_and_explicit_suffix(tmp_path):
    p = io.save_result(tmp_path / "theory_g3.0_delta0.5", dict(a=1), dict(x=np.ones(2)))
    assert p.name == "theory_g3.0_delta0.5.json"
    assert (tmp_path / "theory_g3.0_delta0.5.npz").exists()
    # loading with or without .json gives the same result
    a, xa = io.load_result(tmp_path / "theory_g3.0_delta0.5")
    b, xb = io.load_result(p)
    assert a == b and np.array_equal(xa["x"], xb["x"])
    # an explicit .json suffix is not doubled
    assert io.save_result(tmp_path / "h0.1.json", dict(a=2)).name == "h0.1.json"


def test_array_dir_and_no_arrays(tmp_path):
    p = io.save_result(tmp_path / "res" / "r0.25", dict(a=1), dict(x=np.zeros(3)), array_dir=tmp_path / "bulk")
    meta = json.loads(p.read_text())
    assert meta["arrays_file"] == str(tmp_path / "bulk" / "r0.25.npz")
    _, arr = io.load_result(p)
    assert np.array_equal(arr["x"], np.zeros(3))
    q = io.save_result(tmp_path / "plain", dict(a=1))
    s, arr = io.load_result(q)
    assert arr == {} and "arrays_file" not in s


def test_source_hash_is_stable():
    assert io.source_hash() == io.source_hash()
