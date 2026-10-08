"""Result files with provenance: every saved result carries what is needed to reproduce it."""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__

PKG_DIR = Path(__file__).resolve().parent


def source_hash() -> str:
    """SHA-256 over the package sources: the code-revision identifier stored with results."""
    hsh = hashlib.sha256()
    for f in sorted(PKG_DIR.glob("*.py")):
        hsh.update(f.name.encode())
        hsh.update(f.read_bytes())
    return hsh.hexdigest()[:16]


def environment() -> dict:
    import scipy
    import torch
    env = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
               torch=torch.__version__, host=socket.gethostname(), platform=platform.platform())
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    return env


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return o


def _json_path(path: str | Path) -> Path:
    """Append ``.json`` (names contain dots such as ``delta0.1``, so never use ``with_suffix``)."""
    path = Path(path)
    return path if path.name.endswith(".json") else path.with_name(path.name + ".json")


def save_result(path: str | Path, summary: dict, arrays: dict | None = None,
                array_dir: str | Path | None = None) -> Path:
    """Write ``<path>.json`` (summary + provenance) and optionally ``<name>.npz`` (arrays).

    Provenance: package version, SHA of the package sources, UTC time, argv, library versions, host and GPU.
    ``array_dir`` lets large arrays live elsewhere (e.g. bulk storage); its location is recorded.
    """
    path = _json_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(summary)
    out["provenance"] = dict(package="rnn_lyapunov", version=__version__, code_sha=source_hash(), created_utc=datetime.now(timezone.utc).isoformat(),
                             argv=sys.argv, **environment())
    if arrays:
        npz = (Path(array_dir) if array_dir else path.parent) / (path.name[:-5] + ".npz")
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, **arrays)
        out["arrays_file"] = str(npz)
    path.write_text(json.dumps(_jsonable(out), indent=1))
    return path


def load_result(path: str | Path) -> tuple[dict, dict]:
    """``(summary, arrays)`` of a result written by :func:`save_result`."""
    path = _json_path(path)
    summary = json.loads(path.read_text())
    arrays = dict(np.load(summary["arrays_file"])) if "arrays_file" in summary else {}
    return summary, arrays
