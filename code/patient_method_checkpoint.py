"""Exact-input checkpoints for completed patient reconstruction methods.

The JSON record is the commit marker for an NPZ terminal field. A crash between
the two atomic replacements cannot make a partial or mismatched pair reusable.
No solver iterate, pickle payload, or approximate input match is accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from zipfile import BadZipFile

import numpy as np


DIAGNOSTIC_KEYS = frozenset({
    "solver_relative_residual", "maximum_state_residual",
    "maximum_admm_iterations", "maximum_mass_source_defect",
    "maximum_graph_projection_residual",
})
SCREENED_UNDEFINED = frozenset({
    "maximum_mass_source_defect", "maximum_graph_projection_residual",
})


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_signature(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
    source_files: Mapping[str, Path],
) -> dict[str, Any]:
    """Describe every input by exact shape, dtype and byte digest."""
    fields: dict[str, Any] = {}
    for name, values in arrays.items():
        array = np.ascontiguousarray(values)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"checkpoint input {name} is not finite numeric data")
        fields[name] = {
            "shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    descriptor = {
        "schema_version": 1, "metadata": dict(metadata), "arrays": fields,
        "sources": {name: file_sha256(path) for name, path in source_files.items()},
    }
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"sha256": hashlib.sha256(payload.encode()).hexdigest(), "inputs": descriptor}


def _diagnostics_json(method: str, diagnostics: Mapping[str, float]) -> dict[str, Any]:
    if set(diagnostics) != DIAGNOSTIC_KEYS:
        raise ValueError("checkpoint diagnostic keys differ from the expected method summary")
    encoded = {}
    for name, value in diagnostics.items():
        number = float(value)
        if method == "screened" and name in SCREENED_UNDEFINED and np.isnan(number):
            encoded[name] = None
        elif not np.isfinite(number) or number < 0.0:
            raise ValueError(f"invalid checkpoint diagnostic {name}")
        else:
            encoded[name] = number
    return encoded


def load_completed(
    root: Path, key: str, method: str, signature: dict[str, Any], n_vertices: int,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Return only a complete exact-input match; reject partial/stale records."""
    record_path = root / f"{key}.json"
    array_path = root / f"{key}.npz"
    if not record_path.is_file() or not array_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record.get("schema_version") != 1 or record.get("completed") is not True
                or record.get("method") != method or record.get("signature") != signature):
            return None
        if file_sha256(array_path) != record.get("npz_sha256"):
            raise ValueError("NPZ checksum differs from committed record")
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != {"terminal_state"}:
                raise ValueError("checkpoint archive has unexpected fields")
            state = np.asarray(archive["terminal_state"], dtype=float)
        if state.shape != (n_vertices,) or not np.isfinite(state).all():
            raise ValueError("checkpoint terminal state has invalid shape or values")
        raw = record["diagnostics"]
        diagnostics = {name: np.nan if value is None else float(value) for name, value in raw.items()}
        if _diagnostics_json(method, diagnostics) != raw:
            raise ValueError("checkpoint diagnostics do not match their declared encoding")
        return state, diagnostics
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
        print(f"checkpoint rejected {key}: {error}; recomputing", flush=True)
        return None


def save_completed(
    root: Path, key: str, method: str, signature: dict[str, Any],
    state: np.ndarray, diagnostics: Mapping[str, float],
) -> None:
    """Commit one completed terminal field and its diagnostics atomically."""
    state = np.asarray(state, dtype=float)
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("only a finite terminal vector may be checkpointed")
    encoded = _diagnostics_json(method, diagnostics)
    root.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=f".{key}-", suffix=".npz", delete=False) as stream:
            array_temp = Path(stream.name)
            temporary.append(array_temp)
            np.savez_compressed(stream, terminal_state=state)
            stream.flush()
            os.fsync(stream.fileno())
        record = {
            "schema_version": 1, "completed": True, "method": method,
            "signature": signature, "npz_sha256": file_sha256(array_temp),
            "diagnostics": encoded,
        }
        with tempfile.NamedTemporaryFile(dir=root, prefix=f".{key}-", suffix=".json", mode="w", encoding="utf-8", delete=False) as stream:
            record_temp = Path(stream.name)
            temporary.append(record_temp)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(array_temp, root / f"{key}.npz")
        os.replace(record_temp, root / f"{key}.json")
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
