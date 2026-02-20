#!/usr/bin/env python3
"""
CHIMERA STORE — Atomic write, versioning, signing
==================================================
Atomic write (temp then replace), keep last 20 versions, SHA256 signature.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
PARAMS_PATH = DATA_DIR / "chimera_params.json"
MASTER_PATH = DATA_DIR / "chimera_master.json"
CALIBRATION_PATH = DATA_DIR / "calibration_curve.json"
HISTORY_DIR = DATA_DIR / "chimera_params_history"
SIGNATURE_PATH = DATA_DIR / "chimera_signature.json"
MAX_VERSIONS = 20


def atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically: temp file then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def save_params_version(params: dict) -> None:
    """Save params to history (params_YYYYMMDD_HHMMSS.json), keep last MAX_VERSIONS."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    p = HISTORY_DIR / f"params_{ts}.json"
    atomic_write(p, params)
    # Prune old versions
    versions = sorted(HISTORY_DIR.glob("params_*.json"), key=lambda x: x.stat().st_mtime)
    for old in versions[:-MAX_VERSIONS]:
        old.unlink(missing_ok=True)


def compute_signature() -> dict:
    """SHA256 of chimera_params + chimera_master + calibration_curve."""
    parts = []
    for name, p in [
        ("chimera_params", PARAMS_PATH),
        ("chimera_master", MASTER_PATH),
        ("calibration_curve", CALIBRATION_PATH),
    ]:
        if p.exists():
            try:
                parts.append(p.read_text())
            except Exception:
                parts.append(f"{name}:error")
        else:
            parts.append(f"{name}:missing")
    h = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return {
        "signature": h,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "files": ["chimera_params.json", "chimera_master.json", "calibration_curve.json"],
    }


def write_signature() -> dict:
    """Compute and write chimera_signature.json."""
    sig = compute_signature()
    atomic_write(SIGNATURE_PATH, sig)
    return sig


def load_params() -> dict:
    """Load chimera_params.json or return defaults."""
    if not PARAMS_PATH.exists():
        return {
            "vel_weight": 100,
            "acc_weight": 2500,
            "tau": 1.0,
            "n_updates": 0,
            "updated_at": None,
            "rolling_mae": None,
        }
    try:
        with open(PARAMS_PATH) as f:
            return json.load(f)
    except Exception:
        return {"vel_weight": 100, "acc_weight": 2500, "tau": 1.0, "n_updates": 0}
