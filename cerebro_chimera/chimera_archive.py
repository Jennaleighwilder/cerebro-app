#!/usr/bin/env python3
"""
CHIMERA ARCHIVE — Sign all outputs (SHA256)
===========================================
Hash chimera outputs + honeycomb + distance_weights + calibration.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_archive_signature.json"
DATA_DIR = SCRIPT_DIR / "cerebro_data"

FILES_TO_HASH = [
    "chimera_reconstruction.json",
    "chimera_forward_simulation.json",
    "chimera_stress_matrix.json",
    "chimera_coupling.json",
    "chimera_evolution.json",
    "chimera_entropy.json",
    "chimera_failure.json",
    "chimera_validation.json",
    "chimera_master.json",
    "honeycomb_latest.json",
    "distance_weights.json",
    "honeycomb_conformal.json",
]


def _git_commit_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_archive() -> dict:
    """Concatenate JSON contents, compute SHA256."""
    parts = []
    for name in FILES_TO_HASH:
        p = DATA_DIR / name
        if p.exists():
            try:
                with open(p) as f:
                    parts.append(f.read())
            except Exception:
                parts.append(f"{name}:error")
        else:
            parts.append(f"{name}:missing")

    concatenated = "\n".join(parts)
    h = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()

    return {
        "version": 1,
        "run_signature": h,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit_hash(),
        "files_hashed": len([n for n in FILES_TO_HASH if (DATA_DIR / n).exists()]),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_archive()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera archive: signature={r.get('run_signature', '')[:16]}... → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
