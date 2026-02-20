#!/usr/bin/env python3
"""
CHIMERA Memory Ledger — Proof layer for auditability and replay.
Every run writes a signed-ish ledger entry: git commit, core hash, input hashes, params hash, outputs hash.
Append-only: cerebro_data/chimera_memory.jsonl
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
MEMORY_LOG = DATA_DIR / "chimera_memory.jsonl"
CORE_HASH_LOCK = DATA_DIR / "core_hash_lock.txt"
PARAMS_DIR = DATA_DIR / "params"
HARM_CLOCK_CSV = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
OECD_DIR = DATA_DIR / "oecd"


def hash_file(path: Path) -> str:
    """SHA256 of file contents. Returns hex digest or empty string if unreadable."""
    if not path.exists() or not path.is_file():
        return ""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def hash_tree(paths: list[Path]) -> dict[str, str]:
    """Hash each path. Returns {str(path): sha256}."""
    out: dict[str, str] = {}
    for p in paths:
        if p.exists():
            out[str(p)] = hash_file(p)
    return out


def _git_commit() -> str:
    """Git rev-parse HEAD. Returns 'unknown' if not a repo or fails."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _gather_input_paths() -> list[Path]:
    """Paths to hash for inputs_hashed."""
    paths: list[Path] = []
    # cerebro_data/*.csv
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*.csv"):
            paths.append(f)
        # cerebro_data/*.json (exclude chimera_memory.jsonl - that's the log)
        for f in DATA_DIR.glob("*.json"):
            if f.name != "chimera_memory.jsonl":
                paths.append(f)
        # cerebro_data/oecd/*.csv
        if OECD_DIR.exists():
            for f in OECD_DIR.glob("*.csv"):
                paths.append(f)
    # cerebro_harm_clock_data.csv (project root)
    if HARM_CLOCK_CSV.exists():
        paths.append(HARM_CLOCK_CSV)
    return sorted(paths)


def _gather_param_paths() -> list[Path]:
    """Paths for params_hash."""
    if not PARAMS_DIR.exists():
        return []
    return sorted(PARAMS_DIR.glob("*.json"))


def snapshot_manifest(
    tag: str = "run",
    outputs_hash: str | None = None,
) -> dict[str, Any]:
    """
    Build a manifest for this run.
    tag: optional label (e.g. 'backward', 'forward', 'run')
    outputs_hash: optional hash of key outputs (e.g. infinity_score + chimera_export)
    """
    input_paths = _gather_input_paths()
    param_paths = _gather_param_paths()

    inputs_hashed = hash_tree(input_paths)
    params_hash = ""
    if param_paths:
        combined = "".join(hash_file(p) for p in param_paths)
        if combined:
            params_hash = hashlib.sha256(combined.encode()).hexdigest()

    core_hash_lock = ""
    if CORE_HASH_LOCK.exists():
        try:
            core_hash_lock = CORE_HASH_LOCK.read_text().strip()
        except Exception:
            pass

    manifest = {
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(),
        "core_hash_lock": core_hash_lock,
        "inputs_hashed": inputs_hashed,
        "params_hash": params_hash,
        "outputs_hash": outputs_hash or "",
        "input_count": len(inputs_hashed),
        "params_count": len(param_paths),
    }
    return manifest


def append_memory_entry(manifest: dict[str, Any]) -> None:
    """Append one JSON line to cerebro_data/chimera_memory.jsonl."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_LOG, "a") as f:
        f.write(json.dumps(manifest) + "\n")


def write_snapshot(tag: str = "run", outputs_hash: str | None = None) -> dict[str, Any]:
    """
    Create snapshot manifest and append to memory log.
    Returns the manifest.
    """
    manifest = snapshot_manifest(tag=tag, outputs_hash=outputs_hash)
    append_memory_entry(manifest)
    return manifest


def main() -> int:
    """CLI: write a snapshot and print summary."""
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    manifest = write_snapshot(tag=tag)
    print(f"CHIMERA Memory: {manifest['input_count']} inputs, git={manifest['git_commit'][:8]}")
    print(f"  → {MEMORY_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
