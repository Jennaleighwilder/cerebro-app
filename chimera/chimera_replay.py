#!/usr/bin/env python3
"""
CHIMERA Replay — Load a memory ledger entry and verify hashes (best-effort).
Prints a replay report. Does not rebuild the full run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import sys
from pathlib import Path

from chimera import chimera_memory

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
MEMORY_LOG = DATA_DIR / "chimera_memory.jsonl"


def load_entry(index: int = -1) -> dict | None:
    """
    Load memory entry by index. -1 = last, -2 = second-to-last, etc.
    Returns None if log empty or index out of range.
    """
    if not MEMORY_LOG.exists():
        return None
    lines = [ln.strip() for ln in MEMORY_LOG.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        entry = json.loads(lines[index])
        return entry
    except (IndexError, json.JSONDecodeError):
        return None


def verify_hashes(entry: dict) -> dict[str, bool | str]:
    """
    Best-effort verification: re-hash inputs and compare.
    Returns {path: True|False|"missing"} for each input.
    """
    results: dict[str, bool | str] = {}
    inputs_hashed = entry.get("inputs_hashed", {})
    for path_str, stored_hash in inputs_hashed.items():
        p = Path(path_str)
        if not p.exists():
            results[path_str] = "missing"
        else:
            current = chimera_memory.hash_file(p)
            results[path_str] = current == stored_hash
    return results


def replay_report(index: int = -1) -> str:
    """
    Generate a replay report for the chosen entry.
    Returns a multi-line string.
    """
    entry = load_entry(index)
    if entry is None:
        return "No memory entry found."

    lines: list[str] = []
    lines.append("=== CHIMERA Replay Report ===")
    lines.append(f"Tag: {entry.get('tag', '?')}")
    lines.append(f"Timestamp: {entry.get('timestamp_utc', '?')}")
    lines.append(f"Git commit: {entry.get('git_commit', '?')}")
    lines.append(f"Core hash lock: {entry.get('core_hash_lock', '?') or '(none)'}")
    lines.append(f"Input count: {entry.get('input_count', 0)}")
    lines.append(f"Params hash: {entry.get('params_hash', '') or '(none)'}")
    lines.append(f"Outputs hash: {entry.get('outputs_hash', '') or '(none)'}")

    verified = verify_hashes(entry)
    ok = sum(1 for v in verified.values() if v is True)
    miss = sum(1 for v in verified.values() if v == "missing")
    fail = sum(1 for v in verified.values() if v is False)
    lines.append("")
    lines.append("Hash verification (best-effort):")
    lines.append(f"  Match: {ok}, Missing: {miss}, Mismatch: {fail}")

    return "\n".join(lines)


def main() -> int:
    idx = -1
    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1])
        except ValueError:
            pass
    report = replay_report(idx)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
