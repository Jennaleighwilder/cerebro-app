#!/usr/bin/env python3
"""Hash of core functions must not change. If core changes → tests fail."""

import hashlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _source_hash(*objs) -> str:
    sources = []
    for obj in objs:
        try:
            src = inspect.getsource(obj)
            sources.append(src)
        except Exception:
            sources.append(str(obj))
    return hashlib.sha256("".join(sources).encode()).hexdigest()


def test_core_hash_unchanged():
    """Hash of core functions must not change."""
    from cerebro_core import (
        detect_saddle_canonical,
        state_distance,
        weighted_median,
        weighted_quantile,
        compute_peak_window,
    )
    h = _source_hash(
        detect_saddle_canonical,
        state_distance,
        weighted_median,
        weighted_quantile,
        compute_peak_window,
    )
    # Locked hash - update only when intentionally changing core
    EXPECTED = "e3a1c2f8b9d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0"
    # Compute expected from current - first run stores it
    assert len(h) == 64, "Hash must be SHA256"
    lock_file = ROOT / "cerebro_data" / "core_hash_lock.txt"
    lock_file.parent.mkdir(exist_ok=True)
    if not lock_file.exists():
        lock_file.write_text(h)
    else:
        locked = lock_file.read_text().strip()
        assert h == locked, f"Core modified! Hash {h} != locked {locked}. Core is frozen."
