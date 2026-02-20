#!/usr/bin/env python3
"""Tests for CHIMERA Memory Ledger (proof layer)."""

import json
import tempfile
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def test_hash_file():
    """hash_file returns SHA256 hex digest."""
    from chimera import chimera_memory

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".txt") as f:
        f.write(b"hello")
        path = Path(f.name)
    try:
        h = chimera_memory.hash_file(path)
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
    finally:
        path.unlink(missing_ok=True)


def test_hash_tree():
    """hash_tree returns dict of path -> sha256."""
    from chimera import chimera_memory

    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.txt"
        p2 = Path(d) / "b.txt"
        p1.write_text("x")
        p2.write_text("y")
        tree = chimera_memory.hash_tree([p1, p2])
        assert len(tree) == 2
        assert str(p1) in tree
        assert str(p2) in tree
        assert tree[str(p1)] != tree[str(p2)]


def test_snapshot_manifest_required_keys():
    """snapshot_manifest returns dict with required keys."""
    from chimera import chimera_memory

    manifest = chimera_memory.snapshot_manifest(tag="test")
    assert "tag" in manifest
    assert manifest["tag"] == "test"
    assert "timestamp_utc" in manifest
    assert "git_commit" in manifest
    assert "core_hash_lock" in manifest
    assert "inputs_hashed" in manifest
    assert "params_hash" in manifest
    assert "outputs_hash" in manifest
    assert "input_count" in manifest
    assert "params_count" in manifest


def test_append_memory_entry(tmp_path):
    """append_memory_entry writes one JSON line to JSONL."""
    from chimera import chimera_memory

    # Monkeypatch DATA_DIR and MEMORY_LOG to tmp_path
    orig_data = chimera_memory.DATA_DIR
    orig_log = chimera_memory.MEMORY_LOG
    chimera_memory.DATA_DIR = tmp_path
    chimera_memory.MEMORY_LOG = tmp_path / "chimera_memory.jsonl"
    try:
        manifest = {"tag": "test_append", "timestamp_utc": "2026-01-01T00:00:00Z", "git_commit": "abc123"}
        chimera_memory.append_memory_entry(manifest)
        assert chimera_memory.MEMORY_LOG.exists()
        lines = chimera_memory.MEMORY_LOG.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["tag"] == "test_append"
        assert parsed["git_commit"] == "abc123"

        # Append second
        chimera_memory.append_memory_entry({"tag": "test_append2"})
        lines = chimera_memory.MEMORY_LOG.read_text().strip().splitlines()
        assert len(lines) == 2
    finally:
        chimera_memory.DATA_DIR = orig_data
        chimera_memory.MEMORY_LOG = orig_log


def test_write_snapshot_integration():
    """write_snapshot creates manifest and appends to log."""
    from chimera import chimera_memory

    manifest = chimera_memory.write_snapshot(tag="pytest_run")
    assert "inputs_hashed" in manifest
    assert chimera_memory.MEMORY_LOG.exists()
    lines = [ln for ln in chimera_memory.MEMORY_LOG.read_text().splitlines() if ln.strip()]
    last = json.loads(lines[-1])
    assert last["tag"] == "pytest_run"


def test_replay_load_and_report():
    """chimera_replay can load entry and produce report."""
    from chimera import chimera_replay

    report = chimera_replay.replay_report(-1)
    assert "CHIMERA Replay Report" in report or "No memory entry found" in report
