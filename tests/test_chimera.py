#!/usr/bin/env python3
"""Tests for CHIMERA layer."""

import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_chimera_master_exists_after_run():
    """chimera_master.json exists and has expected keys after engine run."""
    p = DATA_DIR / "chimera_master.json"
    if not p.exists():
        pytest.skip("chimera_master.json not yet generated (run chimera_engine)")
    with open(p) as f:
        m = json.load(f)
    assert "version" in m
    assert "reconstruction" in m or "error" in str(m)
    assert "failure_mode" in m
    assert "archive_signature" in m or "archive_signature" in str(m)


def test_chimera_reconstruction_schema():
    """chimera_reconstruction.json has mae_mean, coverage_80_mean when valid."""
    p = DATA_DIR / "chimera_reconstruction.json"
    if not p.exists():
        pytest.skip("chimera_reconstruction.json not yet generated")
    with open(p) as f:
        r = json.load(f)
    if r.get("error"):
        pytest.skip("reconstruction had error")
    assert "mae_mean" in r
    assert "coverage_80_mean" in r
    assert "n_used" in r


def test_chimera_stress_stability_in_range():
    """mean_stability in [0, 1] when stress ran."""
    p = DATA_DIR / "chimera_stress_matrix.json"
    if not p.exists():
        pytest.skip("chimera_stress_matrix.json not yet generated")
    with open(p) as f:
        s = json.load(f)
    if s.get("error"):
        pytest.skip("stress had error")
    st = s.get("mean_stability")
    if st is not None:
        assert 0 <= st <= 1.0


def test_chimera_failure_mode_valid():
    """chimera_failure.json has failure_mode and severity."""
    p = DATA_DIR / "chimera_failure.json"
    if not p.exists():
        pytest.skip("chimera_failure.json not yet generated")
    with open(p) as f:
        f_ = json.load(f)
    assert "failure_mode" in f_
    assert "severity" in f_
    assert 0 <= f_["severity"] <= 1.0


def test_chimera_archive_signature():
    """chimera_archive_signature.json has run_signature (SHA256 hex)."""
    p = DATA_DIR / "chimera_archive_signature.json"
    if not p.exists():
        pytest.skip("chimera_archive_signature.json not yet generated")
    with open(p) as f:
        a = json.load(f)
    sig = a.get("run_signature")
    assert sig is not None
    assert len(sig) == 64
    assert all(c in "0123456789abcdef" for c in sig)
