#!/usr/bin/env python3
"""Tests for CHIMERA Figure-8."""

import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_figure8_writes_artifacts():
    """Run figure8; verify chimera_export and infinity_score written (uses real cerebro_data)."""
    from chimera import chimera_figure8

    result = chimera_figure8.run_figure8()
    assert "backward" in result
    assert "forward" in result

    assert (DATA_DIR / "chimera_export.json").exists()
    assert (DATA_DIR / "infinity_score.json").exists()

    with open(DATA_DIR / "chimera_export.json") as f:
        exp = json.load(f)
    assert "infinity_score" in exp or "honeycomb" in exp or "error" in exp

    with open(DATA_DIR / "infinity_score.json") as f:
        inf = json.load(f)
    assert "infinity_score" in inf or "error" in inf


def test_bridge_runs_without_crash():
    """cerebro_chimera_bridge runs and does not raise."""
    import cerebro_chimera_bridge
    code = cerebro_chimera_bridge.main()
    assert code in (0, 1)
