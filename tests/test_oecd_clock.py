#!/usr/bin/env python3
"""Tests for OECD clock ingest."""

from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OECD_DIR = SCRIPT_DIR / "cerebro_data" / "oecd"


def test_oecd_clock_files_exist():
    """OECD clock directory exists and has clock files after ingest."""
    # Run ingest to ensure files exist
    import cerebro_oecd_clock_ingest
    cerebro_oecd_clock_ingest.main()
    assert OECD_DIR.exists()
    files = list(OECD_DIR.glob("*_clock.csv"))
    assert len(files) >= 1, "At least one OECD clock file should exist"


def test_oecd_schema():
    """OECD clock files have required schema."""
    import cerebro_oecd_clock_ingest
    cerebro_oecd_clock_ingest.main()
    if not OECD_DIR.exists():
        pytest.skip("OECD dir not created")
    files = list(OECD_DIR.glob("*_clock.csv"))
    if not files:
        pytest.skip("No OECD clock files")
    df = pd.read_csv(files[0])
    required = {"year", "position", "velocity", "acceleration", "saddle_score_phase1"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"


def test_oecd_ingest_runs():
    """OECD ingest completes without error."""
    import cerebro_oecd_clock_ingest
    code = cerebro_oecd_clock_ingest.main()
    assert code == 0
