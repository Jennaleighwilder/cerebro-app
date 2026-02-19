#!/usr/bin/env python3
"""Tests for cerebro_data_gather.py — run with pytest or python -m pytest"""

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_output_dir_exists_after_run():
    """After running gather, cerebro_data dir should exist."""
    assert DATA_DIR.exists(), "Run cerebro_data_gather.py first"


def test_fred_csv_exists():
    """FRED combined CSV should exist and have data."""
    p = DATA_DIR / "FRED_combined.csv"
    assert p.exists(), f"Missing {p}"
    import pandas as pd
    df = pd.read_csv(p, index_col=0)
    assert len(df) >= 50, "FRED should have 50+ years"
    assert "unemployment_rate" in df.columns or len(df.columns) >= 1


def test_ucdp_downloaded():
    """UCDP xlsx should exist."""
    p = DATA_DIR / "UCDP_armed_conflict_251.xlsx"
    assert p.exists(), f"Missing {p}"
    assert p.stat().st_size > 1000, "UCDP file too small"


def test_consolidated_csv():
    """Consolidated raw CSV should exist."""
    p = DATA_DIR / "cerebro_gathered_raw.csv"
    assert p.exists(), f"Missing {p}"
    import pandas as pd
    df = pd.read_csv(p, index_col=0)
    assert len(df) >= 60, "Should have 60+ years"


if __name__ == "__main__":
    for name, fn in [
        ("output_dir", test_output_dir_exists_after_run),
        ("fred_csv", test_fred_csv_exists),
        ("ucdp", test_ucdp_downloaded),
        ("consolidated", test_consolidated_csv),
    ]:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
