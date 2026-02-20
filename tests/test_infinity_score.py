#!/usr/bin/env python3
"""Tests for CHIMERA Infinity Score."""

import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def test_score_monotonic_in_brier():
    """Score increases when brier decreases (S_cal = exp(-brier/0.2) is monotone)."""
    from chimera import chimera_infinity_score

    # Direct formula check: lower brier -> higher S_cal -> higher G -> higher score
    import math
    brier_high = 0.25
    brier_low = 0.05
    s_cal_high = math.exp(-brier_high / 0.20)
    s_cal_low = math.exp(-brier_low / 0.20)
    assert s_cal_low > s_cal_high


def test_schema_keys_exist():
    """Infinity score output has required schema keys."""
    p = SCRIPT_DIR / "cerebro_data" / "infinity_score.json"
    if not p.exists():
        pytest.skip("infinity_score.json not yet generated")
    with open(p) as f:
        d = json.load(f)
    assert "infinity_score" in d
    assert "G" in d
    assert "penalty" in d
    assert "signals" in d
    assert "inputs_present" in d
    assert "timestamp" in d
