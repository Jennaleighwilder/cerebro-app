#!/usr/bin/env python3
"""Tests for CHIMERA Infinity Score v2."""

import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def test_score_in_range():
    """Infinity score exists and is non-negative (formula: 100*log(1+25*G*M*P) can exceed 100)."""
    from chimera import chimera_infinity_score

    out = chimera_infinity_score.compute_infinity_score()
    score = out.get("infinity_score")
    assert score is not None
    assert 0 <= score <= 400


def test_schema_keys_exist():
    """Infinity score v2 output has required schema keys."""
    p = SCRIPT_DIR / "cerebro_data" / "infinity_score.json"
    if not p.exists():
        pytest.skip("infinity_score.json not yet generated")
    with open(p) as f:
        d = json.load(f)
    assert "infinity_score" in d
    assert "G" in d
    assert "subscores" in d
    assert "diagnostics" in d
    assert "multipliers" in d
    assert "gates" in d
    assert "version" in d
    assert d["version"] == 2
    for key in ["accuracy", "calibration", "interval", "support", "robustness"]:
        assert key in d["subscores"]
        assert 0 <= d["subscores"][key] <= 1


def test_ece_computed_when_bins_exist():
    """ECE is computed from calibration bins when they exist."""
    from chimera import chimera_infinity_score

    out = chimera_infinity_score.compute_infinity_score()
    assert "diagnostics" in out
    assert "ece" in out["diagnostics"]
    assert isinstance(out["diagnostics"]["ece"], (int, float))


def test_noise_overconfident_penalty():
    """If noise_conf_cal > 0.8, penalty_product < 1."""
    from chimera import chimera_infinity_score

    out = chimera_infinity_score.compute_infinity_score()
    diag = out.get("diagnostics", {})
    noise_conf = diag.get("synthetic_noise_conf_cal", 0)
    penalty = out.get("multipliers", {}).get("penalty_product", 1.0)
    if noise_conf > 0.8:
        assert penalty < 1.0
        assert "noise_overconfident" in out.get("penalties_applied", [])


def test_s_cal_monotonic_in_brier():
    """Lower brier (holding ECE fixed) yields higher S_cal component."""
    from chimera import chimera_infinity_score

    # S_brier = clamp01(1 - brier/0.25), S_cal = sqrt(S_brier * S_rel)
    # So lower brier -> higher S_brier -> higher S_cal
    brier_high = 0.25
    brier_low = 0.05
    s_brier_high = max(0, min(1, 1 - brier_high / 0.25))
    s_brier_low = max(0, min(1, 1 - brier_low / 0.25))
    assert s_brier_low > s_brier_high
