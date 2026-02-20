#!/usr/bin/env python3
"""Tests for CHIMERA Synthetic Adversarial Worlds."""

import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_synthetic_worlds_run():
    """Synthetic worlds module runs and writes cerebro_data/synthetic_worlds.json."""
    from chimera import chimera_synthetic_worlds

    out = chimera_synthetic_worlds.run_synthetic_worlds()
    assert "worlds" in out
    assert "false_positive_rate" in out
    assert "noise_world_confidence_mean" in out
    for name in ["linear_trend", "cyclical", "high_noise", "sudden_regime_jump", "near_saddle_illusion"]:
        assert name in out["worlds"]


def test_synthetic_worlds_schema():
    """Output schema has required metrics per world."""
    p = DATA_DIR / "synthetic_worlds.json"
    if not p.exists():
        pytest.skip("synthetic_worlds.json not yet generated")
    with open(p) as f:
        d = json.load(f)
    assert "worlds" in d
    for wname, w in d["worlds"].items():
        if "error" in w:
            continue
        assert "false_positive_rate" in w
        assert "confidence_mean" in w
        assert "window_width_mean" in w
        assert "confidence_mean_calibrated" in w


def test_synthetic_noise_confidence_calibrated():
    """Noise world must not get absurd confidence after calibration (military-level: noise doesn't get high certainty)."""
    from chimera import chimera_synthetic_worlds

    out = chimera_synthetic_worlds.run_synthetic_worlds()
    noise = out.get("worlds", {}).get("high_noise", {})
    if "error" in noise:
        pytest.skip("high_noise world had error")
    conf_cal = noise.get("confidence_mean_calibrated", noise.get("confidence_mean", 1.0))
    assert conf_cal <= 0.75, f"high_noise confidence_mean_calibrated={conf_cal} must be <= 0.75"


def test_confidence_calibrator_mapping():
    """Calibration mapping produces values in [50, 95] and applies caps."""
    from chimera import chimera_confidence_calibrator

    cal_pct, reason = chimera_confidence_calibrator.calibrate(85, n_eff=10, average_integrity=0.8)
    assert 50 <= cal_pct <= 95
    cal_low, r = chimera_confidence_calibrator.calibrate(85, n_eff=3, average_integrity=0.8)
    assert cal_low <= 85 or "low_n_eff" in r


def test_infinity_score_uses_synthetic():
    """Infinity score includes synthetic_worlds in inputs_present."""
    from chimera import chimera_synthetic_worlds
    from chimera import chimera_infinity_score

    chimera_synthetic_worlds.run_synthetic_worlds()
    out = chimera_infinity_score.compute_infinity_score()
    assert "inputs_present" in out
    assert "synthetic_worlds" in out["inputs_present"]
