#!/usr/bin/env python3
"""Tests for CHIMERA confidence calibration and volatility."""

import pytest


def test_volatility_index():
    """Volatility index returns 0-1, high for noisy series."""
    from chimera import chimera_volatility

    # Flat series -> low vol
    flat_pos = [1.0] * 15
    flat_vel = [0.0] * 15
    flat_acc = [0.0] * 15
    vol_flat = chimera_volatility.compute_volatility_index(flat_pos, flat_vel, flat_acc, k=10)
    assert 0 <= vol_flat <= 1.0

    # Noisy series -> higher vol
    import numpy as np
    np.random.seed(42)
    r = np.random.normal(0, 0.5, 20)
    noisy_pos = np.cumsum(r).tolist()
    noisy_vel = np.concatenate([[0], np.diff(noisy_pos)]).tolist()
    noisy_acc = np.concatenate([[0], np.diff(noisy_vel)]).tolist()
    vol_noisy = chimera_volatility.compute_volatility_index(noisy_pos, noisy_vel, noisy_acc, k=10)
    assert vol_noisy >= vol_flat


def test_calibrate_peak_window_adds_raw_and_calibrated():
    """calibrate_peak_window adds confidence_pct_raw and confidence_pct_calibrated."""
    from chimera import chimera_confidence_calibrator

    pw = {"confidence_pct": 85, "window_start": 2025, "window_end": 2030, "analogue_count": 10}
    out = chimera_confidence_calibrator.calibrate_peak_window(pw)
    assert "confidence_pct_raw" in out
    assert "confidence_pct" in out
    assert out["confidence_pct_raw"] == 85
    assert 50 <= out["confidence_pct"] <= 95
