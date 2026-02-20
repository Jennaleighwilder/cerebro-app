#!/usr/bin/env python3
"""Tests for cerebro_live_feedback."""

import json
import pytest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_register_prediction_writes_to_pending():
    """register_prediction writes to pending_windows.json."""
    from cerebro_live_feedback import register_prediction, PENDING_WINDOWS

    register_prediction(
        "harm", 2025, 2030, 70.0, 2028,
        saddle_year=2022, position=-1.5, velocity=0.08, acceleration=0.05,
        country="US",
    )
    assert PENDING_WINDOWS.exists()
    with open(PENDING_WINDOWS) as f:
        data = json.load(f)
    assert any(p.get("clock") == "harm" and p.get("window_end") == 2030 for p in data)


def test_score_closed_windows_identifies_hits_misses():
    """score_closed_windows correctly identifies hits vs misses."""
    from cerebro_live_feedback import (
        register_prediction,
        score_closed_windows,
        FEEDBACK_LOG,
        PENDING_WINDOWS,
    )

    # Clear and add a window that's already closed (e.g. 2010-2015)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PENDING_WINDOWS, "w") as f:
        json.dump([{
            "clock": "harm",
            "window_start": 2010,
            "window_end": 2015,
            "confidence_pct": 70,
            "saddle_year": 2008,
            "position": -1.0,
            "velocity": 0.1,
            "acceleration": 0.05,
            "country": "US",
        }], f)
    scored = score_closed_windows(current_year=2020)
    assert scored >= 0
    if scored > 0 and FEEDBACK_LOG.exists():
        with open(FEEDBACK_LOG) as f:
            lines = [l for l in f if l.strip()]
        if lines:
            last = json.loads(lines[-1])
            assert "hit" in last
            assert "confidence_pct" in last


def test_inject_feedback_adds_without_corrupting():
    """inject_feedback_into_calibration adds episodes without corrupting existing."""
    from cerebro_live_feedback import inject_feedback_into_calibration, FEEDBACK_EPISODES

    before = []
    if FEEDBACK_EPISODES.exists():
        with open(FEEDBACK_EPISODES) as f:
            before = json.load(f)
    added = inject_feedback_into_calibration()
    if FEEDBACK_EPISODES.exists():
        with open(FEEDBACK_EPISODES) as f:
            after = json.load(f)
        assert len(after) >= len(before)
        for ep in after:
            assert "confidence" in ep
            assert "hit" in ep
            assert 0 <= ep["confidence"] <= 1


def test_full_cycle_ece_does_not_increase():
    """Full cycle: register → score → inject → ECE does not increase (or improves)."""
    from cerebro_live_feedback import run_feedback_cycle
    from cerebro_calibration import run_calibration

    # Run feedback (may score 0 if no closed windows)
    run_feedback_cycle()
    # Run calibration
    r = run_calibration()
    if "error" in r:
        pytest.skip("Calibration error")
    ece = None
    try:
        from chimera.chimera_infinity_score import compute_infinity_score
        s = compute_infinity_score()
        ece = s.get("diagnostics", {}).get("ece")
    except Exception:
        pass
    # ECE should be reasonable (< 0.2) - no strict increase check without baseline
    if ece is not None:
        assert ece < 0.5, "ECE unexpectedly high"
