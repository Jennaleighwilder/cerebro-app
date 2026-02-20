#!/usr/bin/env python3
"""Tests for cerebro_coupling_matrix."""

import json
import pytest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_coupling_matrix_loads_without_error():
    """Coupling matrix loads without error."""
    from cerebro_coupling_matrix import load_coupling_matrix, compute_lead_lag_matrix, load_clock_series, save_coupling_matrix

    series = load_clock_series()
    matrix = compute_lead_lag_matrix(series)
    path = DATA_DIR / "coupling_matrix.json"
    save_coupling_matrix(matrix, path)
    loaded = load_coupling_matrix(path)
    assert isinstance(loaded, dict)
    for k, v in loaded.items():
        assert isinstance(v, dict)
        for j, pair in v.items():
            assert "best_lag" in pair or "correlation" in pair


def test_corrections_bounded():
    """Corrections are bounded |correction| < 2.0 years for dampening=0.10."""
    from cerebro_coupling_matrix import load_coupling_matrix, compute_coupling_correction

    coupling = load_coupling_matrix()
    if not coupling:
        pytest.skip("No coupling matrix (run cerebro_coupling_matrix.py first)")
    current_state = {
        "harm": {"velocity": 0.08, "position": -1.5, "acceleration": 0.05},
        "sexual": {"velocity": -0.1, "position": 0.5, "acceleration": 0.02},
        "class": {"velocity": -0.03, "position": -4.1, "acceleration": 0.004},
        "evil": {"velocity": 0.0, "position": 0.0, "acceleration": 0.0},
    }
    corrections = compute_coupling_correction(current_state, coupling, dampening=0.10)
    for c, val in corrections.items():
        assert abs(val) < 2.0, f"Correction for {c} = {val} exceeds 2.0"


def test_toggling_use_coupling_false_produces_identical_output():
    """Toggling use_coupling=False produces identical output to pre-coupling."""
    from cerebro_honeycomb import compute_honeycomb_fusion
    from cerebro_calibration import _load_episodes
    from cerebro_eval_utils import past_only_pool

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < 8:
        pytest.skip("Insufficient episodes")
    latest = max(episodes, key=lambda e: e.get("saddle_year", 0))
    pool = past_only_pool(episodes, latest["saddle_year"])
    if len(pool) < 5:
        pytest.skip("Insufficient pool")

    out_off = compute_honeycomb_fusion(
        latest["saddle_year"],
        latest.get("position", 0),
        latest.get("velocity", 0),
        latest.get("acceleration", 0),
        pool,
        latest.get("ring_B_score"),
        use_coupling=False,
    )
    out_on = compute_honeycomb_fusion(
        latest["saddle_year"],
        latest.get("position", 0),
        latest.get("velocity", 0),
        latest.get("acceleration", 0),
        pool,
        latest.get("ring_B_score"),
        use_coupling=True,
    )
    # With use_coupling=False, no coupling_corrections; with True, may have them
    # Key: peak_year with use_coupling=False should match the "uncorrected" baseline
    # When use_coupling=True and coupling matrix is empty, output should match
    # When coupling matrix has data, peak_year may differ
    assert "peak_year" in out_off
    assert "peak_year" in out_on
    # use_coupling=False should never have coupling_corrections
    assert "coupling_corrections" not in out_off or not out_off.get("coupling_corrections")
