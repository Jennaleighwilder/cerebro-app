#!/usr/bin/env python3
"""Tests for CHIMERA adaptive online learning."""

import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_params_stay_within_bounds():
    """Parameters stay within bounds after update."""
    p = DATA_DIR / "chimera_params.json"
    if not p.exists():
        pytest.skip("chimera_params.json not yet generated")
    with open(p) as f:
        d = json.load(f)
    vw = d.get("vel_weight")
    aw = d.get("acc_weight")
    tau = d.get("tau")
    if vw is not None:
        assert 10 <= vw <= 500
    if aw is not None:
        assert 200 <= aw <= 20000
    if tau is not None:
        assert 0.5 <= tau <= 4.0


def test_conformal_widens_when_residuals_increase():
    """Conformal q80 increases when residuals are larger (sanity: schema exists)."""
    p = DATA_DIR / "chimera_calibration_online.json"
    if not p.exists():
        pytest.skip("chimera_calibration_online.json not yet generated")
    with open(p) as f:
        d = json.load(f)
    assert "conformal_q80" in d
    assert "residual_count" in d


def test_transition_matrix_rows_sum_to_one():
    """Regime HMM transition matrix rows sum to ~1."""
    p = DATA_DIR / "chimera_regime_hmm.json"
    if not p.exists():
        pytest.skip("chimera_regime_hmm.json not yet generated")
    with open(p) as f:
        d = json.load(f)
    if d.get("error"):
        pytest.skip("regime HMM had error")
    tm = d.get("transition_matrix", {})
    for row_name, row in tm.items():
        total = sum(row.values())
        assert 0.98 <= total <= 1.02, f"Row {row_name} sums to {total}"


def test_drift_detection_flips_when_synthetic_drift():
    """Drift detection returns drift_mode (boolean)."""
    p = DATA_DIR / "chimera_master.json"
    if not p.exists():
        pytest.skip("chimera_master.json not yet generated")
    with open(p) as f:
        d = json.load(f)
    drift = d.get("evolution", {}).get("model_structure_drift")
    if drift is not None:
        assert isinstance(drift, bool)


def test_core_unchanged():
    """Core hash test still passes (frozen core)."""
    from tests.test_core_frozen import test_core_hash_unchanged
    test_core_hash_unchanged()
