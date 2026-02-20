#!/usr/bin/env python3
"""
CHIMERA ONLINE — Wrapper for adaptive learning pipeline
=======================================================
Runs weights, calibration, regime HMM, drift. Respects drift_mode (freeze when True).
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def run_online_learning() -> dict:
    """
    Run online updates: weights, conformal, regime HMM, drift.
    If drift_mode: skip weight updates, still run drift detection and conformal.
    """
    from cerebro_chimera import chimera_drift
    from cerebro_chimera import chimera_store

    drift_report = chimera_drift.detect_drift()
    drift_mode = drift_report.get("drift_mode", False)

    results = {"drift": drift_report}

    if not drift_mode:
        from cerebro_chimera import chimera_weights_online
        try:
            w = chimera_weights_online.update_weights()
            results["weights"] = w
        except Exception as e:
            results["weights"] = {"error": str(e)}
    else:
        results["weights"] = {"updated": False, "reason": "drift_mode"}

    from cerebro_chimera import chimera_calibration_online
    try:
        cal = chimera_calibration_online.update_conformal()
        results["calibration"] = cal
    except Exception as e:
        results["calibration"] = {"error": str(e)}

    from cerebro_chimera import chimera_regime_hmm
    try:
        regime = chimera_regime_hmm.update_transition_matrix()
        results["regime_hmm"] = regime
    except Exception as e:
        results["regime_hmm"] = {"error": str(e)}

    return results
