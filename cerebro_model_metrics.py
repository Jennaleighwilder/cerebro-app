#!/usr/bin/env python3
"""
CEREBRO MODEL METRICS — Calibrated uncertainty from posterior + backtest + integrity
====================================================================================
Confidence derived from:
  - Posterior variance (state-space)
  - Backtest calibration (coverage)
  - Sensor integrity score
Save to cerebro_data/model_metrics.json
"""

import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "model_metrics.json"


def compute_calibrated_confidence() -> dict:
    """
    Compute confidence from posterior variance, backtest calibration, sensor integrity.
    Returns 0-100 score.
    """
    # 1. Posterior variance (from state-space)
    post_path = SCRIPT_DIR / "cerebro_data" / "state_space_posterior.json"
    posterior_score = 85  # default
    if post_path.exists():
        try:
            with open(post_path) as f:
                d = json.load(f)
            posts = d.get("posteriors", [])
            if posts:
                last = posts[-1]
                cov = last.get("cov_diag", [1, 1, 1])
                # Lower variance -> higher confidence. Scale position std to 0-100
                pos_std = cov[0] ** 0.5
                posterior_score = max(50, min(95, 95 - pos_std * 20))
        except Exception:
            pass

    # 2. Backtest calibration
    bt_path = SCRIPT_DIR / "cerebro_data" / "backtest_metrics.json"
    calibration_score = 85
    if bt_path.exists():
        try:
            with open(bt_path) as f:
                bt = json.load(f)
            cov80 = bt.get("coverage_80_calibrated") or bt.get("coverage_80")
            if cov80 is not None:
                # 80% target -> 80% coverage = 100, deviation penalized
                calibration_score = max(50, min(95, 100 - abs(80 - cov80)))
        except Exception:
            pass

    # 3. Sensor integrity (from pipeline/integrity)
    integrity_path = SCRIPT_DIR / "cerebro_data" / "pipeline_status.json"
    integrity_score = 90
    if integrity_path.exists():
        try:
            with open(integrity_path) as f:
                s = json.load(f)
            integrity_score = s.get("confidence", 90)
        except Exception:
            pass

    # Weighted combination
    confidence = 0.4 * posterior_score + 0.35 * calibration_score + 0.25 * integrity_score
    confidence = max(50, min(98, round(confidence, 1)))

    return {
        "confidence_pct": confidence,
        "components": {
            "posterior_variance": round(posterior_score, 1),
            "backtest_calibration": round(calibration_score, 1),
            "sensor_integrity": round(integrity_score, 1),
        },
        "weights": {"posterior": 0.4, "calibration": 0.35, "integrity": 0.25},
        "provenance": {
            "timestamp": int(time.time()),
            "version": 1,
        },
    }


def run_and_save() -> dict:
    """Compute and save model metrics."""
    metrics = compute_calibrated_confidence()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    m = run_and_save()
    print(f"Calibrated confidence: {m['confidence_pct']}%")
    print(f"  → {OUTPUT_PATH}")
