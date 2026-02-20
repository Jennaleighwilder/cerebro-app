#!/usr/bin/env python3
"""
CEREBRO HONEYCOMB CONFORMAL — Calibrated 80% interval
=====================================================
Walk-forward nonconformity scores, quantile radius, apply to Honeycomb window.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "honeycomb_conformal.json"
MIN_TRAIN = 5
TARGET_ALPHA = 0.8


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def run_conformal() -> dict:
    """Walk-forward honeycomb intervals, compute s_hat, empirical coverage."""
    from cerebro_calibration import _load_episodes
    from cerebro_honeycomb import compute_honeycomb_fusion

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "version": 1}

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    scores = []

    for ep in sorted_ep:
        Y = ep.get("saddle_year")
        if Y is None:
            continue
        pool = _past_only_pool(episodes, Y)
        if len(pool) < MIN_TRAIN:
            continue

        try:
            honey = compute_honeycomb_fusion(
                Y,
                ep.get("position", 0),
                ep.get("velocity", 0),
                ep.get("acceleration", 0),
                pool,
                ep.get("ring_B_score"),
                sim_summary=None,
                shift_dict={"confidence_modifier": 1.0},
            )
        except Exception:
            continue

        ws = honey.get("window_start")
        we = honey.get("window_end")
        event_yr = ep.get("event_year", Y + 5)

        # Nonconformity: distance outside interval
        s_i = max(0, ws - event_yr, event_yr - we)
        scores.append(s_i)

    if len(scores) < 5:
        return {"error": "Too few scores", "version": 1}

    # s_hat = quantile at alpha (higher convention: ceiling)
    q_idx = int(np.ceil(TARGET_ALPHA * len(scores))) - 1
    q_idx = max(0, min(q_idx, len(scores) - 1))
    sorted_s = sorted(scores)
    s_hat = float(sorted_s[q_idx])

    # Empirical coverage (before applying s_hat)
    hits = sum(1 for s in scores if s == 0)
    empirical = hits / len(scores)

    return {
        "target_coverage": TARGET_ALPHA,
        "min_train": MIN_TRAIN,
        "n_used": len(scores),
        "s_hat": round(s_hat, 2),
        "empirical_coverage": round(empirical, 2),
        "method": "walkforward_conformal_interval",
        "version": 1,
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_conformal()
    if r.get("error"):
        print(f"Honeycomb conformal: {r['error']}")
        return 1
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Honeycomb conformal: s_hat={r['s_hat']}, emp_cov={r['empirical_coverage']} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
