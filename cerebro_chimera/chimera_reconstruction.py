#!/usr/bin/env python3
"""
CHIMERA RECONSTRUCTION — Backward structural replay
====================================================
Year-by-year replay using past-only data. Records MAE, coverage, n_eff, disagreement.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_reconstruction.json"
MIN_TRAIN = 5


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def run_reconstruction() -> dict:
    """Walk-forward reconstruction: for each year Y, predict using past-only pool."""
    from cerebro_calibration import _load_episodes
    from cerebro_core import compute_peak_window
    from cerebro_sister_engine import sister_predict
    from cerebro_honeycomb import compute_honeycomb_fusion

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 2:
        return {"error": "Insufficient episodes", "version": 1}

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    mae_list = []
    coverage_list = []
    records = []
    start_year = min(e.get("saddle_year", 9999) for e in sorted_ep)
    last_event_year = max(e.get("event_year", 0) for e in sorted_ep)

    for ep in sorted_ep:
        Y = ep.get("saddle_year")
        if Y is None or Y >= last_event_year - 1:
            continue
        pool = _past_only_pool(episodes, Y)
        if len(pool) < MIN_TRAIN:
            continue

        try:
            core = compute_peak_window(
                Y, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                ep.get("ring_B_score"), pool, interval_alpha=0.8,
            )
            sis = sister_predict(Y, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0), pool)
            honey = compute_honeycomb_fusion(
                Y, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                pool, ep.get("ring_B_score"), sim_summary=None, shift_dict={"confidence_modifier": 1.0},
            )
        except Exception:
            continue

        event_yr = ep.get("event_year", Y + 5)
        pred_peak = honey.get("peak_year", core.get("peak_year"))
        ws, we = honey.get("window_start"), honey.get("window_end")
        hit = ws <= event_yr <= we if ws and we else False

        peaks = [core.get("peak_year"), sis.get("peak_year"), honey.get("peak_year")]
        peaks = [p for p in peaks if p is not None]
        disagreement_std = float(np.std(peaks)) if len(peaks) > 1 else 0.0
        n_eff = core.get("analogue_count", len(pool))

        mae_list.append(abs(pred_peak - event_yr))
        coverage_list.append(1.0 if hit else 0.0)
        records.append({
            "saddle_year": Y,
            "event_year": event_yr,
            "predicted_peak": pred_peak,
            "error": abs(pred_peak - event_yr),
            "interval_hit": hit,
            "n_eff": n_eff,
            "disagreement_std": round(disagreement_std, 2),
        })

    if not records:
        return {"error": "No valid predictions", "version": 1}

    mae_trajectory = [r["error"] for r in records]
    coverage_trajectory = [r["interval_hit"] for r in records]
    mae_mean = sum(mae_list) / len(mae_list)
    coverage_mean = sum(coverage_list) / len(coverage_list)

    return {
        "version": 1,
        "min_train": MIN_TRAIN,
        "n_used": len(records),
        "start_year": start_year,
        "last_event_year": last_event_year,
        "mae_mean": round(mae_mean, 3),
        "coverage_80_mean": round(coverage_mean, 4),
        "mae_trajectory": [round(x, 2) for x in mae_trajectory],
        "coverage_trajectory": coverage_trajectory,
        "records": records[-20:],
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_reconstruction()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera reconstruction: MAE={r.get('mae_mean')}, coverage={r.get('coverage_80_mean')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
