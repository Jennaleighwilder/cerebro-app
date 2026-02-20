#!/usr/bin/env python3
"""
CEREBRO CONFIDENCE CALIBRATION — Walk-forward (past-only analogues)
==================================================================
Bin predictions by confidence decile, compute empirical hit rate.
Each episode's prediction uses only past episodes. No future leakage.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "calibration_curve.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

EVENT_TOLERANCE = 10
MIN_TRAIN = 5


def _get_labeled_events():
    from cerebro_event_loader import load_event_years
    return load_event_years()


def _load_episodes():
    import pandas as pd
    from cerebro_core import detect_saddle_canonical

    if not CSV_PATH.exists():
        return []
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(100)
    if len(df) < 20:
        return []
    raw = []
    for yr, row in df.iterrows():
        v, a, pos = row.get("velocity"), row.get("acceleration"), row.get("clock_position_10pt")
        rb = row.get("ring_B_score")
        if any(x is None or (hasattr(x, "__float__") and pd.isna(x)) for x in [v, a, pos]):
            continue
        v, a, pos = float(v), float(a), float(pos)
        rb = float(rb) if rb is not None and not pd.isna(rb) else None
        is_sad, _ = detect_saddle_canonical(pos, v, a, rb)
        if not is_sad:
            continue
        best_event = None
        best_d = 999
        for ey in _get_labeled_events():
            if ey > yr and ey - yr <= EVENT_TOLERANCE and ey - yr < best_d:
                best_d = ey - yr
                best_event = ey
        if best_event is None:
            best_event = yr + 5
        raw.append({
            "saddle_year": int(yr),
            "event_year": best_event,
            "position": pos,
            "velocity": v,
            "acceleration": a,
            "ring_B_score": rb,
        })
    return raw


def run_calibration() -> dict:
    from cerebro_eval_utils import walkforward_predictions

    raw = _load_episodes()
    if len(raw) < 10:
        return {"error": "Insufficient episodes", "bins": [], "method": "walkforward"}

    episodes = walkforward_predictions(raw, interval_alpha=0.8, min_train=MIN_TRAIN)
    if len(episodes) < 10:
        return {"error": "Insufficient walk-forward predictions", "bins": [], "method": "walkforward"}

    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        subset = [e for e in episodes if lo <= e["confidence"] < hi]
        if not subset:
            bins.append({"conf_mid": (lo + hi) / 2, "empirical_hit_rate": None, "n": 0})
            continue
        hit_rate = sum(1 for e in subset if e["hit"]) / len(subset)
        bins.append({"conf_mid": round((lo + hi) / 2, 2), "empirical_hit_rate": round(hit_rate, 4), "n": len(subset)})

    brier = sum((e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e in episodes) / len(episodes)
    in_80 = sum(1 for e in episodes if e["hit"])
    coverage_80 = in_80 / len(episodes)

    return {
        "bins": bins,
        "brier_score": round(brier, 4),
        "n_episodes": len(episodes),
        "n_used": len(episodes),
        "min_train": MIN_TRAIN,
        "method": "walkforward",
        "coverage_80": round(coverage_80, 4),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_calibration()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    # Write backtest_metrics.json with walk-forward labels (no silent overwrite)
    bt_path = SCRIPT_DIR / "cerebro_data" / "backtest_metrics.json"
    if "error" not in r and r.get("method") == "walkforward":
        bt = {
            "brier_walkforward": r.get("brier_score"),
            "coverage_80_walkforward": r.get("coverage_80"),
            "n_used": r.get("n_used"),
            "min_train": r.get("min_train"),
            "method": "walkforward",
            "stored_in": str(bt_path),
        }
        with open(bt_path, "w") as f:
            json.dump(bt, f, indent=2)
    print(f"Calibration: Brier={r.get('brier_score')}, bins={len(r.get('bins', []))}, method={r.get('method')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
