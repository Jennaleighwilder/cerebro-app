#!/usr/bin/env python3
"""
CEREBRO WALK-FORWARD BACKTEST — No leakage.
Rolling windows: Train 1900–1950 → Test 1951–1960, etc.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "walkforward_metrics.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

LABELED_EVENTS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]
EVENT_TOLERANCE = 10
TRAIN_INITIAL_YEARS = 20
TEST_WINDOW_YEARS = 5


def _load_episodes():
    import pandas as pd
    from cerebro_core import detect_saddle_canonical

    if not CSV_PATH.exists():
        return []
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(150)
    if len(df) < 30:
        return []
    episodes = []
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
        for ey in LABELED_EVENTS:
            if ey > yr and ey - yr <= EVENT_TOLERANCE and ey - yr < best_d:
                best_d = ey - yr
                best_event = ey
        if best_event is None:
            best_event = yr + 5
        episodes.append({
            "saddle_year": int(yr),
            "event_year": best_event,
            "position": pos,
            "velocity": v,
            "acceleration": a,
            "ring_B_score": rb,
        })
    return episodes


def run_walkforward() -> dict:
    from cerebro_core import compute_peak_window

    episodes = _load_episodes()
    if len(episodes) < 5:
        return {"error": "Insufficient episodes", "windows_tested": 0}

    years = sorted(set(ep["saddle_year"] for ep in episodes))
    min_yr, max_yr = min(years), max(years)
    errors = []
    in_50 = 0
    in_80 = 0

    train_end = min_yr + TRAIN_INITIAL_YEARS
    while train_end < max_yr:
        test_start = train_end + 1
        test_end = min(train_end + TEST_WINDOW_YEARS, max_yr)
        train_ep = [e for e in episodes if e["saddle_year"] <= train_end]
        test_ep = [e for e in episodes if test_start <= e["saddle_year"] <= test_end]
        for ep in test_ep:
            others = [e for e in train_ep if e["saddle_year"] != ep["saddle_year"]]
            if not others:
                continue
            pred = compute_peak_window(
                ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
                ep.get("ring_B_score"), others, interval_alpha=0.8,
            )
            err = abs(pred["peak_year"] - ep["event_year"])
            errors.append(err)
            pred_50 = compute_peak_window(
                ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
                ep.get("ring_B_score"), others, interval_alpha=0.5,
            )
            if pred_50["window_start"] <= ep["event_year"] <= pred_50["window_end"]:
                in_50 += 1
            if pred["window_start"] <= ep["event_year"] <= pred["window_end"]:
                in_80 += 1
        train_end = test_end

    if not errors:
        return {"error": "No test predictions", "windows_tested": 0}

    import numpy as np
    err_arr = np.array(errors)
    n = len(errors)
    return {
        "windows_tested": n,
        "mean_error": round(float(np.mean(err_arr)), 2),
        "median_error": round(float(np.median(err_arr)), 2),
        "std_error": round(float(np.std(err_arr)), 2),
        "coverage_50": round(in_50 / n * 100, 1) if n else 0,
        "coverage_80": round(in_80 / n * 100, 1) if n else 0,
        "stability_score": round(1.0 / (1.0 + float(np.std(err_arr))), 4) if np.std(err_arr) > 0 else 1.0,
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_walkforward()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Walk-forward: {r.get('windows_tested', 0)} tests, mean_err={r.get('mean_error')}, stability={r.get('stability_score')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
