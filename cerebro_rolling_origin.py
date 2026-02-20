#!/usr/bin/env python3
"""
CEREBRO ROLLING-ORIGIN VALIDATION
Full walk-forward across every decade. Not one split.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "rolling_origin_metrics.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

LABELED_EVENTS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]
EVENT_TOLERANCE = 10
MIN_TRAIN = 5
ORIGIN_STEP_YEARS = 5


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


def run_rolling_origin() -> dict:
    from cerebro_core import compute_peak_window

    episodes = _load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes", "decade_rows": [], "overall_mae": None, "overall_coverage_80": None, "drift_flags": []}

    years = sorted(set(ep["saddle_year"] for ep in episodes))
    min_yr, max_yr = min(years), max(years)

    decade_rows = []
    all_errors = []
    drift_flags = []

    origin = min_yr + 20
    prev_mae = None
    while origin < max_yr:
        train_ep = [e for e in episodes if e["saddle_year"] < origin]
        test_ep = [e for e in episodes if origin <= e["saddle_year"] < origin + ORIGIN_STEP_YEARS]
        if len(train_ep) < MIN_TRAIN or not test_ep:
            origin += ORIGIN_STEP_YEARS
            continue
        dec_errors = []
        dec_in_80 = 0
        for ep in test_ep:
            pred = compute_peak_window(
                ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
                ep.get("ring_B_score"), train_ep, interval_alpha=0.8,
            )
            err = abs(pred["peak_year"] - ep["event_year"])
            dec_errors.append(err)
            all_errors.append(err)
            if pred["window_start"] <= ep["event_year"] <= pred["window_end"]:
                dec_in_80 += 1
        dec_mae = sum(dec_errors) / len(dec_errors) if dec_errors else 0
        dec_cov = dec_in_80 / len(dec_errors) if dec_errors else 0
        decade_rows.append({
            "origin_year": origin,
            "n_predictions": len(dec_errors),
            "mae": round(dec_mae, 2),
            "coverage_80": round(dec_cov, 2),
        })
        # Drift: MAE jumps >2x from previous decade
        if prev_mae is not None and prev_mae > 0 and dec_mae > prev_mae * 2:
            drift_flags.append({"origin_year": origin, "reason": "mae_doubled", "prev_mae": prev_mae, "current_mae": dec_mae})
        prev_mae = dec_mae
        origin += ORIGIN_STEP_YEARS

    overall_mae = sum(all_errors) / len(all_errors) if all_errors else 0
    in_80_total = sum(int(r["coverage_80"] * r["n_predictions"]) for r in decade_rows)
    n_total = sum(r["n_predictions"] for r in decade_rows)
    overall_coverage_80 = in_80_total / n_total if n_total else 0

    return {
        "decade_rows": decade_rows,
        "overall_mae": round(overall_mae, 2),
        "overall_coverage_80": round(overall_coverage_80, 2),
        "drift_flags": drift_flags,
        "origins_tested": len(decade_rows),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_rolling_origin()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Rolling-origin: origins={r.get('origins_tested')}, overall_mae={r.get('overall_mae')}, overall_coverage_80={r.get('overall_coverage_80')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
