#!/usr/bin/env python3
"""
CEREBRO CONFIDENCE CALIBRATION — No vibes.
Bin predictions by confidence decile, compute empirical hit rate.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "calibration_curve.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

LABELED_EVENTS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]
EVENT_TOLERANCE = 10


def _load_episodes():
    import pandas as pd
    from cerebro_core import detect_saddle_canonical, compute_peak_window

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
        for ey in LABELED_EVENTS:
            if ey > yr and ey - yr <= EVENT_TOLERANCE and ey - yr < best_d:
                best_d = ey - yr
                best_event = ey
        if best_event is None:
            best_event = yr + 5
        raw.append({"saddle_year": int(yr), "event_year": best_event, "position": pos, "velocity": v, "acceleration": a, "ring_B_score": rb})

    episodes = []
    for ep in raw:
        others = [e for e in raw if e["saddle_year"] != ep["saddle_year"]]
        pred = compute_peak_window(ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.8)
        hit = pred["window_start"] <= ep["event_year"] <= pred["window_end"]
        episodes.append({**ep, "confidence": pred["confidence_pct"] / 100.0, "hit": hit})
    return episodes


def run_calibration() -> dict:
    episodes = _load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes", "bins": []}

    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        subset = [e for e in episodes if lo <= e["confidence"] < hi]
        if not subset:
            bins.append({"conf_mid": (lo + hi) / 2, "empirical_hit_rate": None, "n": 0})
            continue
        hit_rate = sum(1 for e in subset if e["hit"]) / len(subset)
        bins.append({"conf_mid": round((lo + hi) / 2, 2), "empirical_hit_rate": round(hit_rate, 4), "n": len(subset)})

    # Brier score: mean((confidence - hit)^2)
    brier = sum((e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e in episodes) / len(episodes)

    return {
        "bins": bins,
        "brier_score": round(brier, 4),
        "n_episodes": len(episodes),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_calibration()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Calibration: Brier={r.get('brier_score')}, bins={len(r.get('bins', []))}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
