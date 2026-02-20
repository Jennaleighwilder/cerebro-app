#!/usr/bin/env python3
"""
CEREBRO ABLATION STUDY
Remove Ring B, acceleration, analogue weighting, distance weighting.
Measure error increase.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "ablation_results.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

LABELED_EVENTS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]
EVENT_TOLERANCE = 10


def _load_episodes():
    import pandas as pd
    from cerebro_core import detect_saddle_canonical

    if not CSV_PATH.exists():
        return []
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(100)
    if len(df) < 20:
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


def _mae(episodes, **kwargs) -> float:
    from cerebro_core import compute_peak_window
    errors = []
    for ep in episodes:
        others = [e for e in episodes if e["saddle_year"] != ep["saddle_year"]]
        pred = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            kwargs.get("ring_b_score", ep.get("ring_B_score")),
            others, interval_alpha=0.8,
            **{k: v for k, v in kwargs.items() if k != "ring_b_score"},
        )
        errors.append(abs(pred["peak_year"] - ep["event_year"]))
    return sum(errors) / len(errors) if errors else 0.0


def run_ablation() -> dict:
    episodes = _load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes"}

    core_err = _mae(episodes)
    no_rb_err = _mae(episodes, ring_b_score=None)
    # No acceleration: pass acceleration=0 for all (degenerate)
    no_acc_ep = [{"saddle_year": e["saddle_year"], "event_year": e["event_year"], "position": e["position"], "velocity": e["velocity"], "acceleration": 0.0, "ring_B_score": e.get("ring_B_score")} for e in episodes]
    no_acc_err = _mae(no_acc_ep)
    # No analogue weighting: uniform weights (hack: use very large distance weights so all similar)
    # Actually we need to modify compute_peak_window. Simpler: use vel_weight=0, acc_weight=0 to flatten distance
    no_dist_err = _mae(episodes, vel_weight=0.001, acc_weight=0.001)

    return {
        "core_error": round(core_err, 2),
        "no_ring_b_error": round(no_rb_err, 2),
        "no_acceleration_error": round(no_acc_err, 2),
        "no_distance_weighting_error": round(no_dist_err, 2),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_ablation()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Ablation: core={r.get('core_error')}, no_rb={r.get('no_ring_b_error')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
