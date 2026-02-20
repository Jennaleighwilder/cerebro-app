#!/usr/bin/env python3
"""
CEREBRO PARAMETER STABILITY MAP
Grid sweep: V_THRESH, DIST_VEL_WEIGHT, DIST_ACC_WEIGHT
Low variance = robust architecture. High variance = fragile overfit.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "parameter_stability.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

LABELED_EVENTS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]
EVENT_TOLERANCE = 10

# Grid (cannot modify core; we pass vel_weight, acc_weight to compute_peak_window)
# V_THRESH is in core - we sweep distance weights and report sensitivity
V_THRESH_VALS = [0.10, 0.15, 0.20]  # for reference; core uses 0.15
DIST_VEL_VALS = [50, 100, 200]
DIST_ACC_VALS = [1000, 2500, 5000]


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


def _mae_at_params(episodes, vel_weight: float, acc_weight: float) -> tuple[float, float]:
    from cerebro_core import compute_peak_window
    errors = []
    in_80 = 0
    for ep in episodes:
        others = [e for e in episodes if e["saddle_year"] != ep["saddle_year"]]
        if not others:
            continue
        pred = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.8,
            vel_weight=vel_weight, acc_weight=acc_weight,
        )
        errors.append(abs(pred["peak_year"] - ep["event_year"]))
        if pred["window_start"] <= ep["event_year"] <= pred["window_end"]:
            in_80 += 1
    mae = sum(errors) / len(errors) if errors else 0
    cov = in_80 / len(errors) if errors else 0
    return mae, cov


def run_stability() -> dict:
    episodes = _load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes"}

    mae_vals = []
    cov_vals = []
    for vw in DIST_VEL_VALS:
        for aw in DIST_ACC_VALS:
            mae, cov = _mae_at_params(episodes, vw, aw)
            mae_vals.append(mae)
            cov_vals.append(cov)

    import numpy as np
    mae_arr = np.array(mae_vals)
    cov_arr = np.array(cov_vals)
    mae_var = float(np.var(mae_arr))
    cov_var = float(np.var(cov_arr))

    return {
        "mae_surface_variance": round(mae_var, 4),
        "coverage_surface_variance": round(cov_var, 4),
        "mae_mean": round(float(np.mean(mae_arr)), 2),
        "mae_std": round(float(np.std(mae_arr)), 2),
        "coverage_mean": round(float(np.mean(cov_arr)), 2),
        "grid_points": len(mae_vals),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_stability()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Parameter stability: mae_var={r.get('mae_surface_variance')}, cov_var={r.get('coverage_surface_variance')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
