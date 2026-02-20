#!/usr/bin/env python3
"""
CEREBRO BACKTEST — Validate peak window predictions
====================================================
Iterates known historical epochs (1900–present for US), detects saddles,
predicts windows, compares to labeled redistribution/policy events.
Output: backtest_metrics.json for UI and versioning.
Includes: pinball loss, sharpness, coverage before/after conformal.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "backtest_metrics.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"


def _pinball_loss(y_true: float, y_pred: float, q: float) -> float:
    e = y_true - y_pred
    return max(q * e, (q - 1) * e)


# Labeled events (US): year of major redistribution/policy shift
LABELED_EVENTS = [
    1933, 1935, 1965, 1981, 1994, 2008, 2020,
]
# Saddle→event mapping for validation (saddle_year -> expected event within N years)
EVENT_TOLERANCE_YEARS = 8


def run_backtest() -> dict:
    """Run backtest, return metrics dict."""
    import pandas as pd
    from cerebro_peak_window import detect_saddle_canonical, compute_peak_window

    if not CSV_PATH.exists():
        return _empty_metrics("CSV not found")

    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(100)
    if len(df) < 20:
        return _empty_metrics("Insufficient data")

    # Build analogue episodes: use phase1 saddle_score (>=2) for historical episodes
    # Canonical rule used for live peak window; backtest uses pipeline saddles
    episodes = []
    for yr, row in df.iterrows():
        v = row.get("velocity")
        a = row.get("acceleration")
        pos = row.get("clock_position_10pt")
        rb = row.get("ring_B_score")
        ss = row.get("saddle_score", 0)
        if pd.isna(v) or pd.isna(a) or pd.isna(pos):
            continue
        v, a, pos = float(v), float(a), float(pos)
        rb = float(rb) if not pd.isna(rb) else None
        is_sad = (ss >= 2) if not pd.isna(ss) else detect_saddle_canonical(pos, v, a, rb)[0]
        if not is_sad:
            continue
        # Find nearest labeled event after saddle year
        best_event = None
        best_dist = 999
        for ey in LABELED_EVENTS:
            if ey > yr:
                d = ey - yr
                if d < best_dist and d <= EVENT_TOLERANCE_YEARS:
                    best_dist = d
                    best_event = ey
        if best_event is None:
            best_event = yr + 5  # fallback
        episodes.append({
            "saddle_year": int(yr),
            "event_year": best_event,
            "position": pos,
            "velocity": v,
            "acceleration": a,
            "ring_B_score": rb,
        })

    if len(episodes) < 5:
        return _empty_metrics(f"Only {len(episodes)} saddle episodes")

    from cerebro_conformal import run_calibration
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    cal = run_calibration(episodes, alpha=0.2, interval_alpha=0.8)
    if "error" not in cal:
        with open(SCRIPT_DIR / "cerebro_data" / "conformal_calibration.json", "w") as f:
            json.dump(cal, f, indent=2)

    errors = []
    in_window_50 = 0
    in_window_80 = 0
    in_window_50_cal = 0
    in_window_80_cal = 0
    pl10_sum, pl90_sum, pl50_sum = 0.0, 0.0, 0.0
    width_80_sum = 0.0

    for ep in episodes:
        others = [e for e in episodes if e["saddle_year"] != ep["saddle_year"]]
        pred_50 = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.5, apply_conformal=False,
        )
        pred_80 = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.8, apply_conformal=False,
        )
        pred_80_cal = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.8, apply_conformal=True,
        )
        pred_50_cal = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.5, apply_conformal=True,
        )
        ey = ep["event_year"]
        dt_true = ey - ep["saddle_year"]
        err = abs(ey - pred_80["peak_year"])
        errors.append(err)
        if pred_50["window_start"] <= ey <= pred_50["window_end"]:
            in_window_50 += 1
        if pred_80["window_start"] <= ey <= pred_80["window_end"]:
            in_window_80 += 1
        if pred_50_cal["window_start"] <= ey <= pred_50_cal["window_end"]:
            in_window_50_cal += 1
        if pred_80_cal["window_start"] <= ey <= pred_80_cal["window_end"]:
            in_window_80_cal += 1
        pl10_sum += _pinball_loss(dt_true, pred_80.get("delta_p10", 0), 0.10)
        pl90_sum += _pinball_loss(dt_true, pred_80.get("delta_p90", 0), 0.90)
        pl50_sum += _pinball_loss(dt_true, pred_80.get("delta_median", 0), 0.50)
        width_80_sum += pred_80_cal["window_end"] - pred_80_cal["window_start"]

    import numpy as np
    n = len(episodes)
    errors_arr = np.array(errors)
    mae = float(np.mean(np.abs(errors_arr)))
    median_ae = float(np.median(np.abs(errors_arr)))
    worst = int(np.max(np.abs(errors_arr)))
    coverage_50 = in_window_50 / n * 100 if n else 0
    coverage_80 = in_window_80 / n * 100 if n else 0
    coverage_50_cal = in_window_50_cal / n * 100 if n else 0
    coverage_80_cal = in_window_80_cal / n * 100 if n else 0
    sharpness_80 = width_80_sum / n if n else 0

    return {
        "n_saddles_tested": n,
        "mae_years": round(mae, 2),
        "median_absolute_error_years": round(median_ae, 2),
        "worst_case_error_years": worst,
        "pinball_loss_q10": round(pl10_sum / n, 4) if n else None,
        "pinball_loss_q90": round(pl90_sum / n, 4) if n else None,
        "pinball_loss_q50": round(pl50_sum / n, 4) if n else None,
        "sharpness_80_years": round(sharpness_80, 2),
        "coverage_50": round(coverage_50, 1),
        "coverage_80": round(coverage_80, 1),
        "coverage_50_calibrated": round(coverage_50_cal, 1),
        "coverage_80_calibrated": round(coverage_80_cal, 1),
        "event_categories": ["redistribution", "policy_shift"],
        "event_library": "US 1900–present (New Deal, Great Society, Reagan, Crime Bill, 2008, 2020)",
        "stored_in": "cerebro_data/backtest_metrics.json",
        "version": 2,
    }


def _empty_metrics(reason: str) -> dict:
    return {
        "n_saddles_tested": 0,
        "mae_years": None,
        "median_absolute_error_years": None,
        "worst_case_error_years": None,
        "coverage_50": None,
        "coverage_80": None,
        "coverage_50_calibrated": None,
        "coverage_80_calibrated": None,
        "event_categories": [],
        "event_library": "",
        "stored_in": str(OUTPUT_PATH),
        "error": reason,
        "version": 2,
    }


def main():
    SCRIPT_DIR / "cerebro_data"
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    metrics = run_backtest()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Backtest: {metrics.get('n_saddles_tested', 0)} saddles, "
          f"MAE={metrics.get('mae_years')} yr, coverage_80_cal={metrics.get('coverage_80_calibrated')}%")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
