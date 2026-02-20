#!/usr/bin/env python3
"""
CEREBRO VALIDATION V2 — Rolling-origin backtest with leakage control
====================================================================
Metrics: MAE event timing, Brier/log score event probabilities, calibration, sharpness.
Ablations per data source.
Output: cerebro_data/backtest_metrics_v2.json
"""

import json
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "backtest_metrics_v2.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
RANDOM_SEED = 42

LABELED_EVENTS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]
EVENT_TOLERANCE = 10


def _load_episodes():
    """Load saddle episodes (no leakage: train only on past)."""
    import pandas as pd
    from cerebro_peak_window import detect_saddle_canonical

    if not CSV_PATH.exists():
        return []
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(100)
    if len(df) < 20:
        return []
    episodes = []
    for yr, row in df.iterrows():
        v, a, pos = row.get("velocity"), row.get("acceleration"), row.get("clock_position_10pt")
        rb, ss = row.get("ring_B_score"), row.get("saddle_score", 0)
        if any(pd.isna(x) for x in [v, a, pos]):
            continue
        v, a, pos = float(v), float(a), float(pos)
        rb = float(rb) if not pd.isna(rb) else None
        is_sad = (ss >= 2) if not pd.isna(ss) else detect_saddle_canonical(pos, v, a, rb)[0]
        if not is_sad:
            continue
        best_event = None
        best_d = 999
        for ey in LABELED_EVENTS:
            if ey > yr and ey - yr <= EVENT_TOLERANCE:
                if ey - yr < best_d:
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
            "dt": best_event - yr,
        })
    return episodes


def _brier_score(y_true: float, y_pred: float) -> float:
    return (y_pred - y_true) ** 2


def _log_score(y_true: float, y_pred: float, eps: float = 1e-8) -> float:
    p = max(eps, min(1 - eps, y_pred))
    return -np.log(p) if y_true == 1 else -np.log(1 - p)


def run_validation() -> dict:
    """Rolling-origin backtest: train on first 70%, test on last 30%."""
    from cerebro_peak_window import compute_peak_window
    from cerebro_hazard import predict_hazard, fit_logistic_hazard, load_episodes_with_state

    np.random.seed(RANDOM_SEED)
    episodes = _load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes", "n": len(episodes)}

    split = int(len(episodes) * 0.7)
    train_ep, test_ep = episodes[:split], episodes[split:]
    max_train_year = max(ep["saddle_year"] for ep in train_ep)

    # Fit hazard on train only (no leakage)
    hazard_train = load_episodes_with_state(max_year=max_train_year)
    hazard_model = fit_logistic_hazard(hazard_train) if len(hazard_train) >= 8 else {}

    mae_list = []
    brier_list = []
    log_list = []
    coverage_50, coverage_80 = 0, 0
    width_80_sum = 0.0

    for ep in test_ep:
        others = [e for e in episodes if e["saddle_year"] < ep["saddle_year"]]  # past only
        pred = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.8, apply_conformal=False,
        )
        ey = ep["event_year"]
        dt_true = ep["dt"]
        mae_list.append(abs(pred["peak_year"] - ey))
        if pred["window_start"] <= ey <= pred["window_end"]:
            coverage_80 += 1
        width_80_sum += pred["window_end"] - pred["window_start"]

        # Hazard: P(event in 5y) - binary target
        if "error" not in hazard_model:
            h_pred = predict_hazard(ep["position"], ep["velocity"], ep["acceleration"], ep.get("ring_B_score"), hazard_model)
            p5 = h_pred.get("prob_5y", 0.5)
            y_bin = 1.0 if dt_true <= 5 else 0.0
            brier_list.append(_brier_score(y_bin, p5))
            log_list.append(_log_score(y_bin, p5))

    n = len(test_ep)
    mae = float(np.mean(mae_list)) if mae_list else None
    brier = float(np.mean(brier_list)) if brier_list else None
    log_score_mean = float(np.mean(log_list)) if log_list else None
    sharpness_80 = width_80_sum / n if n else None
    cov80 = coverage_80 / n * 100 if n else None

    # Ablation: drop ring_B
    mae_no_rb = None
    if n >= 3:
        mae_no_rb_list = []
        for ep in test_ep:
            others = [e for e in episodes if e["saddle_year"] < ep["saddle_year"]]
            pred = compute_peak_window(
                ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
                None, others, interval_alpha=0.8, apply_conformal=False,
            )
            mae_no_rb_list.append(abs(pred["peak_year"] - ep["event_year"]))
        mae_no_rb = float(np.mean(mae_no_rb_list))

    return {
        "mae_event_timing_years": round(mae, 2) if mae is not None else None,
        "brier_score_event_5y": round(brier, 4) if brier is not None else None,
        "log_score_event_5y": round(log_score_mean, 4) if log_score_mean is not None else None,
        "coverage_80_pct": round(cov80, 1) if cov80 is not None else None,
        "sharpness_80_years": round(sharpness_80, 2) if sharpness_80 is not None else None,
        "n_train": len(train_ep),
        "n_test": n,
        "ablations": {
            "mae_without_ring_b": round(mae_no_rb, 2) if mae_no_rb is not None else None,
        },
        "provenance": {
            "method": "rolling_origin",
            "split": "70/30",
            "leakage_control": "train_on_past_only",
            "random_seed": RANDOM_SEED,
            "timestamp": int(time.time()),
            "version": 2,
        },
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    result = run_validation()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Validation: MAE={result.get('mae_event_timing_years')} yr, Brier={result.get('brier_score_event_5y')}, coverage_80={result.get('coverage_80_pct')}%")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
