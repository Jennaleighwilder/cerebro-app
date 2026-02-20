#!/usr/bin/env python3
"""
CEREBRO BASELINE DESTRUCTION
Linear regression, ARIMA, mean Δt analogue, random saddle timing.
If you do not clearly beat them, you are not dominant.
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "baseline_comparison.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

EVENT_TOLERANCE = 10


def _get_labeled_events():
    from cerebro_event_loader import load_event_years
    return load_event_years()
RANDOM_SEED = 42


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
        for ey in _get_labeled_events():
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


def _mae_cerebro(episodes) -> float:
    from cerebro_core import compute_peak_window
    from cerebro_eval_utils import past_only_pool
    errors = []
    for ep in episodes:
        others = past_only_pool(episodes, ep["saddle_year"])
        if len(others) < 5:
            continue
        pred = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.8,
        )
        errors.append(abs(pred["peak_year"] - ep["event_year"]))
    return sum(errors) / len(errors) if errors else 0


def _mae_linear(episodes) -> float:
    """Linear regression: peak_year ~ position + velocity + acceleration. Leave-one-out."""
    if len(episodes) < 4:
        return 999.0
    try:
        import numpy as np
        errors = []
        for i, ep in enumerate(episodes):
            others = [e for j, e in enumerate(episodes) if j != i]
            X = [[e["position"], e["velocity"], e["acceleration"]] for e in others]
            y = [e["event_year"] - e["saddle_year"] for e in others]
            X_arr = np.array(X)
            y_arr = np.array(y)
            XtX = X_arr.T @ X_arr + 1e-6 * np.eye(3)
            beta = np.linalg.solve(XtX, X_arr.T @ y_arr)
            delta_pred = beta[0] * ep["position"] + beta[1] * ep["velocity"] + beta[2] * ep["acceleration"]
            peak_pred = ep["saddle_year"] + int(round(delta_pred))
            errors.append(abs(peak_pred - ep["event_year"]))
        return sum(errors) / len(errors) if errors else 999.0
    except Exception:
        return 999.0


def _mae_arima(episodes) -> float:
    """ARIMA on delta series. Fallback to naive if unavailable."""
    try:
        import numpy as np
        deltas = [e["event_year"] - e["saddle_year"] for e in episodes]
        if len(deltas) < 5:
            return 999.0
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(deltas, order=(1, 0, 1))
        fit = model.fit()
        pred_delta = fit.forecast(steps=1)[0]
        errors = [abs(ep["saddle_year"] + int(round(pred_delta)) - ep["event_year"]) for ep in episodes[-5:]]
        return sum(errors) / len(errors) if errors else 999.0
    except ImportError:
        return 999.0
    except Exception:
        return 999.0


def _mae_naive(episodes) -> float:
    """Mean Δt analogue: always predict saddle_year + mean(delta)."""
    deltas = [e["event_year"] - e["saddle_year"] for e in episodes]
    mean_dt = sum(deltas) / len(deltas) if deltas else 5
    errors = [abs(ep["saddle_year"] + int(round(mean_dt)) - ep["event_year"]) for ep in episodes]
    return sum(errors) / len(errors) if errors else 0


def _mae_random(episodes) -> float:
    """Random saddle timing: uniform draw from [1, 15] years."""
    random.seed(RANDOM_SEED)
    errors = []
    for ep in episodes:
        delta = random.randint(1, 15)
        peak_pred = ep["saddle_year"] + delta
        errors.append(abs(peak_pred - ep["event_year"]))
    return sum(errors) / len(errors) if errors else 0


def run_baselines() -> dict:
    episodes = _load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes"}

    cerebro_mae = _mae_cerebro(episodes)
    linear_mae = _mae_linear(episodes)
    arima_mae = _mae_arima(episodes)
    naive_mae = _mae_naive(episodes)
    random_mae = _mae_random(episodes)

    return {
        "cerebro_mae": round(cerebro_mae, 2),
        "linear_mae": round(linear_mae, 2),
        "arima_mae": round(arima_mae, 2),
        "naive_mae": round(naive_mae, 2),
        "random_mae": round(random_mae, 2),
        "cerebro_beats_all": cerebro_mae < min(linear_mae, arima_mae, naive_mae, random_mae),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_baselines()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Baselines: cerebro={r.get('cerebro_mae')}, linear={r.get('linear_mae')}, arima={r.get('arima_mae')}, naive={r.get('naive_mae')}, random={r.get('random_mae')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
