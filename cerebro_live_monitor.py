#!/usr/bin/env python3
"""
CEREBRO ROLLING LIVE PERFORMANCE TRACKER
Track active predictions, rolling MAE, drift detection.
"""

import json
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "live_monitor.json"
HISTORY_PATH = SCRIPT_DIR / "cerebro_data" / "live_predictions_history.json"
WINDOW_MAE = 10


def _load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(history: list) -> None:
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history[-100:], f, indent=2)


def record_prediction(
    saddle_year: int,
    predicted_peak: int,
    window_start: int,
    window_end: int,
    actual_event_year: Optional[int] = None,
) -> None:
    """Record a prediction. Call with actual_event_year when resolved."""
    history = _load_history()
    entry = {
        "saddle_year": saddle_year,
        "predicted_peak": predicted_peak,
        "window_start": window_start,
        "window_end": window_end,
        "actual_event_year": actual_event_year,
        "resolved": actual_event_year is not None,
    }
    history.append(entry)
    _save_history(history)


def compute_monitor() -> dict:
    """Rolling MAE, drift detection, active count."""
    history = _load_history()
    resolved = [h for h in history if h.get("resolved") and h.get("actual_event_year") is not None]
    active = len([h for h in history if not h.get("resolved")])

    errors = [abs(h["predicted_peak"] - h["actual_event_year"]) for h in resolved]
    rolling = errors[-WINDOW_MAE:] if len(errors) >= WINDOW_MAE else errors
    rolling_mae = sum(rolling) / len(rolling) if rolling else None

    # Drift: compare first half to second half of errors
    drift_detected = False
    if len(errors) >= 10:
        mid = len(errors) // 2
        mae1 = sum(errors[:mid]) / mid
        mae2 = sum(errors[mid:]) / (len(errors) - mid)
        if abs(mae2 - mae1) > 1.0:
            drift_detected = True

    return {
        "predictions_active": active,
        "predictions_resolved": len(resolved),
        "rolling_mae_last_10": round(rolling_mae, 2) if rolling_mae is not None else None,
        "drift_detected": drift_detected,
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = compute_monitor()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Live: active={r['predictions_active']}, rolling_mae={r['rolling_mae_last_10']}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
