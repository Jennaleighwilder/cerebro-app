#!/usr/bin/env python3
"""
CEREBRO CONFORMAL CALIBRATION — Time-to-event interval calibration
====================================================================
Conformal prediction for calibrated prediction intervals.
Inputs: backtest episodes with true dt, predicted q_lo/q_hi.
Nonconformity: s_i = max(q_lo - dt, dt - q_hi, 0)
s_hat = quantile(s, ceil((n+1)(1-alpha))/n) for target coverage 1-alpha
Calibrated interval: [q_lo - s_hat, q_hi + s_hat]
"""

import json
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "conformal_calibration.json"


def nonconformity_score(dt_true: float, q_lo: float, q_hi: float) -> float:
    """s_i = max(q_lo - dt, dt - q_hi, 0) — how far outside interval."""
    if q_lo <= dt_true <= q_hi:
        return 0.0
    if dt_true < q_lo:
        return q_lo - dt_true
    return dt_true - q_hi


def compute_s_hat(scores: list[float], alpha: float) -> float:
    """Quantile at ceil((n+1)(1-alpha))/n for finite-sample correction."""
    if not scores:
        return 0.0
    n = len(scores)
    # 1-alpha coverage: we want P(s <= s_hat) >= 1-alpha
    # s_hat = Q(ceil((n+1)(1-alpha))/n)
    q_level = (n + 1) * (1 - alpha) / n
    q_level = min(1.0, max(0.0, q_level))
    sorted_s = sorted(scores)
    idx = max(0, min(int(q_level * n), n - 1))
    return sorted_s[idx]


def run_calibration(
    episodes: list[dict],
    alpha: float = 0.2,
    interval_alpha: Optional[float] = None,
) -> dict:
    """
    Run conformal calibration on backtest episodes.
    episodes: list of {saddle_year, event_year, position, velocity, acceleration, ring_B_score}
    alpha: miscoverage (0.2 = 80% coverage)
    interval_alpha: 0.8 for p10-p90, 0.5 for p25-p75
    """
    from cerebro_peak_window import compute_peak_window

    interval_alpha = interval_alpha or 0.8
    q_lo, q_hi = (0.10, 0.90) if interval_alpha == 0.8 else (0.25, 0.75)

    scores = []
    for ep in episodes:
        others = [e for e in episodes if e["saddle_year"] != ep["saddle_year"]]
        pred = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=interval_alpha,
        )
        dt_true = ep["event_year"] - ep["saddle_year"]
        p_lo = pred.get("delta_p10" if interval_alpha == 0.8 else "delta_p25")
        p_hi = pred.get("delta_p90" if interval_alpha == 0.8 else "delta_p75")
        if p_lo is None or p_hi is None:
            continue
        s = nonconformity_score(dt_true, float(p_lo), float(p_hi))
        scores.append(s)

    if len(scores) < 5:
        return {"error": "Insufficient episodes", "n_samples": len(scores)}

    s_hat = compute_s_hat(scores, alpha)
    return {
        "alpha": alpha,
        "s_hat": round(s_hat, 4),
        "n_samples": len(scores),
        "timestamp": int(time.time()),
        "version": 1,
        "interval_alpha": interval_alpha,
        "coverage_target": 1 - alpha,
    }


def load_calibration() -> Optional[dict]:
    """Load conformal calibration artifact if exists."""
    if not OUTPUT_PATH.exists():
        return None
    try:
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def apply_conformal(
    window_start: int,
    window_end: int,
    delta_p_lo: float,
    delta_p_hi: float,
    calibration: Optional[dict] = None,
) -> tuple[int, int, float, bool]:
    """
    Apply conformal widening to interval.
    Returns (new_window_start, new_window_end, s_hat_used, was_applied).
    """
    if calibration is None:
        calibration = load_calibration()
    if not calibration or "s_hat" not in calibration:
        return window_start, window_end, 0.0, False
    s_hat = float(calibration["s_hat"])
    if s_hat <= 0:
        return window_start, window_end, 0.0, False
    # Calibrated: [q_lo - s_hat, q_hi + s_hat] in delta space
    # window_start = now_year + round(delta_p_lo), so we subtract s_hat from delta
    widen = int(round(s_hat))
    return (
        window_start - widen,
        window_end + widen,
        s_hat,
        True,
    )
