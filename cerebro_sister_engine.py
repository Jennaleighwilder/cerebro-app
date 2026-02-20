#!/usr/bin/env python3
"""
CEREBRO SISTER ENGINE — Competing peak estimator (Huber regression on Δt)
=========================================================================
Uses robust regression on Δt vs state. Cross-checks core without touching it.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "sister_latest.json"
MIN_TRAIN = 5


def _feature_vec(ep: dict) -> np.ndarray:
    """Features: [position, velocity, acceleration, |velocity|, sign(velocity), sign(acceleration)]."""
    pos = float(ep.get("position", 0))
    vel = float(ep.get("velocity", 0))
    acc = float(ep.get("acceleration", 0))
    vel_abs = abs(vel)
    sign_vel = 1.0 if vel > 0 else (-1.0 if vel < 0 else 0.0)
    sign_acc = 1.0 if acc > 0 else (-1.0 if acc < 0 else 0.0)
    return np.array([pos, vel, acc, vel_abs, sign_vel, sign_acc], dtype=np.float64)


def _fit_model(X: np.ndarray, y: np.ndarray):
    """Fit HuberRegressor if available; else ridge via normal equations. Returns (predict_fn, preds_array)."""
    try:
        from sklearn.linear_model import HuberRegressor
        reg = HuberRegressor(max_iter=200)
        reg.fit(X, y)
        preds = reg.predict(X)
        def predict(Xnew):
            return reg.predict(Xnew)
        return predict, preds
    except ImportError:
        n, d = X.shape
        lam = 0.01
        Xb = np.hstack([np.ones((n, 1)), X])
        w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(d + 1), Xb.T @ y)
        preds = Xb @ w

        def predict(Xnew):
            Xb_new = np.hstack([np.ones((Xnew.shape[0], 1)), Xnew])
            return Xb_new @ w

        return predict, preds


def sister_train_predict_fn(analogue_episodes: list[dict]):
    """Train on pool, return (predict_delta_fn, residual_iqr). predict_delta_fn(pos,vel,acc)->float."""
    if len(analogue_episodes) < MIN_TRAIN:
        return lambda p, v, a: 5.0, 5.0
    X = np.array([_feature_vec(e) for e in analogue_episodes])
    y = np.array([e.get("event_year", 0) - e.get("saddle_year", 0) for e in analogue_episodes], dtype=float)
    predict_fn, preds = _fit_model(X, y)
    residuals = np.abs(y - preds)
    iqr = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))

    def predict_delta(pos: float, vel: float, acc: float) -> float:
        x = _feature_vec({"position": pos, "velocity": vel, "acceleration": acc}).reshape(1, -1)
        return float(predict_fn(x)[0])

    return predict_delta, iqr


def sister_predict(
    now_year: int,
    position: float,
    velocity: float,
    acceleration: float,
    analogue_episodes: list[dict],
) -> dict:
    """
    Sister peak using Huber regression on Δt vs state.
    Returns peak_year, window_start, window_end, confidence_pct, method, n_train, residual_iqr.
    """
    if len(analogue_episodes) < MIN_TRAIN:
        return {
            "peak_year": now_year + 5,
            "window_start": now_year + 3,
            "window_end": now_year + 10,
            "confidence_pct": 50,
            "method": "sister_huber",
            "n_train": 0,
            "residual_iqr": 5.0,
        }

    X = np.array([_feature_vec(e) for e in analogue_episodes])
    y = np.array([e.get("event_year", 0) - e.get("saddle_year", 0) for e in analogue_episodes], dtype=float)

    predict_fn, preds = _fit_model(X, y)
    residuals = np.abs(y - preds)
    q10 = float(np.percentile(residuals, 10))
    q90 = float(np.percentile(residuals, 90))
    residual_iqr = float(np.percentile(residuals, 75) - np.percentile(residuals, 25))

    x_current = _feature_vec({"position": position, "velocity": velocity, "acceleration": acceleration})
    delta_pred = float(predict_fn(x_current.reshape(1, -1))[0])
    delta_pred = max(1, min(15, delta_pred))

    peak_year = now_year + int(round(delta_pred))
    window_start = now_year + int(round(max(1, delta_pred - (q90 - q10) / 2)))
    window_end = now_year + int(round(min(15, delta_pred + (q90 - q10) / 2)))
    if window_start >= window_end:
        window_start = peak_year - 2
        window_end = peak_year + 2

    conf = 70
    conf -= min(30, int(residual_iqr * 4))
    n_eff = len(analogue_episodes)
    if n_eff < 5:
        conf = int(conf * 0.6)
    conf = max(40, min(95, conf))

    return {
        "peak_year": peak_year,
        "window_start": window_start,
        "window_end": window_end,
        "confidence_pct": conf,
        "method": "sister_huber",
        "n_train": n_eff,
        "residual_iqr": round(residual_iqr, 2),
    }


def run_sister_engine() -> dict:
    """Run sister for latest state. Uses cerebro_peak_window for pool."""
    from cerebro_calibration import _load_episodes
    from cerebro_peak_window import compute_peak_window
    from cerebro_eval_utils import past_only_pool

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 2:
        return {"error": "Insufficient episodes", "peak_year": 0, "window_start": 0, "window_end": 0, "confidence_pct": 50}

    latest = max(episodes, key=lambda e: e.get("saddle_year", 0))
    pool = past_only_pool(episodes, latest["saddle_year"])
    if len(pool) < MIN_TRAIN:
        return {"error": "Insufficient past", "peak_year": 0, "window_start": 0, "window_end": 0, "confidence_pct": 50}

    pos = latest.get("position", 0)
    vel = latest.get("velocity", 0)
    acc = latest.get("acceleration", 0)
    now_year = latest["saddle_year"]

    out = sister_predict(now_year, pos, vel, acc, pool)
    return out


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_sister_engine()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Sister: peak={r.get('peak_year')}, conf={r.get('confidence_pct')}% → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
