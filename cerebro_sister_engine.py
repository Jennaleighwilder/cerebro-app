#!/usr/bin/env python3
"""
CEREBRO SISTER ENGINE — Parametric hazard model (non-analogue)
=============================================================
Learns P(event within H years) from state features via logistic regression.
Walk-forward: training at year Y uses only episodes with saddle_year < Y.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "hazard_curve_sister.json"
HORIZONS = [1, 3, 5, 10]
MIN_TRAIN = 5


def _load_episodes():
    """Load expanded candidate episodes (saddle_score>=2 etc)."""
    from cerebro_calibration import _load_episodes as _cal_load
    raw, _ = _cal_load(score_threshold=2.0)
    return raw


def _feature_vec(ep: dict) -> np.ndarray:
    """Features: [pos, vel, acc, ring_b]. ring_b in [-1,1] → use 0 if missing."""
    pos = float(ep.get("position", 0))
    vel = float(ep.get("velocity", 0))
    acc = float(ep.get("acceleration", 0))
    rb = ep.get("ring_B_score")
    rb = float(rb) if rb is not None else 0.0
    return np.array([pos, vel, acc, rb], dtype=np.float64)


def _fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> tuple:
    """Fit L2-regularized logistic regression. Returns (coef, intercept)."""
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=500, random_state=42)
        clf.fit(X, y)
        return clf.coef_.flatten(), float(clf.intercept_[0])
    except ImportError:
        return _logistic_nr(X, y, C)


def _logistic_nr(X: np.ndarray, y: np.ndarray, C: float, max_iter: int = 200) -> tuple:
    """Newton-Raphson logistic regression in pure numpy."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    lam = 1.0 / C
    for _ in range(max_iter):
        eta = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        grad = Xb.T @ (p - y) + lam * np.concatenate([[0], w[1:]])
        W = np.diag(p * (1 - p))
        H = Xb.T @ W @ Xb + lam * np.eye(d + 1)
        H[0, 0] -= lam
        try:
            dw = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            break
        w += dw
        if np.abs(dw).max() < 1e-6:
            break
    return w[1:], float(w[0])


def _predict_proba(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """P(y=1) for each row."""
    eta = X @ coef + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))


def _sister_hazard_at_episode(ep: dict, models: dict, horizons: list) -> dict:
    """Compute P_1yr, P_3yr, P_5yr, P_10yr for one episode using fitted models."""
    x = _feature_vec(ep).reshape(1, -1)
    out = {}
    for h in horizons:
        key = f"P_{h}yr"
        if key in models and models[key] is not None:
            coef, intercept = models[key]
            out[key] = round(float(_predict_proba(x, coef, intercept)[0]), 4)
        else:
            out[key] = 0.0
    # Enforce monotonicity
    for i in range(1, len(horizons)):
        out[f"P_{horizons[i]}yr"] = max(out.get(f"P_{horizons[i]}yr", 0), out.get(f"P_{horizons[i-1]}yr", 0))
    return out


def _sister_peak_window(saddle_year: int, probs: dict) -> tuple:
    """Find peak_year and window from P_by_T. T where P >= 0.5 = median time."""
    T_median = 10
    for h in [1, 3, 5, 10]:
        if probs.get(f"P_{h}yr", 0) >= 0.5:
            T_median = h
            break
    peak_year = saddle_year + T_median
    # Window: 10% and 90% probability years
    T_lo, T_hi = 1, 10
    for h in [1, 3, 5, 10]:
        if probs.get(f"P_{h}yr", 0) >= 0.1:
            T_lo = h
            break
    for h in [10, 5, 3, 1]:
        if probs.get(f"P_{h}yr", 1) <= 0.9:
            T_hi = h
            break
    window_start = saddle_year + T_lo
    window_end = saddle_year + T_hi
    return peak_year, window_start, window_end


def run_sister_engine() -> dict:
    """Walk-forward: for each episode, train on past only, predict hazard."""
    episodes = _load_episodes()
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "P_1yr": 0, "P_3yr": 0, "P_5yr": 0, "P_10yr": 0}

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    # Use latest episode for live hazard output
    latest = sorted_ep[-1]
    pool = [e for e in sorted_ep if e.get("saddle_year", 0) < latest.get("saddle_year", 0)]
    if len(pool) < MIN_TRAIN:
        return {"error": "Insufficient past episodes", "P_1yr": 0, "P_3yr": 0, "P_5yr": 0, "P_10yr": 0}

    models = {}
    for H in HORIZONS:
        X = np.array([_feature_vec(e) for e in pool])
        y = np.array([1.0 if (e.get("event_year", 0) - e.get("saddle_year", 0)) <= H else 0.0 for e in pool])
        if y.sum() < 2 or (1 - y).sum() < 2:
            models[f"P_{H}yr"] = None
            continue
        coef, intercept = _fit_logistic(X, y)
        models[f"P_{H}yr"] = (coef, intercept)

    probs = _sister_hazard_at_episode(latest, models, HORIZONS)
    peak_year, ws, we = _sister_peak_window(latest["saddle_year"], probs)

    out = {
        "P_1yr": probs.get("P_1yr", 0),
        "P_3yr": probs.get("P_3yr", 0),
        "P_5yr": probs.get("P_5yr", 0),
        "P_10yr": probs.get("P_10yr", 0),
        "peak_year": peak_year,
        "window_start": ws,
        "window_end": we,
        "now_year": latest["saddle_year"],
        "method": "sister_logistic",
        "n_train": len(pool),
    }
    return out


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_sister_engine()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Sister hazard: P_5yr={r.get('P_5yr')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
