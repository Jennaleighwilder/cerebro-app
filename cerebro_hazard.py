#!/usr/bin/env python3
"""
CEREBRO EVENT HAZARD FORECASTING — Time-to-event probability
============================================================
Event library schema: {event_type, event_year, country, notes}.
Fit discrete-time logistic hazard using features from latent state + coupling.
Output: P(event in 1y/3y/5y) + expected time-to-event distribution.

Provenance: deterministic seed, feature list, coefficients.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "hazard_model.json"
RANDOM_SEED = 42

# Event library schema (US 1900–present)
EVENT_LIBRARY = [
    {"event_type": "redistribution", "event_year": 1933, "country": "US", "notes": "New Deal"},
    {"event_type": "redistribution", "event_year": 1935, "country": "US", "notes": "Wagner Act"},
    {"event_type": "redistribution", "event_year": 1965, "country": "US", "notes": "Great Society"},
    {"event_type": "policy_shift", "event_year": 1981, "country": "US", "notes": "Reagan shift"},
    {"event_type": "policy_shift", "event_year": 1994, "country": "US", "notes": "Crime Bill"},
    {"event_type": "policy_shift", "event_year": 2008, "country": "US", "notes": "Financial crisis"},
    {"event_type": "policy_shift", "event_year": 2020, "country": "US", "notes": "COVID response"},
]


def load_episodes_with_state(max_year: Optional[int] = None) -> list[dict]:
    """Load saddle episodes with (position, velocity, acceleration) at saddle year."""
    import pandas as pd
    from cerebro_peak_window import detect_saddle_canonical

    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(100)
    if len(df) < 15:
        return []

    episodes = []
    for yr, row in df.iterrows():
        v = row.get("velocity")
        a = row.get("acceleration")
        pos = row.get("clock_position_10pt")
        rb = row.get("ring_B_score")
        ss = row.get("saddle_score", 0)
        if any(pd.isna(x) for x in [v, a, pos]):
            continue
        v, a, pos = float(v), float(a), float(pos)
        rb = float(rb) if not pd.isna(rb) else None
        is_sad = (ss >= 2) if not pd.isna(ss) else detect_saddle_canonical(pos, v, a, rb)[0]
        if not is_sad:
            continue
        best_event = None
        best_dist = 999
        for ev in EVENT_LIBRARY:
            ey = ev["event_year"]
            if ey > yr and ey - yr <= 10:
                d = ey - yr
                if d < best_dist:
                    best_dist = d
                    best_event = ey
        if best_event is None:
            best_event = yr + 5
        if max_year is not None and yr > max_year:
            continue
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


def _features(ep: dict, coupling: Optional[dict] = None) -> np.ndarray:
    """Feature vector: |v|, |a|, saddle_intensity, position (normalized)."""
    v, a, p = ep["velocity"], ep["acceleration"], ep["position"]
    rb = ep.get("ring_B_score") or 0.5
    intensity = 0.4 * (1 - min(1, abs(v) / 0.2)) + 0.35 * min(1, abs(a) / 0.1) + 0.25 * (rb + 1) / 2
    return np.array([
        abs(v),
        abs(a),
        intensity,
        (p + 10) / 20,  # normalize position to [0,1]
    ], dtype=float)


def fit_logistic_hazard(
    episodes: list[dict],
    horizon_years: list[int] = [1, 3, 5],
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Fit discrete-time logistic hazard: P(event in next h years | state).
    Features: |v|, |a|, saddle_intensity, position.
    """
    np.random.seed(seed)
    if len(episodes) < 8:
        return {"error": "Insufficient episodes", "n": len(episodes)}

    X = np.vstack([_features(ep) for ep in episodes])
    # Target: event within horizon (binary per horizon)
    n = len(episodes)
    betas = []
    for h in horizon_years:
        y = np.array([1 if ep["dt"] <= h else 0 for ep in episodes], dtype=float)
        if y.sum() < 2 or (n - y.sum()) < 2:
            betas.append([0.0] * (X.shape[1] + 1))
            continue
        # Logistic regression via Newton (simple)
        Xb = np.column_stack([np.ones(n), X])
        beta = np.zeros(Xb.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-Xb @ beta))
            p = np.clip(p, 1e-8, 1 - 1e-8)
            W = np.diag(p * (1 - p))
            grad = Xb.T @ (y - p)
            Hess = Xb.T @ W @ Xb + 0.01 * np.eye(Xb.shape[1])
            delta = np.linalg.solve(Hess, grad)
            beta += delta
            if np.abs(delta).max() < 1e-6:
                break
        betas.append(beta.tolist())

    return {
        "horizon_years": horizon_years,
        "coefficients": betas,
        "feature_names": ["intercept", "abs_velocity", "abs_acceleration", "saddle_intensity", "position_norm"],
        "n_episodes": n,
        "provenance": {
            "model": "discrete_time_logistic",
            "random_seed": seed,
            "timestamp": int(time.time()),
            "version": 1,
        },
    }


def predict_hazard(
    position: float,
    velocity: float,
    acceleration: float,
    ring_b_score: Optional[float] = None,
    model: Optional[dict] = None,
) -> dict:
    """
    Predict P(event in 1y/3y/5y) and expected time-to-event.
    """
    if model is None:
        model_path = SCRIPT_DIR / "cerebro_data" / "hazard_model.json"
        if not model_path.exists():
            return {"error": "Model not fitted", "prob_1y": None, "prob_3y": None, "prob_5y": None}
        with open(model_path) as f:
            model = json.load(f)
    if "error" in model:
        return {"error": model["error"], "prob_1y": None, "prob_3y": None, "prob_5y": None}

    ep = {"position": position, "velocity": velocity, "acceleration": acceleration, "ring_B_score": ring_b_score}
    x = _features(ep)
    xb = np.concatenate([[1], x])

    probs = {}
    for i, h in enumerate(model["horizon_years"]):
        beta = np.array(model["coefficients"][i])
        if len(beta) != len(xb):
            beta = np.zeros(len(xb))
        logit = float(np.dot(beta, xb))
        p = 1 / (1 + np.exp(-logit))
        probs[f"prob_{h}y"] = round(min(0.99, max(0.01, p)), 4)

    # Expected time-to-event: weighted average of horizon midpoints
    e_tte = 0.0
    total_p = 0.0
    prev_p = 0.0
    for h in model["horizon_years"]:
        p = probs[f"prob_{h}y"] - prev_p
        prev_p = probs[f"prob_{h}y"]
        e_tte += (h - 0.5) * max(0, p)
        total_p += max(0, p)
    if total_p > 0:
        e_tte = e_tte / total_p
    else:
        e_tte = 5.0

    return {
        **probs,
        "expected_time_to_event_years": round(e_tte, 2),
        "provenance": {"model_version": model.get("provenance", {}).get("version", 1)},
    }


def run_fit() -> dict:
    """Fit hazard model, save to cerebro_data/hazard_model.json."""
    episodes = load_episodes_with_state()
    result = fit_logistic_hazard(episodes)
    if "error" not in result:
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    r = run_fit()
    if "error" in r:
        print(f"Error: {r['error']}")
    else:
        print(f"Fitted on {r['n_episodes']} episodes → {OUTPUT_PATH}")
        # Sample prediction
        pred = predict_hazard(-1.0, -0.1, 0.05, 0.5)
        print(f"Sample: P(1y)={pred.get('prob_1y')}, P(3y)={pred.get('prob_3y')}, P(5y)={pred.get('prob_5y')}")
