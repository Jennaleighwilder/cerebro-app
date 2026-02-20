#!/usr/bin/env python3
"""
CEREBRO DISTRIBUTION SHIFT DETECTOR
==================================
Detect if current state lies outside historical memory cloud.
Mahalanobis distance + percentile rank.
"""

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "distribution_shift.json"


def _load_historical_states() -> list[tuple[float, float, float]]:
    """Load historical (position, velocity, acceleration) from harm clock."""
    import pandas as pd
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["clock_position_10pt"].notna()]
    if df.empty or len(df) < 5:
        return []
    states = []
    for _, row in df.iterrows():
        pos = row.get("clock_position_10pt")
        vel = row.get("velocity")
        acc = row.get("acceleration")
        if pd.isna(pos) or pd.isna(vel) or pd.isna(acc):
            continue
        states.append((float(pos), float(vel), float(acc)))
    return states


def _mahalanobis(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> float:
    """Mahalanobis distance: sqrt((x - mu)^T Sigma^-1 (x - mu))."""
    d = x - mu
    return float(np.sqrt(np.maximum(0, d.T @ sigma_inv @ d)))


def _outside_convex_hull(x: np.ndarray, points: np.ndarray) -> bool:
    """True if x is outside convex hull of points. Simplified: check if beyond max in any direction."""
    if len(points) < 4:
        return False
    for j in range(3):
        if x[j] > points[:, j].max() + 1e-6 or x[j] < points[:, j].min() - 1e-6:
            return True
    return False


def run_distribution_shift() -> dict:
    """Compute Mahalanobis distance of current state vs historical, percentile, confidence modifier."""
    states = _load_historical_states()
    if len(states) < 5:
        return {"error": "Insufficient historical states", "version": 1}

    X = np.array(states)
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    if np.linalg.cond(cov) > 1e10:
        cov += np.eye(3) * 1e-6
    try:
        sigma_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(cov)

    current = X[-1]
    d_current = _mahalanobis(current, mu, sigma_inv)

    distances = [_mahalanobis(X[i], mu, sigma_inv) for i in range(len(X))]
    percentile = sum(1 for d in distances if d <= d_current) / len(distances)
    outside_hull = _outside_convex_hull(current, X[:-1])

    # OOD gate: hard clamp production confidence
    if percentile > 0.97:
        ood_level = "SEVERE"
        confidence_modifier = 0.65
    elif percentile > 0.90:
        ood_level = "ELEVATED"
        confidence_modifier = 0.85
    else:
        ood_level = "NORMAL"
        confidence_modifier = 1.0

    return {
        "version": 1,
        "mahalanobis_distance": round(float(d_current), 2),
        "percentile": round(float(percentile), 4),
        "outside_historical_convex_hull": outside_hull,
        "ood_level": ood_level,
        "confidence_modifier": round(confidence_modifier, 2),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_distribution_shift()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Distribution shift: d={r.get('mahalanobis_distance')}, pct={r.get('percentile')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
