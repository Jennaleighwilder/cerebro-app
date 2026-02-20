#!/usr/bin/env python3
"""
CEREBRO REGIME MARKOV — Sister #2: regime transition matrix
============================================================
Predicts regime trajectory probabilities forward k years using learned transition matrix.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "regime_markov.json"
REGIMES = ["Stable", "Redistribution", "Crackdown", "Reform", "Polarization"]


def _regime_from_state(pos: float, vel: float, acc: float) -> str:
    """Heuristic: assign max-prob regime from state geometry (same as cerebro_regime)."""
    stable = 1.0 - min(1.0, abs(vel) * 3 + abs(acc) * 5)
    redist = 0.5 + 0.5 * max(0, -pos) * (1 if vel < 0 else 0.5)
    crackdown = min(1.0, max(0, -acc) * 4) if acc < 0 else 0.2
    reform = min(1.0, max(0, acc) * 4) if acc > 0 else 0.2
    polar = min(1.0, abs(pos) * 1.5)
    scores = [stable, redist, crackdown, reform, polar]
    return REGIMES[np.argmax(scores)]


def _load_year_regimes() -> list[tuple[int, str]]:
    """Load (year, regime) from harm clock history."""
    import pandas as pd
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(80)
    if len(df) < 10:
        return []
    out = []
    for yr, row in df.iterrows():
        pos = row.get("clock_position_10pt")
        vel = row.get("velocity")
        acc = row.get("acceleration")
        if pd.isna(pos) or pd.isna(vel) or pd.isna(acc):
            continue
        reg = _regime_from_state(float(pos), float(vel), float(acc))
        out.append((int(yr), reg))
    return sorted(out, key=lambda x: x[0])


def _estimate_transition_matrix(year_regimes: list[tuple[int, str]]) -> np.ndarray:
    """T[i,j] = P(next regime j | current regime i). Laplace +1 smoothing."""
    n = len(REGIMES)
    idx = {r: i for i, r in enumerate(REGIMES)}
    counts = np.ones((n, n))
    for i in range(len(year_regimes) - 1):
        curr = year_regimes[i][1]
        nxt = year_regimes[i + 1][1]
        if curr in idx and nxt in idx:
            counts[idx[curr], idx[nxt]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    T = counts / np.maximum(row_sums, 1)
    return T


def run_regime_markov() -> dict:
    """Build T, compute p_k = p0 * T^k, output distributions + dominant path."""
    year_regimes = _load_year_regimes()
    if len(year_regimes) < 10:
        return {"error": "Insufficient year-regime history", "version": 1}

    T = _estimate_transition_matrix(year_regimes)

    # Current regime distribution from cerebro_regime
    try:
        from cerebro_regime import compute_regime_probabilities
        probs = compute_regime_probabilities()
        p0 = np.array([probs.get(r, 0.2) for r in REGIMES])
    except Exception:
        p0 = np.ones(len(REGIMES)) / len(REGIMES)

    p0 = p0 / p0.sum()

    # p_k = p0 @ T^k for k = 1, 3, 5, 10
    p_1yr = (p0 @ T).tolist()
    p_3yr = (p0 @ np.linalg.matrix_power(T, 3)).tolist()
    p_5yr = (p0 @ np.linalg.matrix_power(T, 5)).tolist()
    p_10yr = (p0 @ np.linalg.matrix_power(T, 10)).tolist()

    # Dominant path: argmax chain for 10 steps
    path = []
    p = p0.copy()
    for _ in range(10):
        i = int(np.argmax(p))
        path.append(REGIMES[i])
        p = (p @ T)

    return {
        "version": 1,
        "p_1yr": {r: round(p_1yr[i], 4) for i, r in enumerate(REGIMES)},
        "p_3yr": {r: round(p_3yr[i], 4) for i, r in enumerate(REGIMES)},
        "p_5yr": {r: round(p_5yr[i], 4) for i, r in enumerate(REGIMES)},
        "p_10yr": {r: round(p_10yr[i], 4) for i, r in enumerate(REGIMES)},
        "dominant_path_10yr": path,
        "n_years_used": len(year_regimes),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_regime_markov()
    if r.get("error"):
        print(f"Regime Markov: {r['error']}")
        return 1
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    top = max(r["p_5yr"].items(), key=lambda x: x[1])
    print(f"Regime Markov: top 5yr={top[0]} {top[1]} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
