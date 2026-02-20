#!/usr/bin/env python3
"""
CHIMERA REGIME HMM — HMM-lite transition matrix
================================================
Map years to regime labels (argmax of heuristic probs), update transition counts,
Dirichlet smoothing, export transition_matrix, entropy_per_row, most_likely_next_regime.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_PATH = DATA_DIR / "chimera_regime_hmm.json"
REGIMES = ["Stable", "Redistribution", "Crackdown", "Reform", "Polarization"]
DIRICHLET_ALPHA = 1.0


def _regime_label_for_year(year: int) -> str:
    """Get regime label from harm clock state at year. Uses heuristic from cerebro_regime."""
    import pandas as pd
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return "Stable"
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["clock_position_10pt"].notna()]
    idx = year if year in df.index else (str(year) if str(year) in df.index else None)
    if idx is None or len(df.loc[df.index == idx]) == 0:
        return "Stable"
    row = df.loc[idx]
    pos = float(row.get("clock_position_10pt", 0))
    vel = float(row.get("velocity", 0)) if not pd.isna(row.get("velocity")) else 0
    acc = float(row.get("acceleration", 0)) if not pd.isna(row.get("acceleration")) else 0

    stable = 1.0 - min(1.0, abs(vel) * 3 + abs(acc) * 5)
    redist = 0.5 + 0.5 * max(0, -pos) * (1 if vel < 0 else 0.5)
    crackdown = min(1.0, max(0, -acc) * 4) if acc < 0 else 0.2
    reform = min(1.0, max(0, acc) * 4) if acc > 0 else 0.2
    polar = min(1.0, abs(pos) * 1.5)
    probs = [stable, redist, crackdown, reform, polar]
    idx = int(np.argmax(probs))
    return REGIMES[idx]


def update_transition_matrix() -> dict:
    """Build transition counts from consecutive year labels, apply Dirichlet, export."""
    import pandas as pd
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return {"error": "No harm clock", "version": 1}

    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["clock_position_10pt"].notna()]
    years = sorted([int(y) for y in df.index if y == y])
    if len(years) < 10:
        return {"error": "Insufficient years", "version": 1}

    K = len(REGIMES)
    regime_to_idx = {r: i for i, r in enumerate(REGIMES)}
    counts = np.ones((K, K)) * DIRICHLET_ALPHA

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        r1 = _regime_label_for_year(y1)
        r2 = _regime_label_for_year(y2)
        if r1 in regime_to_idx and r2 in regime_to_idx:
            counts[regime_to_idx[r1], regime_to_idx[r2]] += 1

    # Normalize rows
    T = counts / counts.sum(axis=1, keepdims=True)
    transition_matrix = {REGIMES[i]: {REGIMES[j]: round(float(T[i, j]), 4) for j in range(K)} for i in range(K)}

    # Entropy per row
    entropy_per_row = {}
    for i, r in enumerate(REGIMES):
        row = T[i, :]
        row = row[row > 0]
        ent = -np.sum(row * np.log(row + 1e-10))
        entropy_per_row[r] = round(float(ent), 4)

    # Most likely next regime from current (use last year)
    last_label = _regime_label_for_year(years[-1])
    idx = regime_to_idx[last_label]
    next_probs = T[idx, :]
    next_idx = int(np.argmax(next_probs))
    most_likely_next_regime = REGIMES[next_idx]

    out = {
        "version": 1,
        "transition_matrix": transition_matrix,
        "entropy_per_row": entropy_per_row,
        "most_likely_next_regime": most_likely_next_regime,
        "n_years_used": len(years),
    }
    from cerebro_chimera import chimera_store
    chimera_store.atomic_write(OUTPUT_PATH, out)
    return out
