#!/usr/bin/env python3
"""
CEREBRO REGIME PROBABILITY LAYER
Hidden Markov switching: Stable, Redistribution, Crackdown, Reform, Polarization.
Now it looks like intelligence, not narrative.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "regime_probabilities.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

REGIMES = ["Stable", "Redistribution", "Crackdown", "Reform", "Polarization"]


def _load_recent_state():
    import pandas as pd
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(20)
    if len(df) < 5:
        return None
    row = df.iloc[-1]
    pos = row.get("clock_position_10pt")
    vel = row.get("velocity")
    acc = row.get("acceleration")
    if pd.isna(pos) or pd.isna(vel) or pd.isna(acc):
        return None
    return {"position": float(pos), "velocity": float(vel), "acceleration": float(acc)}


def compute_regime_probabilities(state: dict | None = None) -> dict:
    """
    Heuristic regime mapping from (position, velocity, acceleration).
    Not full HMM — that would require transition matrix estimation.
    Uses state-space geometry as proxy for regime likelihood.
    """
    if state is None:
        state = _load_recent_state()
    if state is None:
        return {r: 0.2 for r in REGIMES}

    pos, vel, acc = state["position"], state["velocity"], state["acceleration"]

    # Heuristic scores (0–1) per regime
    # Stable: low |vel|, low |acc|
    stable = 1.0 - min(1.0, abs(vel) * 3 + abs(acc) * 5)

    # Redistribution: negative position (leftward), negative velocity
    redist = 0.5 + 0.5 * max(0, -pos) * (1 if vel < 0 else 0.5)

    # Crackdown: high negative acceleration (deceleration into repression)
    crackdown = min(1.0, max(0, -acc) * 4) if acc < 0 else 0.2

    # Reform: positive acceleration (building momentum)
    reform = min(1.0, max(0, acc) * 4) if acc > 0 else 0.2

    # Polarization: extreme position
    polar = min(1.0, abs(pos) * 1.5)

    raw = [stable, redist, crackdown, reform, polar]
    total = sum(raw) + 1e-6
    probs = [r / total for r in raw]
    s = sum(probs)
    probs = [p / s for p in probs]
    rounded = [round(p, 2) for p in probs]
    diff = 1.0 - sum(rounded)
    # Absorb rounding error into largest value
    idx = max(range(5), key=lambda i: rounded[i])
    rounded[idx] = round(rounded[idx] + diff, 2)
    rounded[idx] = max(0, min(1, rounded[idx]))
    return {
        "Stable": rounded[0],
        "Redistribution": rounded[1],
        "Crackdown": rounded[2],
        "Reform": rounded[3],
        "Polarization": rounded[4],
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = compute_regime_probabilities()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Regime: {r}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
