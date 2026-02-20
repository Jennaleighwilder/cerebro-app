#!/usr/bin/env python3
"""
CHIMERA EVOLUTION — Drift detector (deterministic, rule-based)
==============================================================
Detect drift in velocity magnitude, acceleration variance, analogue density, distance weights.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_evolution.json"
DRIFT_THRESHOLD = 2.0  # sigma


def _load_harm_clock() -> list[tuple[int, float, float, float]]:
    """Load (year, position, velocity, acceleration) from harm clock."""
    import pandas as pd
    p = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p, index_col=0)
    df = df[df["clock_position_10pt"].notna()]
    rows = []
    for yr, row in df.iterrows():
        v = row.get("velocity")
        a = row.get("acceleration")
        if pd.isna(v) or pd.isna(a):
            continue
        rows.append((int(yr), float(row.get("clock_position_10pt", 0)), float(v), float(a)))
    return rows


def run_evolution() -> dict:
    """Detect structural drift. model_structure_drift = True if any drift > threshold."""
    from cerebro_calibration import _load_episodes

    data = _load_harm_clock()
    if len(data) < 20:
        return {"error": "Insufficient harm clock data", "version": 1}

    years = [y for y, p, v, a in data]
    vels = [abs(v) for y, p, v, a in data]
    accs = [a for y, p, v, a in data]

    # Baseline: first half
    n = len(data)
    half = n // 2
    baseline_vel_mean = np.mean(vels[:half])
    baseline_vel_std = max(1e-6, np.std(vels[:half]))
    baseline_acc_var = np.var(accs[:half])
    baseline_acc_std = max(1e-6, np.std(accs[:half]))

    # Rolling 10-year window (recent)
    window = 10
    recent_vels = vels[-window:]
    recent_accs = accs[-window:]
    recent_vel_mean = np.mean(recent_vels)
    recent_vel_std = max(1e-6, np.std(recent_vels))
    recent_acc_var = np.var(recent_accs)

    vel_drift_sigma = abs(recent_vel_mean - baseline_vel_mean) / baseline_vel_std
    acc_drift_sigma = abs(np.sqrt(recent_acc_var) - np.sqrt(baseline_acc_var)) / baseline_acc_std

    # Analogue density drift: mean n_eff over time (from reconstruction if available)
    episodes, _ = _load_episodes(score_threshold=2.0)
    rec_path = SCRIPT_DIR / "cerebro_data" / "chimera_reconstruction.json"
    n_eff_drift = 0.0
    if rec_path.exists() and episodes:
        try:
            with open(rec_path) as f:
                rec = json.load(f)
            records = rec.get("records", [])
            if len(records) >= 10:
                n_effs = [r.get("n_eff", 5) for r in records]
                half_n = len(n_effs) // 2
                baseline_n = np.mean(n_effs[:half_n]) if half_n > 0 else 5
                recent_n = np.mean(n_effs[-5:]) if len(n_effs) >= 5 else 5
                n_eff_drift = abs(recent_n - baseline_n) / max(1, baseline_n)
        except Exception:
            pass

    # Distance weight optimal drift: compare current to historical (we don't have historical weights)
    # Skip for now; would need to store weight history
    weight_drift = 0.0

    model_structure_drift = bool(
        vel_drift_sigma > DRIFT_THRESHOLD or
        acc_drift_sigma > DRIFT_THRESHOLD or
        n_eff_drift > 0.5
    )

    drift_magnitude = min(1.0, (vel_drift_sigma + acc_drift_sigma) / 4.0 + n_eff_drift * 0.5)

    return {
        "version": 1,
        "model_structure_drift": model_structure_drift,
        "drift_magnitude_score": round(float(drift_magnitude), 4),
        "velocity_drift_sigma": round(float(vel_drift_sigma), 2),
        "acceleration_drift_sigma": round(float(acc_drift_sigma), 2),
        "analogue_density_drift": round(float(n_eff_drift), 4),
        "threshold_sigma": DRIFT_THRESHOLD,
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_evolution()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera evolution: drift={r.get('model_structure_drift')}, magnitude={r.get('drift_magnitude_score')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
