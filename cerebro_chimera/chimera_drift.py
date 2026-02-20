#!/usr/bin/env python3
"""
CHIMERA DRIFT — Detect when to freeze learning
==============================================
Rolling drift on velocity mean/var, acceleration var, analogue density, error inflation.
If drift triggers: freeze updates, widen conformal, set drift_mode=true.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
WINDOW = 20
DRIFT_THRESHOLD_SIGMA = 2.0


def _load_harm_clock() -> list[tuple[int, float, float, float]]:
    """(year, position, velocity, acceleration)."""
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


def detect_drift() -> dict:
    """
    Compute drift on: velocity mean/var, acceleration var, analogue density, error inflation.
    drift_mode=true if any exceeds threshold.
    """
    data = _load_harm_clock()
    if len(data) < WINDOW * 2:
        return {"drift_mode": False, "reason": "Insufficient data", "version": 1}

    vels = [v for _, _, v, _ in data]
    accs = [a for _, _, _, a in data]

    # Past 20 vs previous 20
    recent_vel = vels[-WINDOW:]
    prev_vel = vels[-2 * WINDOW : -WINDOW]
    recent_acc = accs[-WINDOW:]
    prev_acc = accs[-2 * WINDOW : -WINDOW]

    vel_mean_drift = abs(np.mean(recent_vel) - np.mean(prev_vel)) / (np.std(prev_vel) + 1e-6)
    vel_var_drift = abs(np.var(recent_vel) - np.var(prev_vel)) / (np.var(prev_vel) + 1e-6)
    acc_var_drift = abs(np.var(recent_acc) - np.var(prev_acc)) / (np.var(prev_acc) + 1e-6)

    # Analogue density: saddles per decade (from reconstruction if available)
    rec_path = DATA_DIR / "chimera_reconstruction.json"
    analogue_drift = 0.0
    if rec_path.exists():
        try:
            with open(rec_path) as f:
                rec = json.load(f)
            records = rec.get("records", [])
            if len(records) >= 20:
                n_effs = [r.get("n_eff", 5) for r in records]
                recent_n = np.mean(n_effs[-10:])
                prev_n = np.mean(n_effs[-20:-10])
                analogue_drift = abs(recent_n - prev_n) / (prev_n + 1e-6)
        except Exception:
            pass

    # Error inflation: rolling MAE from params
    params_path = DATA_DIR / "chimera_params.json"
    mae_now = None
    mae_prev = None
    if params_path.exists():
        try:
            with open(params_path) as f:
                p = json.load(f)
            mae_now = p.get("rolling_mae")
        except Exception:
            pass
    # Check history for previous MAE
    hist_dir = DATA_DIR / "chimera_params_history"
    if hist_dir.exists() and mae_now is not None:
        versions = sorted(hist_dir.glob("params_*.json"), key=lambda x: x.stat().st_mtime)
        if len(versions) >= 2:
            try:
                with open(versions[-2]) as f:
                    p = json.load(f)
                mae_prev = p.get("rolling_mae")
            except Exception:
                pass
    error_inflation = 0.0
    if mae_now is not None and mae_prev is not None and mae_prev > 0:
        error_inflation = (mae_now - mae_prev) / mae_prev

    drift_mode = (
        vel_mean_drift > DRIFT_THRESHOLD_SIGMA or
        vel_var_drift > DRIFT_THRESHOLD_SIGMA or
        acc_var_drift > DRIFT_THRESHOLD_SIGMA or
        analogue_drift > 0.5 or
        error_inflation > 0.3
    )

    return {
        "version": 1,
        "drift_mode": bool(drift_mode),
        "vel_mean_drift_sigma": round(float(vel_mean_drift), 2),
        "vel_var_drift": round(float(vel_var_drift), 2),
        "acc_var_drift": round(float(acc_var_drift), 2),
        "analogue_density_drift": round(float(analogue_drift), 4),
        "error_inflation": round(float(error_inflation), 4),
        "threshold_sigma": DRIFT_THRESHOLD_SIGMA,
    }
