#!/usr/bin/env python3
"""
CEREBRO HONEYCOMB — Local mixture-of-experts (Core + Sister)
==========================================================
Partition state-space into cells; learn blending weights per cell from walk-forward MAE.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "hazard_curve_honeycomb.json"
WEIGHTS_PATH = SCRIPT_DIR / "cerebro_data" / "honeycomb_weights.json"

# Bin edges
POS_BINS = [-10, -5, 0, 5, 10]
VEL_BINS = [-float("inf"), -0.15, -0.05, 0.05, 0.15, float("inf")]
ACC_BINS = [-float("inf"), -0.05, 0, 0.05, float("inf")]
TAU = 1.0
MIN_TRAIN = 5


def _get_cell(pos: float, vel: float, acc: float) -> tuple:
    """Return (pbin, vbin, abin) indices."""
    pbin = sum(1 for b in POS_BINS if pos >= b) - 1
    pbin = max(0, min(len(POS_BINS) - 2, pbin))
    vbin = sum(1 for b in VEL_BINS if vel >= b) - 1
    vbin = max(0, min(len(VEL_BINS) - 2, vbin))
    abin = sum(1 for b in ACC_BINS if acc >= b) - 1
    abin = max(0, min(len(ACC_BINS) - 2, abin))
    return (pbin, vbin, abin)


def _softmax_weights(mae_core: float, mae_sister: float, tau: float = 1.0) -> tuple:
    """w_core, w_sister from softmax(-mae/tau). Lower MAE = higher weight."""
    s0 = np.exp(-mae_core / tau)
    s1 = np.exp(-mae_sister / tau)
    z = s0 + s1
    if z < 1e-10:
        return 0.5, 0.5
    return s0 / z, s1 / z


def _load_episodes():
    from cerebro_calibration import _load_episodes as _cal_load
    raw, _ = _cal_load(score_threshold=2.0)
    return raw


def _core_prediction(ep: dict, pool: list) -> dict:
    """Core (analogue) peak window for episode."""
    from cerebro_core import compute_peak_window
    pred = compute_peak_window(
        ep["saddle_year"],
        ep.get("position", 0),
        ep.get("velocity", 0),
        ep.get("acceleration", 0),
        ep.get("ring_B_score"),
        pool,
        interval_alpha=0.8,
    )
    return {
        "peak_year": pred["peak_year"],
        "window_start": pred["window_start"],
        "window_end": pred["window_end"],
        "P_1yr": _core_p_by_horizon(ep["saddle_year"], pred, 1),
        "P_3yr": _core_p_by_horizon(ep["saddle_year"], pred, 3),
        "P_5yr": _core_p_by_horizon(ep["saddle_year"], pred, 5),
        "P_10yr": _core_p_by_horizon(ep["saddle_year"], pred, 10),
    }


def _core_p_by_horizon(now_year: int, pred: dict, H: int) -> float:
    """Approximate P(event within H yr) from core window (linear in window)."""
    ws, we = pred["window_start"], pred["window_end"]
    T = now_year + H
    if T >= we:
        return 1.0
    if T <= ws:
        return 0.0
    span = max(1, we - ws)
    return min(1.0, (T - ws) / span)


def _sister_prediction(ep: dict, pool: list) -> dict:
    """Sister hazard for episode (train on pool, predict for ep)."""
    from cerebro_sister_engine import _feature_vec, _fit_logistic, _predict_proba, _sister_peak_window
    HORIZONS = [1, 3, 5, 10]
    models = {}
    for H in HORIZONS:
        X = np.array([_feature_vec(e) for e in pool])
        y = np.array([1.0 if (e.get("event_year", 0) - e.get("saddle_year", 0)) <= H else 0.0 for e in pool])
        if y.sum() < 2 or (1 - y).sum() < 2:
            models[f"P_{H}yr"] = None
            continue
        coef, intercept = _fit_logistic(X, y)
        models[f"P_{H}yr"] = (coef, intercept)
    x = _feature_vec(ep).reshape(1, -1)
    probs = {}
    for h in HORIZONS:
        key = f"P_{h}yr"
        if models.get(key):
            coef, intercept = models[key]
            probs[key] = float(_predict_proba(x, coef, intercept)[0])
        else:
            probs[key] = 0.0
    for i in range(1, len(HORIZONS)):
        probs[f"P_{HORIZONS[i]}yr"] = max(probs.get(f"P_{HORIZONS[i]}yr", 0), probs.get(f"P_{HORIZONS[i-1]}yr", 0))
    peak_year, ws, we = _sister_peak_window(ep["saddle_year"], probs)
    return {
        "peak_year": peak_year,
        "window_start": ws,
        "window_end": we,
        "P_1yr": probs.get("P_1yr", 0),
        "P_3yr": probs.get("P_3yr", 0),
        "P_5yr": probs.get("P_5yr", 0),
        "P_10yr": probs.get("P_10yr", 0),
    }


def _compute_cell_weights(episodes: list) -> dict:
    """Walk-forward: for each cell, compute MAE of core vs sister, derive weights."""
    from cerebro_eval_utils import past_only_pool
    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    cell_mae_core = {}
    cell_mae_sister = {}
    cell_n = {}

    for ep in sorted_ep:
        t = ep.get("saddle_year")
        if t is None:
            continue
        pool = past_only_pool(episodes, t)
        if len(pool) < MIN_TRAIN:
            continue
        cell = _get_cell(ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0))
        event_yr = ep.get("event_year", t + 5)

        try:
            core_pred = _core_prediction(ep, pool)
            sister_pred = _sister_prediction(ep, pool)
        except Exception:
            continue

        mae_core = abs(core_pred["peak_year"] - event_yr)
        mae_sister = abs(sister_pred["peak_year"] - event_yr)

        if cell not in cell_mae_core:
            cell_mae_core[cell] = []
            cell_mae_sister[cell] = []
        cell_mae_core[cell].append(mae_core)
        cell_mae_sister[cell].append(mae_sister)
        cell_n[cell] = cell_n.get(cell, 0) + 1

    weights = {}
    for cell in set(cell_mae_core.keys()) | set(cell_mae_sister.keys()):
        mae_c = np.mean(cell_mae_core.get(cell, [5.0]))
        mae_s = np.mean(cell_mae_sister.get(cell, [5.0]))
        w_c, w_s = _softmax_weights(mae_c, mae_s, TAU)
        weights[str(cell)] = {
            "w_core": round(w_c, 4),
            "w_sister": round(w_s, 4),
            "mae_core": round(mae_c, 2),
            "mae_sister": round(mae_s, 2),
            "n": cell_n.get(cell, 0),
        }
    return weights


def run_honeycomb() -> dict:
    """Compute honeycomb hazard for latest episode."""
    episodes = _load_episodes()
    if len(episodes) < MIN_TRAIN + 3:
        return {"error": "Insufficient episodes", "P_5yr": 0}

    weights = _compute_cell_weights(episodes)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump({"cells": weights, "tau": TAU}, f, indent=2)

    latest = max(episodes, key=lambda e: e.get("saddle_year", 0))
    pool = [e for e in episodes if e.get("saddle_year", 0) < latest.get("saddle_year", 0)]
    if len(pool) < MIN_TRAIN:
        return {"error": "Insufficient past", "P_5yr": 0}

    cell = _get_cell(latest.get("position", 0), latest.get("velocity", 0), latest.get("acceleration", 0))
    w_c = weights.get(str(cell), {}).get("w_core", 0.5)
    w_s = weights.get(str(cell), {}).get("w_sister", 0.5)
    if abs(w_c + w_s - 1.0) > 0.01:
        w_c, w_s = 0.5, 0.5

    core_pred = _core_prediction(latest, pool)
    sister_pred = _sister_prediction(latest, pool)

    P1 = w_c * core_pred["P_1yr"] + w_s * sister_pred["P_1yr"]
    P3 = w_c * core_pred["P_3yr"] + w_s * sister_pred["P_3yr"]
    P5 = w_c * core_pred["P_5yr"] + w_s * sister_pred["P_5yr"]
    P10 = w_c * core_pred["P_10yr"] + w_s * sister_pred["P_10yr"]
    peak_year = int(round(w_c * core_pred["peak_year"] + w_s * sister_pred["peak_year"]))
    ws = int(round(w_c * core_pred["window_start"] + w_s * sister_pred["window_start"]))
    we = int(round(w_c * core_pred["window_end"] + w_s * sister_pred["window_end"]))

    out = {
        "P_1yr": round(P1, 4),
        "P_3yr": round(P3, 4),
        "P_5yr": round(P5, 4),
        "P_10yr": round(min(1.0, P10), 4),
        "peak_year": peak_year,
        "window_start": ws,
        "window_end": we,
        "now_year": latest["saddle_year"],
        "method": "honeycomb",
        "w_core": round(w_c, 4),
        "w_sister": round(w_s, 4),
        "cell": list(cell),
    }
    return out


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_honeycomb()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Honeycomb: P_5yr={r.get('P_5yr')}, w_core={r.get('w_core')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
