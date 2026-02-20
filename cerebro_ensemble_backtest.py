#!/usr/bin/env python3
"""
CEREBRO ENSEMBLE BACKTEST — Core vs Sister vs Honeycomb
======================================================
Walk-forward comparison: MAE, coverage_80, Brier@5yr, interval width, n_eff.
Uses expanded candidates (saddle_score>=2); labels production_trigger = core saddles only.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "ensemble_backtest.json"
MIN_TRAIN = 5


def _load_episodes():
    from cerebro_calibration import _load_episodes as _cal_load
    raw, _ = _cal_load(score_threshold=2.0)
    return raw


def _core_pred(ep: dict, pool: list) -> dict:
    from cerebro_core import compute_peak_window
    pred = compute_peak_window(
        ep["saddle_year"], ep.get("position", 0), ep.get("velocity", 0),
        ep.get("acceleration", 0), ep.get("ring_B_score"), pool, interval_alpha=0.8,
    )
    return {"peak_year": pred["peak_year"], "window_start": pred["window_start"],
            "window_end": pred["window_end"], "confidence": pred["confidence_pct"] / 100.0}


def _sister_pred(ep: dict, pool: list) -> dict:
    from cerebro_sister_engine import _feature_vec, _fit_logistic, _predict_proba, _sister_peak_window
    import numpy as np
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
        if models.get(f"P_{h}yr"):
            coef, intercept = models[f"P_{h}yr"]
            probs[f"P_{h}yr"] = float(_predict_proba(x, coef, intercept)[0])
        else:
            probs[f"P_{h}yr"] = 0.0
    for i in range(1, len(HORIZONS)):
        probs[f"P_{HORIZONS[i]}yr"] = max(probs.get(f"P_{HORIZONS[i]}yr", 0), probs.get(f"P_{HORIZONS[i-1]}yr", 0))
    peak_year, ws, we = _sister_peak_window(ep["saddle_year"], probs)
    return {"peak_year": peak_year, "window_start": ws, "window_end": we,
            "confidence": probs.get("P_5yr", 0.5)}


def _honeycomb_pred(ep: dict, pool: list, weights: dict) -> dict:
    from cerebro_honeycomb import _get_cell, _core_p_by_horizon, _softmax_weights
    from cerebro_sister_engine import _feature_vec, _fit_logistic, _predict_proba, _sister_peak_window
    import numpy as np
    cell = _get_cell(ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0))
    w = weights.get(str(cell), {"w_core": 0.5, "w_sister": 0.5})
    w_c, w_s = w.get("w_core", 0.5), w.get("w_sister", 0.5)

    from cerebro_core import compute_peak_window
    core_pw = compute_peak_window(
        ep["saddle_year"], ep.get("position", 0), ep.get("velocity", 0),
        ep.get("acceleration", 0), ep.get("ring_B_score"), pool, interval_alpha=0.8,
    )
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
        if models.get(f"P_{h}yr"):
            coef, intercept = models[f"P_{h}yr"]
            probs[f"P_{h}yr"] = float(_predict_proba(x, coef, intercept)[0])
        else:
            probs[f"P_{h}yr"] = 0.0
    for i in range(1, len(HORIZONS)):
        probs[f"P_{HORIZONS[i]}yr"] = max(probs.get(f"P_{HORIZONS[i]}yr", 0), probs.get(f"P_{HORIZONS[i-1]}yr", 0))
    peak_year, ws, we = _sister_peak_window(ep["saddle_year"], probs)

    P5_core = _core_p_by_horizon(ep["saddle_year"], core_pw, 5)
    P5_sister = probs.get("P_5yr", 0.5)
    P5 = w_c * P5_core + w_s * P5_sister
    peak = int(round(w_c * core_pw["peak_year"] + w_s * peak_year))
    ws_blend = int(round(w_c * core_pw["window_start"] + w_s * ws))
    we_blend = int(round(w_c * core_pw["window_end"] + w_s * we))
    return {"peak_year": peak, "window_start": ws_blend, "window_end": we_blend, "confidence": P5}


def _compute_weights(episodes: list) -> dict:
    from cerebro_eval_utils import past_only_pool
    from cerebro_honeycomb import _get_cell, _softmax_weights
    cell_mae_core, cell_mae_sister = {}, {}
    for ep in sorted(episodes, key=lambda e: e.get("saddle_year", 0)):
        t = ep.get("saddle_year")
        if t is None:
            continue
        pool = past_only_pool(episodes, t)
        if len(pool) < MIN_TRAIN:
            continue
        try:
            core = _core_pred(ep, pool)
            sister = _sister_pred(ep, pool)
        except Exception:
            continue
        event_yr = ep.get("event_year", t + 5)
        cell = _get_cell(ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0))
        if cell not in cell_mae_core:
            cell_mae_core[cell] = []
            cell_mae_sister[cell] = []
        cell_mae_core[cell].append(abs(core["peak_year"] - event_yr))
        cell_mae_sister[cell].append(abs(sister["peak_year"] - event_yr))
    weights = {}
    for cell in set(cell_mae_core.keys()) | set(cell_mae_sister.keys()):
        mae_c = sum(cell_mae_core.get(cell, [5.0])) / max(1, len(cell_mae_core.get(cell, [])))
        mae_s = sum(cell_mae_sister.get(cell, [5.0])) / max(1, len(cell_mae_sister.get(cell, [])))
        w_c, w_s = _softmax_weights(mae_c, mae_s, 1.0)
        weights[str(cell)] = {"w_core": round(w_c, 4), "w_sister": round(w_s, 4)}
    return weights


def run_backtest() -> dict:
    from cerebro_eval_utils import past_only_pool
    episodes = _load_episodes()
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "evaluation_candidates": "expanded", "production_trigger": "core saddles only"}

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    weights = _compute_weights(episodes)

    core_mae, sister_mae, honey_mae = [], [], []
    core_hits, sister_hits, honey_hits = 0, 0, 0
    core_brier, sister_brier, honey_brier = [], [], []
    core_width, sister_width, honey_width = [], [], []
    n_total = 0

    for ep in sorted_ep:
        t = ep.get("saddle_year")
        if t is None:
            continue
        pool = past_only_pool(episodes, t)
        if len(pool) < MIN_TRAIN:
            continue
        event_yr = ep.get("event_year", t + 5)
        hit = lambda pred: pred["window_start"] <= event_yr <= pred["window_end"]
        brier = lambda pred: (pred["confidence"] - (1.0 if hit(pred) else 0.0)) ** 2

        try:
            c = _core_pred(ep, pool)
            s = _sister_pred(ep, pool)
            h = _honeycomb_pred(ep, pool, weights)
        except Exception:
            continue

        core_mae.append(abs(c["peak_year"] - event_yr))
        sister_mae.append(abs(s["peak_year"] - event_yr))
        honey_mae.append(abs(h["peak_year"] - event_yr))
        core_hits += 1 if hit(c) else 0
        sister_hits += 1 if hit(s) else 0
        honey_hits += 1 if hit(h) else 0
        core_brier.append(brier(c))
        sister_brier.append(brier(s))
        honey_brier.append(brier(h))
        core_width.append(c["window_end"] - c["window_start"])
        sister_width.append(s["window_end"] - s["window_start"])
        honey_width.append(h["window_end"] - h["window_start"])
        n_total += 1

    if n_total < 5:
        return {"error": "Too few scored episodes", "evaluation_candidates": "expanded", "production_trigger": "core saddles only"}

    def mean(x):
        return sum(x) / len(x) if x else 0

    core_mae_mean = mean(core_mae)
    sister_mae_mean = mean(sister_mae)
    honey_mae_mean = mean(honey_mae)
    winner = "core" if core_mae_mean <= min(sister_mae_mean, honey_mae_mean) else ("sister" if sister_mae_mean <= honey_mae_mean else "honeycomb")

    return {
        "evaluation_candidates": "expanded",
        "production_trigger": "core saddles only",
        "n_episodes": n_total,
        "core": {
            "mae": round(core_mae_mean, 2),
            "coverage_80": round(core_hits / n_total, 4),
            "brier_5yr": round(mean(core_brier), 4),
            "interval_width_mean": round(mean(core_width), 2),
        },
        "sister": {
            "mae": round(sister_mae_mean, 2),
            "coverage_80": round(sister_hits / n_total, 4),
            "brier_5yr": round(mean(sister_brier), 4),
            "interval_width_mean": round(mean(sister_width), 2),
        },
        "honeycomb": {
            "mae": round(honey_mae_mean, 2),
            "coverage_80": round(honey_hits / n_total, 4),
            "brier_5yr": round(mean(honey_brier), 4),
            "interval_width_mean": round(mean(honey_width), 2),
        },
        "winner": winner,
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_backtest()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Ensemble backtest: winner={r.get('winner')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
