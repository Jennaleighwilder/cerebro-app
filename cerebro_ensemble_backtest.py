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
BACKTEST_SIM_RUNS = 50


def _load_episodes():
    from cerebro_calibration import _load_episodes as _cal_load
    raw, _ = _cal_load(score_threshold=2.0)
    return raw


def _core_pred(ep: dict, pool: list) -> dict:
    from cerebro_peak_window import compute_peak_window
    pred = compute_peak_window(
        ep["saddle_year"], ep.get("position", 0), ep.get("velocity", 0),
        ep.get("acceleration", 0), ep.get("ring_B_score"), pool, interval_alpha=0.8,
    )
    return {"peak_year": pred["peak_year"], "window_start": pred["window_start"],
            "window_end": pred["window_end"], "confidence": pred.get("confidence_pct", 50) / 100.0}


def _sister_pred(ep: dict, pool: list) -> dict:
    from cerebro_sister_engine import sister_predict
    out = sister_predict(
        ep["saddle_year"],
        ep.get("position", 0),
        ep.get("velocity", 0),
        ep.get("acceleration", 0),
        pool,
    )
    return {"peak_year": out["peak_year"], "window_start": out["window_start"],
            "window_end": out["window_end"], "confidence": out.get("confidence_pct", 70) / 100.0}


def _honeycomb_pred(ep: dict, pool: list) -> dict:
    from cerebro_forward_simulation import run_forward_simulation
    from cerebro_honeycomb import compute_honeycomb_fusion

    initial = (ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0))
    sim = run_forward_simulation(
        initial_state=initial,
        pool=pool,
        now_year=ep["saddle_year"],
        n_runs=BACKTEST_SIM_RUNS,
    )
    honey = compute_honeycomb_fusion(
        ep["saddle_year"],
        ep.get("position", 0),
        ep.get("velocity", 0),
        ep.get("acceleration", 0),
        pool,
        ep.get("ring_B_score"),
        sim_summary=sim if not sim.get("error") else None,
        shift_dict={"confidence_modifier": 1.0},
    )
    return {"peak_year": honey["peak_year"], "window_start": honey["window_start"],
            "window_end": honey["window_end"], "confidence": honey.get("confidence_pct", 70) / 100.0}


def run_backtest() -> dict:
    from cerebro_eval_utils import past_only_pool
    episodes = _load_episodes()
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "evaluation_candidates": "expanded", "production_trigger": "core saddles only"}

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))

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
            h = _honeycomb_pred(ep, pool)
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
