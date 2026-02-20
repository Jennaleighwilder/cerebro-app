#!/usr/bin/env python3
"""
CEREBRO FIT DISTANCE WEIGHTS — Learn (vel_weight, acc_weight) from walk-forward MAE
==================================================================================
Grid search over past-only pools. Writes cerebro_data/distance_weights.json.
Does not modify core; only writes the weights file already supported by _get_distance_weights().
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "distance_weights.json"
SWEEP_PATH = SCRIPT_DIR / "cerebro_data" / "distance_weight_sweep.json"

VEL_GRID = [10, 25, 50, 75, 100, 150, 200, 300, 500]
ACC_GRID = [100, 250, 500, 1000, 2500, 5000, 10000]
MIN_TRAIN = 5


def _past_only_pool(episodes: list, t: int) -> list:
    """Episodes with saddle_year < t."""
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def _evaluate_pair(episodes: list, vw: float, aw: float) -> dict:
    """Walk-forward MAE, coverage_80, interval_width_mean for (vw, aw)."""
    from cerebro_core import compute_peak_window

    mae_list = []
    hits = 0
    widths = []
    n_used = 0

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    for ep in sorted_ep:
        Y = ep.get("saddle_year")
        if Y is None:
            continue
        pool = _past_only_pool(episodes, Y)
        if len(pool) < MIN_TRAIN:
            continue

        pred = compute_peak_window(
            Y,
            ep.get("position", 0),
            ep.get("velocity", 0),
            ep.get("acceleration", 0),
            ep.get("ring_B_score"),
            pool,
            interval_alpha=0.8,
            vel_weight=vw,
            acc_weight=aw,
        )
        event_yr = ep.get("event_year", Y + 5)
        mae_list.append(abs(pred["peak_year"] - event_yr))
        if pred["window_start"] <= event_yr <= pred["window_end"]:
            hits += 1
        widths.append(pred["window_end"] - pred["window_start"])
        n_used += 1

    if not mae_list:
        return {"mae": 999.0, "coverage_80": 0.0, "interval_width_mean": 999.0, "n_used": 0}

    mae = sum(mae_list) / len(mae_list)
    coverage = hits / len(mae_list)
    width_mean = sum(widths) / len(widths)
    return {"mae": mae, "coverage_80": coverage, "interval_width_mean": width_mean, "n_used": n_used}


def run_fit() -> dict:
    """Grid search: primary MAE, tie-break coverage_80, tie-break smaller interval width."""
    from cerebro_calibration import _load_episodes

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "version": 1}

    results = []
    for vw in VEL_GRID:
        for aw in ACC_GRID:
            r = _evaluate_pair(episodes, vw, aw)
            if r["n_used"] < MIN_TRAIN:
                continue
            results.append({
                "vel_weight": vw,
                "acc_weight": aw,
                "mae": round(r["mae"], 4),
                "coverage_80": round(r["coverage_80"], 4),
                "interval_width_mean": round(r["interval_width_mean"], 2),
                "n_used": r["n_used"],
            })

    if not results:
        return {"error": "No valid grid results", "version": 1}

    # Sort: primary MAE (lower better), tie-break coverage (higher better), tie-break width (smaller better)
    results.sort(key=lambda x: (x["mae"], -x["coverage_80"], x["interval_width_mean"]))
    best = results[0]

    out = {
        "vel_weight": best["vel_weight"],
        "acc_weight": best["acc_weight"],
        "fit": {
            "min_train": MIN_TRAIN,
            "n_used": best["n_used"],
            "mae": round(best["mae"], 2),
            "coverage_80": round(best["coverage_80"], 2),
            "interval_width_mean": round(best["interval_width_mean"], 2),
        },
        "grid_searched": {"vel": len(VEL_GRID), "acc": len(ACC_GRID)},
        "version": 1,
    }

    sweep_top20 = results[:20]
    SWEEP_PATH.parent.mkdir(exist_ok=True)
    with open(SWEEP_PATH, "w") as f:
        json.dump({"top_20": sweep_top20, "version": 1}, f, indent=2)

    return out


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_fit()
    if r.get("error"):
        print(f"Fit distance weights: {r['error']}")
        return 1
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Distance weights: vel={r['vel_weight']}, acc={r['acc_weight']}, mae={r['fit']['mae']} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
