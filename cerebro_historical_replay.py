#!/usr/bin/env python3
"""
CEREBRO HISTORICAL REPLAY — Year-by-year reconstruction
========================================================
Replay the entire dataset year-by-year and record what each engine would have predicted.
Past-only training: no future leakage.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "historical_replay.json"
EVENT_TOLERANCE = 10
MIN_TRAIN = 5
REPLAY_SIM_RUNS = 100


def _load_candidate_years():
    """Load candidate years: saddle_score >= 2, exclude years >= max_event_year - EVENT_TOLERANCE."""
    from cerebro_calibration import _load_episodes
    raw, _ = _load_episodes(score_threshold=2.0)
    return raw


def _past_only_pool(episodes: list, t: int) -> list:
    """Episodes with saddle_year < t."""
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def _core_pred(ep: dict, pool: list) -> dict:
    from cerebro_peak_window import compute_peak_window
    pred = compute_peak_window(
        ep["saddle_year"],
        ep.get("position", 0),
        ep.get("velocity", 0),
        ep.get("acceleration", 0),
        ep.get("ring_B_score"),
        pool,
        interval_alpha=0.8,
    )
    return {"peak_year": pred["peak_year"], "confidence": pred.get("confidence_pct", 50) / 100.0}


def _sister_pred(ep: dict, pool: list) -> dict | None:
    try:
        from cerebro_sister_engine import sister_predict
        out = sister_predict(
            ep["saddle_year"],
            ep.get("position", 0),
            ep.get("velocity", 0),
            ep.get("acceleration", 0),
            pool,
        )
        return {"peak_year": out["peak_year"], "window_start": out["window_start"], "window_end": out["window_end"]}
    except Exception:
        return None


def _honeycomb_pred(ep: dict, pool: list) -> dict | None:
    """Honeycomb fusion with past-only sim. Runs lightweight forward sim per year."""
    try:
        from cerebro_forward_simulation import run_forward_simulation
        from cerebro_honeycomb import compute_honeycomb_fusion

        initial = (ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0))
        sim = run_forward_simulation(
            initial_state=initial,
            pool=pool,
            now_year=ep["saddle_year"],
            n_runs=REPLAY_SIM_RUNS,
        )
        if sim.get("error"):
            sim = None
        honey = compute_honeycomb_fusion(
            ep["saddle_year"],
            ep.get("position", 0),
            ep.get("velocity", 0),
            ep.get("acceleration", 0),
            pool,
            ep.get("ring_B_score"),
            sim_summary=sim,
            shift_dict={"confidence_modifier": 1.0},
        )
        return honey
    except Exception:
        return None


def run_historical_replay() -> dict:
    episodes = _load_candidate_years()
    if len(episodes) < MIN_TRAIN + 3:
        return {"error": "Insufficient episodes", "version": 1, "n_years": 0, "mean_errors": {}, "records": []}

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    records = []
    core_errors, sister_errors, honey_errors = [], [], []

    for ep in sorted_ep:
        Y = ep.get("saddle_year")
        if Y is None:
            continue
        pool = _past_only_pool(episodes, Y)
        if len(pool) < MIN_TRAIN:
            continue

        actual_next_event_year = ep.get("event_year", Y + 5)

        try:
            core = _core_pred(ep, pool)
        except Exception:
            continue

        sister = _sister_pred(ep, pool)
        honey = _honeycomb_pred(ep, pool)

        core_peak = core["peak_year"]
        sister_peak = sister["peak_year"] if sister else None
        honey_peak = honey["peak_year"] if honey else None
        honey_ws = honey.get("window_start") if honey else None
        honey_we = honey.get("window_end") if honey else None

        core_err = abs(core_peak - actual_next_event_year)
        sister_err = abs(sister_peak - actual_next_event_year) if sister_peak is not None else None
        honey_err = abs(honey_peak - actual_next_event_year) if honey_peak is not None else None

        core_errors.append(core_err)
        if sister_err is not None:
            sister_errors.append(sister_err)
        if honey_err is not None:
            honey_errors.append(honey_err)

        rec = {
            "year": Y,
            "core_peak_year": core_peak,
            "sister_peak_year": sister_peak,
            "honeycomb_peak_year": honey_peak,
            "honeycomb_window_start": honey_ws,
            "honeycomb_window_end": honey_we,
            "actual_next_event_year": actual_next_event_year,
            "core_error": round(core_err, 2),
            "sister_error": round(sister_err, 2) if sister_err is not None else None,
            "honeycomb_error": round(honey_err, 2) if honey_err is not None else None,
            "n_eff": len(pool),
            "confidence": round(core.get("confidence", 0.5), 4),
        }
        records.append(rec)

    def mean(x):
        return sum(x) / len(x) if x else 0

    return {
        "version": 1,
        "n_years": len(records),
        "mean_errors": {
            "core": round(mean(core_errors), 2),
            "sister": round(mean(sister_errors), 2) if sister_errors else None,
            "honeycomb": round(mean(honey_errors), 2) if honey_errors else None,
        },
        "records": records,
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_historical_replay()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Historical replay: n_years={r.get('n_years')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
