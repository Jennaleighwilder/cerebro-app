#!/usr/bin/env python3
"""
CEREBRO PLACEBO TEST — Permutation / placebo test (evaluation layer only)
========================================================================
Shuffle event_year labels across episodes N=1000; re-run walk-forward pipeline.
Output p-values: fraction of placebo runs that match or beat real metrics.
Proves calibration is not coincidence.
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLACEBO_PATH = SCRIPT_DIR / "cerebro_data" / "placebo_test.json"
CANDIDATE_SWEEP_PATH = SCRIPT_DIR / "cerebro_data" / "candidate_sweep.json"
N_PLACEBO = 1000
SWEEP_THRESHOLDS = [2.0, 2.5, 3.0]


def _run_placebo():
    from cerebro_calibration import _load_episodes
    from cerebro_eval_utils import walkforward_predictions

    raw, _ = _load_episodes(score_threshold=2.0)
    if len(raw) < 8:
        return None, None, None

    MIN_TRAIN = 3
    episodes = walkforward_predictions(raw, interval_alpha=0.8, min_train=MIN_TRAIN)
    if len(episodes) < 8:
        return None, None, None

    real_brier = sum((e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e in episodes) / len(episodes)
    real_coverage = sum(1 for e in episodes if e["hit"]) / len(episodes)

    event_years = [e["event_year"] for e in raw]
    placebo_briers = []
    placebo_coverages = []

    for _ in range(N_PLACEBO):
        shuffled = event_years.copy()
        random.shuffle(shuffled)
        placebo_raw = [
            {**ep, "event_year": shuffled[i]}
            for i, ep in enumerate(raw)
        ]
        pl_episodes = walkforward_predictions(placebo_raw, interval_alpha=0.8, min_train=MIN_TRAIN)
        if len(pl_episodes) < 8:
            continue
        pl_brier = sum((e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e in pl_episodes) / len(pl_episodes)
        pl_coverage = sum(1 for e in pl_episodes if e["hit"]) / len(pl_episodes)
        placebo_briers.append(pl_brier)
        placebo_coverages.append(pl_coverage)

    n_valid = len(placebo_briers)
    p_value_brier = sum(1 for b in placebo_briers if b <= real_brier) / n_valid if n_valid else None
    p_value_coverage = sum(1 for c in placebo_coverages if c >= real_coverage) / n_valid if n_valid else None

    return {
        "real_brier": round(real_brier, 4),
        "real_coverage_80": round(real_coverage, 4),
        "n_placebo": n_valid,
        "p_value_brier": round(p_value_brier, 4) if p_value_brier is not None else None,
        "p_value_coverage": round(p_value_coverage, 4) if p_value_coverage is not None else None,
        "placebo_brier_mean": round(sum(placebo_briers) / n_valid, 4) if n_valid else None,
        "placebo_coverage_mean": round(sum(placebo_coverages) / n_valid, 4) if n_valid else None,
    }, placebo_briers, placebo_coverages


def _run_candidate_sweep():
    from cerebro_calibration import run_calibration

    sweep = []
    for thresh in SWEEP_THRESHOLDS:
        r = run_calibration(score_threshold=thresh)
        if "error" in r:
            sweep.append({
                "threshold": thresh,
                "n_used": 0,
                "brier": None,
                "coverage_80": None,
                "mean_n_eff": None,
                "interval_width_mean": None,
            })
            continue
        sweep.append({
            "threshold": thresh,
            "n_used": r.get("n_used", 0),
            "brier": r.get("brier_score"),
            "coverage_80": r.get("coverage_80"),
            "mean_n_eff": r.get("mean_n_eff"),
            "interval_width_mean": r.get("interval_width_mean"),
        })
    return sweep


def main():
    PLACEBO_PATH.parent.mkdir(exist_ok=True)

    # Placebo test
    result, _, _ = _run_placebo()
    if result:
        with open(PLACEBO_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Placebo: p_brier={result['p_value_brier']}, p_coverage={result['p_value_coverage']} → {PLACEBO_PATH}")
    else:
        print("Placebo: insufficient episodes, skipped")

    # Candidate sweep
    sweep = _run_candidate_sweep()
    sweep_out = {"thresholds": SWEEP_THRESHOLDS, "sweep": sweep}
    with open(CANDIDATE_SWEEP_PATH, "w") as f:
        json.dump(sweep_out, f, indent=2)
    print(f"Candidate sweep: {CANDIDATE_SWEEP_PATH}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
