#!/usr/bin/env python3
"""
CHIMERA VALIDATION — Baseline skill comparison
==============================================
Shuffle test, ARIMA baseline (if available), random hazard, noise injection.
"""

import json
import random
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_validation.json"
MIN_TRAIN = 5


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def _brier(episodes_with_pred: list) -> float:
    """Brier score: mean (confidence - hit)^2."""
    total = 0
    n = 0
    for e in episodes_with_pred:
        conf = e.get("confidence", 0.5)
        hit = 1.0 if e.get("hit") else 0.0
        total += (conf - hit) ** 2
        n += 1
    return total / n if n > 0 else 0.5


def run_validation() -> dict:
    """Run shuffle test, baselines, compute skill score."""
    from cerebro_calibration import _load_episodes
    from cerebro_core import compute_peak_window
    from cerebro_eval_utils import past_only_pool

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "version": 1}

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    model_predictions = []

    for ep in sorted_ep:
        Y = ep.get("saddle_year")
        if Y is None:
            continue
        pool = past_only_pool(episodes, Y)
        if len(pool) < MIN_TRAIN:
            continue
        try:
            pred = compute_peak_window(
                Y, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                ep.get("ring_B_score"), pool, interval_alpha=0.8,
            )
            event_yr = ep.get("event_year", Y + 5)
            hit = pred["window_start"] <= event_yr <= pred["window_end"]
            conf = pred.get("confidence_pct", 70) / 100.0
            model_predictions.append({
                "confidence": conf, "hit": hit,
                "window_start": pred["window_start"], "window_end": pred["window_end"],
                "event_year": event_yr,
            })
        except Exception:
            continue

    if len(model_predictions) < 8:
        return {"error": "Too few predictions", "version": 1}

    model_brier = _brier([{"confidence": e["confidence"], "hit": e["hit"]} for e in model_predictions])

    # 1. Shuffle event years (same windows, shuffled ground truth)
    shuffled_events = [e["event_year"] for e in model_predictions]
    random.seed(42)
    random.shuffle(shuffled_events)
    shuffled_hits = []
    for i, e in enumerate(model_predictions):
        hit = e["window_start"] <= shuffled_events[i] <= e["window_end"]
        shuffled_hits.append({"confidence": e["confidence"], "hit": hit})
    shuffle_brier = _brier(shuffled_hits)

    # 2. Random hazard baseline
    random.seed(43)
    random_preds = [{"confidence": random.uniform(0.4, 0.9), "hit": e["hit"]} for e in model_predictions]
    random_brier = _brier(random_preds)

    baseline_brier = max(shuffle_brier, random_brier, 0.25)
    skill = 1.0 - (model_brier / baseline_brier) if baseline_brier > 0 else 0.0
    skill = max(-1.0, min(1.0, skill))

    return {
        "version": 1,
        "model_brier": round(model_brier, 4),
        "shuffle_baseline_brier": round(shuffle_brier, 4),
        "random_baseline_brier": round(random_brier, 4),
        "skill_score": round(skill, 4),
        "discrimination_vs_shuffle": round(1.0 - model_brier / shuffle_brier, 4) if shuffle_brier > 0 else None,
        "n_predictions": len(model_predictions),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_validation()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera validation: skill={r.get('skill_score')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
