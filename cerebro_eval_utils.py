#!/usr/bin/env python3
"""
CEREBRO EVALUATION UTILITIES — Past-only analogue pools
=======================================================
No future leakage: for episode at year t, analogue pool = episodes with saddle_year < t.
"""

from typing import List, Dict, Any, Optional


def past_only_pool(episodes: List[Dict[str, Any]], t: int) -> List[Dict[str, Any]]:
    """
    Return analogue pool for prediction at year t: only episodes with saddle_year < t.
    For a 1990 prediction, you cannot use 2008 analogues.
    """
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def walkforward_predictions(
    episodes: List[Dict[str, Any]],
    interval_alpha: float = 0.8,
    min_train: int = 5,
) -> List[Dict[str, Any]]:
    """
    For each episode, predict using only past-only pool. Returns list of
    {saddle_year, event_year, position, velocity, acceleration, ring_B_score,
     pred_peak_year, pred_window_start, pred_window_end, confidence, hit}.
    Skips episodes with insufficient past analogues.
    """
    from cerebro_core import compute_peak_window

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    results = []
    for ep in sorted_ep:
        t = ep.get("saddle_year")
        if t is None:
            continue
        pool = past_only_pool(episodes, t)
        if len(pool) < min_train:
            continue
        pred = compute_peak_window(
            t,
            ep.get("position", 0),
            ep.get("velocity", 0),
            ep.get("acceleration", 0),
            ep.get("ring_B_score"),
            pool,
            interval_alpha=interval_alpha,
        )
        event_yr = ep.get("event_year", t + 5)
        hit = pred["window_start"] <= event_yr <= pred["window_end"]
        results.append({
            **ep,
            "pred_peak_year": pred["peak_year"],
            "pred_window_start": pred["window_start"],
            "pred_window_end": pred["window_end"],
            "confidence": pred["confidence_pct"] / 100.0,
            "hit": hit,
        })
    return results
