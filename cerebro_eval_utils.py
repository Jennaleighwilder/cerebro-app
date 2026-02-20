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
    use_honeycomb: bool = False,
    component_weights: Optional[dict] = None,
    use_coupling: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each episode, predict using only past-only pool. Returns list of
    {saddle_year, event_year, position, velocity, acceleration, ring_B_score,
     pred_peak_year, pred_window_start, pred_window_end, confidence, hit}.
    Skips episodes with insufficient past analogues.
    use_honeycomb: if True, use honeycomb fusion (core+sister+sim) instead of core only.
    component_weights: optional {core, sister, simulation} for inverse-MAE weighting.
    """
    from cerebro_peak_window import compute_peak_window

    if use_honeycomb:
        from cerebro_forward_simulation import run_forward_simulation
        from cerebro_honeycomb import compute_honeycomb_fusion
        sim_runs = 50

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    results = []
    for ep in sorted_ep:
        t = ep.get("saddle_year")
        if t is None:
            continue
        pool = past_only_pool(episodes, t)
        if len(pool) < min_train:
            continue

        if use_honeycomb:
            try:
                initial = (ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0))
                sim = run_forward_simulation(
                    initial_state=initial, pool=pool, now_year=t, n_runs=sim_runs,
                )
                raw_window_width = ep.get("window_end", 0) - ep.get("window_start", 0)
                if ep.get("window_start") is None or ep.get("window_end") is None:
                    core_pred = compute_peak_window(
                        t, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                        ep.get("ring_B_score"), pool, interval_alpha=interval_alpha,
                    )
                    ws, we = core_pred.get("window_start"), core_pred.get("window_end")
                    raw_window_width = (we - ws) if (ws is not None and we is not None) else 0
                pred = compute_honeycomb_fusion(
                    t, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                    pool, ep.get("ring_B_score"),
                    sim_summary=sim if not sim.get("error") else None,
                    shift_dict={"confidence_modifier": 1.0},
                    apply_conformal=False,
                    component_weights=component_weights,
                    use_coupling=use_coupling,
                    raw_window_width=raw_window_width if raw_window_width and raw_window_width > 0 else None,
                )
            except Exception:
                pred = compute_peak_window(
                    t, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                    ep.get("ring_B_score"), pool, interval_alpha=interval_alpha,
                )
        else:
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
        ws, we = pred["window_start"], pred["window_end"]
        out_ep = {
            **ep,
            "pred_peak_year": pred["peak_year"],
            "pred_window_start": ws,
            "pred_window_end": we,
            "interval_width": we - ws if (ws is not None and we is not None) else None,
            "confidence": pred["confidence_pct"] / 100.0,
            "hit": hit,
        }
        if use_honeycomb and "sister_trust_score" in pred:
            out_ep["sister_trust_score"] = pred["sister_trust_score"]
            out_ep["effective_sister_weight"] = pred["effective_sister_weight"]
            out_ep["fusion_mode"] = pred["fusion_mode"]
        results.append(out_ep)
    return results
