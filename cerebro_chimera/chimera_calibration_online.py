#!/usr/bin/env python3
"""
CHIMERA CALIBRATION ONLINE — Rolling conformal residuals
========================================================
Maintain rolling residuals, quantile at 80%, widen windows for honest coverage.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_PATH = DATA_DIR / "chimera_calibration_online.json"
RESIDUALS_PATH = DATA_DIR / "chimera_residuals.json"
MAX_RESIDUALS = 200
TARGET_COVERAGE = 0.8
MIN_TRAIN = 5


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def _load_residuals() -> list:
    if not RESIDUALS_PATH.exists():
        return []
    try:
        with open(RESIDUALS_PATH) as f:
            d = json.load(f)
        return d.get("residuals", [])[-MAX_RESIDUALS:]
    except Exception:
        return []


def _save_residuals(residuals: list) -> None:
    from cerebro_chimera import chimera_store
    chimera_store.atomic_write(RESIDUALS_PATH, {"residuals": residuals[-MAX_RESIDUALS:], "version": 1})


def update_conformal(episodes: list | None = None) -> dict:
    """
    Walk-forward: compute residuals, append, compute conformal_q80.
    residual = max(0, ws - event_year, event_year - we)
    """
    from cerebro_calibration import _load_episodes
    from cerebro_core import compute_peak_window
    from cerebro_chimera import chimera_store
    from cerebro_chimera.chimera_store import load_params

    if episodes is None:
        episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "updated": False}

    params = load_params()
    vw = params.get("vel_weight", 100)
    aw = params.get("acc_weight", 2500)

    residuals = _load_residuals()
    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))

    for ep in sorted_ep:
        Y = ep.get("saddle_year")
        if Y is None:
            continue
        pool = _past_only_pool(episodes, Y)
        if len(pool) < MIN_TRAIN:
            continue
        try:
            pred = compute_peak_window(
                Y, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                ep.get("ring_B_score"), pool, interval_alpha=0.8,
                vel_weight=vw, acc_weight=aw,
            )
            ws = pred.get("window_start")
            we = pred.get("window_end")
            event_yr = ep.get("event_year", Y + 5)
            if ws is not None and we is not None:
                r = max(0, ws - event_yr, event_yr - we)
                residuals.append(float(r))
        except Exception:
            continue

    residuals = residuals[-MAX_RESIDUALS:]
    _save_residuals(residuals)

    if len(residuals) < 5:
        conformal_q80 = 1.0
        coverage_50 = coverage_200 = None
    else:
        idx = int(np.ceil(TARGET_COVERAGE * len(residuals))) - 1
        idx = max(0, min(idx, len(residuals) - 1))
        conformal_q80 = float(np.sort(residuals)[idx])
        hits_50 = sum(1 for r in residuals[-50:] if r == 0)
        hits_200 = sum(1 for r in residuals if r == 0)
        coverage_50 = hits_50 / min(50, len(residuals[-50:])) if residuals[-50:] else None
        coverage_200 = hits_200 / len(residuals) if residuals else None

    out = {
        "version": 1,
        "conformal_q80": round(conformal_q80, 2),
        "residual_count": len(residuals),
        "coverage_last_50": round(coverage_50, 4) if coverage_50 is not None else None,
        "coverage_last_200": round(coverage_200, 4) if coverage_200 is not None else None,
        "target_coverage": TARGET_COVERAGE,
    }
    chimera_store.atomic_write(OUTPUT_PATH, out)
    return out
