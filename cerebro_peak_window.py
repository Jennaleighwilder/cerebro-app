#!/usr/bin/env python3
"""
CEREBRO PEAK WINDOW RULE — Explicit, Deterministic
===================================================
Implements the canonical peak window computation for Oracle answers.
Used by cerebro_export_ui_data.py and cerebro_backtest.py.

SADDLE DETECTION:
  Saddle when: |v| < v_thresh AND sign(a) opposes sign(v)
  (deceleration toward turning point)
  Saddle intensity from normalized (|v|, |a|, distance-to-extreme) + Ring B

PEAK TIMING (analogue-based):
  - Build analogue library of historical saddle episodes
  - For each episode i: Δt_i = (event_year_i - saddle_year_i)
  - Similarity weight w_i from distance between current state and episode vector
  - Predicted peak year = now_year + weighted_median(Δt_i)
  - Window = [weighted_p25(Δt_i), weighted_p75(Δt_i)] + now_year
  - Confidence from: analogue count, dispersion, Ring B status, data completeness
"""

import json
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

# Thresholds (explicit, for /method and Show math)
V_THRESH = 0.15  # |v| below this = near turning point
SADDLE_SIGN_OPPOSE = True  # sign(a) opposes sign(v) required


def detect_saddle_canonical(
    position: float,
    velocity: float,
    acceleration: float,
    ring_b_score: Optional[float] = None,
) -> tuple[bool, float]:
    """
    Canonical saddle: |v| < v_thresh AND sign(a) opposes sign(v).
    Returns (is_saddle, intensity_0_to_1).
    """
    import math
    v_abs = abs(velocity)
    opposes = (velocity > 0 and acceleration < 0) or (velocity < 0 and acceleration > 0)
    below_thresh = v_abs < V_THRESH
    is_saddle = below_thresh and opposes

    # Intensity: combine |v| proximity to 0, |a| magnitude, Ring B
    v_factor = 1.0 - min(1.0, v_abs / (V_THRESH + 0.01))
    a_factor = min(1.0, abs(acceleration) / 0.1) if acceleration != 0 else 0.5
    ring_factor = (ring_b_score + 1) / 2 if ring_b_score is not None else 0.5  # [-1,1] -> [0,1]
    intensity = 0.4 * v_factor + 0.35 * a_factor + 0.25 * ring_factor
    intensity = min(1.0, max(0.0, intensity))
    return is_saddle, intensity


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Weighted median of values with weights."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    total = sum(weights)
    if total <= 0:
        return values[len(values) // 2] if values else 0.0
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    cum = 0.0
    for v, w in paired:
        cum += w
        if cum >= total / 2:
            return v
    return paired[-1][0] if paired else 0.0


def weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    """Weighted quantile (q in [0,1])."""
    if not values or not weights:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return values[0]
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    cum = 0.0
    for v, w in paired:
        cum += w
        if cum >= total * q:
            return v
    return paired[-1][0]


def state_distance(
    pos1: float, vel1: float, acc1: float,
    pos2: float, vel2: float, acc2: float,
) -> float:
    """Euclidean distance in normalized state space (pos, vel*10, acc*50)."""
    return (
        (pos1 - pos2) ** 2
        + 100 * (vel1 - vel2) ** 2
        + 2500 * (acc1 - acc2) ** 2
    ) ** 0.5


def compute_peak_window(
    now_year: int,
    position: float,
    velocity: float,
    acceleration: float,
    ring_b_score: Optional[float] = None,
    analogue_episodes: Optional[list[dict]] = None,
) -> dict:
    """
    Compute peak window using analogue-based time-to-event.
    analogue_episodes: list of {saddle_year, event_year, position, velocity, acceleration}
    """
    import pandas as pd

    if analogue_episodes is None:
        # Build from CSV if available
        analogue_episodes = _load_analogue_episodes()

    if not analogue_episodes:
        # Fallback: use fixed window from velocity/acceleration heuristic
        delta_med = 5.0 if velocity < 0 else 8.0
        return {
            "peak_year": now_year + int(round(delta_med)),
            "window_start": now_year + 3,
            "window_end": now_year + 10,
            "confidence_pct": 50,
            "analogue_count": 0,
            "method": "heuristic_fallback",
        }

    # Compute similarity weights
    deltas = []
    weights = []
    for ep in analogue_episodes:
        dt = ep.get("event_year", 0) - ep.get("saddle_year", 0)
        dist = state_distance(
            position, velocity, acceleration,
            ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
        )
        w = 1.0 / (1.0 + dist)  # closer = higher weight
        deltas.append(float(dt))
        weights.append(w)

    med = weighted_median(deltas, weights)
    p25 = weighted_quantile(deltas, weights, 0.25)
    p75 = weighted_quantile(deltas, weights, 0.75)

    # Dispersion affects confidence
    disp = p75 - p25 if len(deltas) > 1 else 5.0
    conf = 95 - min(40, int(disp * 3))
    if ring_b_score is not None and abs(ring_b_score) > 0.3:
        conf = min(95, conf + 5)

    return {
        "peak_year": now_year + int(round(med)),
        "window_start": now_year + int(round(p25)),
        "window_end": now_year + int(round(p75)),
        "confidence_pct": max(50, conf),
        "analogue_count": len(analogue_episodes),
        "delta_median": round(med, 1),
        "delta_p25": round(p25, 1),
        "delta_p75": round(p75, 1),
        "method": "analogue_weighted",
    }


def _load_analogue_episodes() -> list[dict]:
    """Load historical saddle→event episodes from CSV."""
    if not CSV_PATH.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH, index_col=0)
        df = df[df["clock_position_10pt"].notna()].tail(80)
        if len(df) < 10:
            return []

        # Labeled redistribution/policy events (US history)
        EVENT_YEARS = {
            1933: "New Deal", 1935: "Wagner Act", 1965: "Great Society",
            1981: "Reagan shift", 1994: "Crime Bill", 2008: "Financial crisis",
            2020: "COVID response",
        }
        episodes = []
        for i, (yr, row) in enumerate(df.iterrows()):
            v = row.get("velocity")
            a = row.get("acceleration")
            pos = row.get("clock_position_10pt")
            if pd.isna(v) or pd.isna(a) or pd.isna(pos):
                continue
            v, a, pos = float(v), float(a), float(pos)
            is_sad, _ = detect_saddle_canonical(pos, v, a, row.get("ring_B_score"))
            if not is_sad:
                continue
            # Find next event year
            event_yr = None
            for ey in sorted(EVENT_YEARS.keys()):
                if ey > yr and (event_yr is None or ey - yr < 15):
                    event_yr = ey
                    break
            if event_yr is None:
                event_yr = yr + 5  # default
            episodes.append({
                "saddle_year": int(yr),
                "event_year": event_yr,
                "position": pos,
                "velocity": v,
                "acceleration": a,
            })
        return episodes[-30:]  # last 30 episodes
    except Exception:
        return []


def get_method_equations() -> dict:
    """Return explicit equations for /method and Show math."""
    return {
        "saddle_rule": (
            "Saddle when: |v| < v_thresh AND sign(a) opposes sign(v). "
            f"v_thresh = {V_THRESH}. "
            "Saddle intensity = 0.4×v_factor + 0.35×a_factor + 0.25×ring_B."
        ),
        "peak_window_rule": (
            "Peak year = now_year + weighted_median(Δt_i). "
            "Window = [now_year + weighted_p25(Δt_i), now_year + weighted_p75(Δt_i)]. "
            "Δt_i = event_year_i - saddle_year_i per analogue episode i. "
            "Weight w_i = 1/(1 + distance(state_now, state_i))."
        ),
        "thresholds": {
            "v_thresh": V_THRESH,
            "saddle_sign_oppose": SADDLE_SIGN_OPPOSE,
        },
    }
