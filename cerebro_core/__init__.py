"""
CEREBRO CORE — FROZEN. DO NOT MODIFY.
======================================
Core math: saddle detection, state distance, weighted quantiles, peak window.
If core changes → tests fail.
"""

import json
from pathlib import Path
from typing import Optional

CORE_VERSION = "1.0.0"
CORE_LOCKED = True

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
V_THRESH = 0.15
DIST_VEL_WEIGHT = 100
DIST_ACC_WEIGHT = 2500


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
    v_abs = abs(velocity)
    opposes = (velocity > 0 and acceleration < 0) or (velocity < 0 and acceleration > 0)
    below_thresh = v_abs < V_THRESH
    is_saddle = below_thresh and opposes
    v_factor = 1.0 - min(1.0, v_abs / (V_THRESH + 0.01))
    a_factor = min(1.0, abs(acceleration) / 0.1) if acceleration != 0 else 0.5
    ring_factor = (ring_b_score + 1) / 2 if ring_b_score is not None else 0.5
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
    vel_weight: Optional[float] = None,
    acc_weight: Optional[float] = None,
) -> float:
    """Euclidean distance in normalized state space."""
    vw = vel_weight if vel_weight is not None else DIST_VEL_WEIGHT
    aw = acc_weight if acc_weight is not None else DIST_ACC_WEIGHT
    return (
        (pos1 - pos2) ** 2
        + vw * (vel1 - vel2) ** 2
        + aw * (acc1 - acc2) ** 2
    ) ** 0.5


def compute_peak_window(
    now_year: int,
    position: float,
    velocity: float,
    acceleration: float,
    ring_b_score: Optional[float] = None,
    analogue_episodes: Optional[list[dict]] = None,
    interval_alpha: Optional[float] = None,
    vel_weight: Optional[float] = None,
    acc_weight: Optional[float] = None,
) -> dict:
    """
    Compute peak window using analogue-based time-to-event.
    analogue_episodes: list of {saddle_year, event_year, position, velocity, acceleration}
    """
    import pandas as pd

    if analogue_episodes is None:
        analogue_episodes = _load_analogue_episodes()

    alpha = interval_alpha if interval_alpha is not None else 0.8
    if alpha == 0.8:
        q_lo, q_hi = 0.10, 0.90
        window_label = "80% window"
    else:
        q_lo, q_hi = 0.25, 0.75
        window_label = "50% window"

    if not analogue_episodes:
        delta_med = 5.0 if velocity < 0 else 8.0
        return {
            "peak_year": now_year + int(round(delta_med)),
            "window_start": now_year + 3,
            "window_end": now_year + 10,
            "confidence_pct": 50,
            "analogue_count": 0,
            "method": "heuristic_fallback",
            "interval_alpha": alpha,
            "window_label": window_label,
            "quantile_lo": q_lo,
            "quantile_hi": q_hi,
        }

    vw = vel_weight if vel_weight is not None else _get_distance_weights()[0]
    aw = acc_weight if acc_weight is not None else _get_distance_weights()[1]
    deltas = []
    weights = []
    for ep in analogue_episodes:
        dt = ep.get("event_year", 0) - ep.get("saddle_year", 0)
        dist = state_distance(
            position, velocity, acceleration,
            ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
            vel_weight=vw, acc_weight=aw,
        )
        w = 1.0 / (1.0 + dist)
        deltas.append(float(dt))
        weights.append(w)

    med = weighted_median(deltas, weights)
    p_lo = weighted_quantile(deltas, weights, q_lo)
    p_hi = weighted_quantile(deltas, weights, q_hi)

    disp = p_hi - p_lo if len(deltas) > 1 else 5.0
    conf = 95 - min(40, int(disp * 3))
    if ring_b_score is not None and abs(ring_b_score) > 0.3:
        conf = min(95, conf + 5)

    ws = now_year + int(round(p_lo))
    we = now_year + int(round(p_hi))

    return {
        "peak_year": now_year + int(round(med)),
        "window_start": ws,
        "window_end": we,
        "confidence_pct": max(50, conf),
        "analogue_count": len(analogue_episodes),
        "delta_median": round(med, 1),
        "delta_p25": round(weighted_quantile(deltas, weights, 0.25), 1),
        "delta_p75": round(weighted_quantile(deltas, weights, 0.75), 1),
        "delta_p10": round(weighted_quantile(deltas, weights, 0.10), 1),
        "delta_p90": round(weighted_quantile(deltas, weights, 0.90), 1),
        "method": "analogue_weighted",
        "interval_alpha": alpha,
        "window_label": window_label,
        "quantile_lo": q_lo,
        "quantile_hi": q_hi,
    }


def _get_distance_weights() -> tuple[float, float]:
    p = SCRIPT_DIR / "cerebro_data" / "distance_weights.json"
    if p.exists():
        try:
            with open(p) as f:
                d = json.load(f)
            return (float(d.get("vel_weight", DIST_VEL_WEIGHT)), float(d.get("acc_weight", DIST_ACC_WEIGHT)))
        except Exception:
            pass
    return (DIST_VEL_WEIGHT, DIST_ACC_WEIGHT)


def _load_analogue_episodes() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH, index_col=0)
        df = df[df["clock_position_10pt"].notna()].tail(80)
        if len(df) < 10:
            return []
        EVENT_YEARS = {
            1933: "New Deal", 1935: "Wagner Act", 1965: "Great Society",
            1981: "Reagan shift", 1994: "Crime Bill", 2008: "Financial crisis",
            2020: "COVID response",
        }
        episodes = []
        for yr, row in df.iterrows():
            v, a, pos = row.get("velocity"), row.get("acceleration"), row.get("clock_position_10pt")
            if pd.isna(v) or pd.isna(a) or pd.isna(pos):
                continue
            v, a, pos = float(v), float(a), float(pos)
            is_sad, _ = detect_saddle_canonical(pos, v, a, row.get("ring_B_score"))
            if not is_sad:
                continue
            event_yr = None
            for ey in sorted(EVENT_YEARS.keys()):
                if ey > yr and (event_yr is None or ey - yr < 15):
                    event_yr = ey
                    break
            if event_yr is None:
                event_yr = yr + 5
            episodes.append({
                "saddle_year": int(yr),
                "event_year": event_yr,
                "position": pos,
                "velocity": v,
                "acceleration": a,
            })
        return episodes[-30:]
    except Exception:
        return []
