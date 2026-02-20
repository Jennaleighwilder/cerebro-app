#!/usr/bin/env python3
"""
CEREBRO HAZARD INTERPOLATION — t_p10, t_p50, t_p90 from P(1), P(3), P(5), P(10)
==============================================================================
Piecewise linear interpolation. Deterministic and testable.
"""

# Anchor points: (t_years, P)
ANCHORS = [(1, "P_1yr"), (3, "P_3yr"), (5, "P_5yr"), (10, "P_10yr")]


def _p_at_t(probs: dict, t: float) -> float:
    """Piecewise linear P(t) from probs[P_1yr], P_3yr, P_5yr, P_10yr."""
    pts = [(1, probs.get("P_1yr", 0)), (3, probs.get("P_3yr", 0)),
           (5, probs.get("P_5yr", 0)), (10, probs.get("P_10yr", 0))]
    for i in range(len(pts) - 1):
        t_lo, p_lo = pts[i]
        t_hi, p_hi = pts[i + 1]
        if t_lo <= t <= t_hi:
            if p_hi == p_lo:
                return p_lo
            return p_lo + (p_hi - p_lo) * (t - t_lo) / (t_hi - t_lo)
    if t <= 1:
        return pts[0][1]
    return pts[-1][1]


def t_for_p(probs: dict, p_target: float) -> float:
    """
    Smallest t where P(t) >= p_target. Piecewise linear over (1,P1),(3,P3),(5,P5),(10,P10).
    If never reached by t=10, return 10.
    """
    pts = [(1, probs.get("P_1yr", 0)), (3, probs.get("P_3yr", 0)),
           (5, probs.get("P_5yr", 0)), (10, probs.get("P_10yr", 0))]
    for i in range(len(pts) - 1):
        t_lo, p_lo = pts[i]
        t_hi, p_hi = pts[i + 1]
        if p_target <= p_lo:
            return t_lo
        if p_lo <= p_target <= p_hi and p_hi != p_lo:
            t = t_lo + (p_target - p_lo) * (t_hi - t_lo) / (p_hi - p_lo)
            return round(t, 1)
        if p_target <= p_hi:
            return t_hi
    return 10.0


def hazard_to_window(probs: dict, now_year: int) -> dict:
    """
    Derive t_p10, t_p50, t_p90, peak_year, window_start, window_end from P(1), P(3), P(5), P(10).
    Enforces monotonicity on probs before interpolation.
    """
    p1 = probs.get("P_1yr", 0)
    p3 = max(probs.get("P_3yr", 0), p1)
    p5 = max(probs.get("P_5yr", 0), p3)
    p10 = max(probs.get("P_10yr", 0), p5)
    probs = {"P_1yr": p1, "P_3yr": p3, "P_5yr": p5, "P_10yr": p10}

    t_p10 = t_for_p(probs, 0.10)
    t_p50 = t_for_p(probs, 0.50)
    t_p90 = t_for_p(probs, 0.90)

    return {
        "t_p10": int(round(t_p10)),
        "t_p50": int(round(t_p50)),
        "t_p90": int(round(t_p90)),
        "peak_year": now_year + int(round(t_p50)),
        "window_start": now_year + int(round(t_p10)),
        "window_end": now_year + int(round(t_p90)),
    }
