#!/usr/bin/env python3
"""
CEREBRO PEAK WINDOW — Thin wrapper over cerebro_core.
Adds conformal calibration layer. Core is frozen in cerebro_core/.
"""

from cerebro_core import (
    CORE_VERSION,
    CORE_LOCKED,
    V_THRESH,
    DIST_VEL_WEIGHT,
    DIST_ACC_WEIGHT,
    detect_saddle_canonical,
    state_distance,
    weighted_median,
    weighted_quantile,
    compute_peak_window as _compute_peak_window_core,
)
from typing import Optional

INTERVAL_ALPHA = 0.8


def compute_peak_window(
    now_year: int,
    position: float,
    velocity: float,
    acceleration: float,
    ring_b_score: Optional[float] = None,
    analogue_episodes: Optional[list[dict]] = None,
    interval_alpha: Optional[float] = None,
    apply_conformal: bool = False,
    vel_weight: Optional[float] = None,
    acc_weight: Optional[float] = None,
) -> dict:
    """Compute peak window. Optionally apply conformal calibration."""
    out = _compute_peak_window_core(
        now_year, position, velocity, acceleration,
        ring_b_score, analogue_episodes, interval_alpha,
        vel_weight, acc_weight,
    )
    if apply_conformal:
        try:
            from cerebro_conformal import load_calibration, apply_conformal as _apply
            cal = load_calibration()
            ws, we, s_hat, applied = _apply(
                out["window_start"], out["window_end"],
                out.get("delta_p10", 0), out.get("delta_p90", 10),
                cal,
            )
            if applied:
                out["window_start"] = ws
                out["window_end"] = we
                out["window_label"] = "80% calibrated window" if (interval_alpha or 0.8) == 0.8 else out["window_label"]
                out["conformal_applied"] = True
                out["conformal_s_hat"] = round(s_hat, 4)
        except Exception:
            pass
    out.setdefault("conformal_applied", False)
    return out


def get_method_equations() -> dict:
    from cerebro_core import _get_distance_weights
    vw, aw = _get_distance_weights()
    alpha = INTERVAL_ALPHA
    q_lo, q_hi = (0.10, 0.90) if alpha == 0.8 else (0.25, 0.75)
    wl = "80% window" if alpha == 0.8 else "50% window"
    return {
        "saddle_rule": f"Saddle when: |v| < {V_THRESH} AND sign(a) opposes sign(v).",
        "peak_window_rule": f"Peak year = now_year + weighted_median(Δt_i). Window = [now_year + p{int(q_lo*100)}, now_year + p{int(q_hi*100)}] ({wl}).",
        "thresholds": {"v_thresh": V_THRESH, "saddle_sign_oppose": True},
        "provenance": {
            "v_thresh": V_THRESH,
            "distance_vel_weight": vw,
            "distance_acc_weight": aw,
            "interval_alpha": alpha,
            "quantile_lo": q_lo,
            "quantile_hi": q_hi,
            "window_label": wl,
            "core_version": CORE_VERSION,
        },
    }
