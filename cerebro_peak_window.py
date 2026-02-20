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
    position_series: Optional[list[float]] = None,
    velocity_series: Optional[list[float]] = None,
    acceleration_series: Optional[list[float]] = None,
) -> dict:
    """Compute peak window. Optionally apply conformal calibration. Confidence is calibrated (raw stored in confidence_pct_raw)."""
    out = _compute_peak_window_core(
        now_year, position, velocity, acceleration,
        ring_b_score, analogue_episodes, interval_alpha,
        vel_weight, acc_weight,
    )
    try:
        from chimera import chimera_confidence_calibrator
        out = chimera_confidence_calibrator.calibrate_peak_window(
            out,
            position_series=position_series,
            velocity_series=velocity_series,
            acceleration_series=acceleration_series,
        )
    except Exception:
        out["confidence_pct_raw"] = out.get("confidence_pct", 50)
    # Contract windows (conformal v2): always attach status; widen if apply_conformal
    contract = None
    try:
        from cerebro_conformal_v2 import load_contract
        contract = load_contract()
        out["contract_status"] = (contract or {}).get("contract_status", "UNKNOWN")
        out["coverage_target"] = (contract or {}).get("coverage_target", 0.8)
        out["window_widen_factor"] = (contract or {}).get("window_widen_factor", 1.0)
    except Exception:
        out["contract_status"] = "UNKNOWN"
        out["coverage_target"] = 0.8
        out["window_widen_factor"] = 1.0

    if apply_conformal:
        try:
            from cerebro_conformal_v2 import apply_conformal_v2 as _apply_v2
            ws, we, s_hat, applied = _apply_v2(
                out["window_start"], out["window_end"],
                out.get("delta_p10", 0), out.get("delta_p90", 10),
                contract,
            )
            if applied:
                out["window_start"] = ws
                out["window_end"] = we
                out["window_label"] = "80% contract window" if (interval_alpha or 0.8) == 0.8 else out["window_label"]
                out["conformal_applied"] = True
                out["conformal_s_hat"] = round(s_hat, 4)
        except Exception:
            pass
    out.setdefault("conformal_applied", False)

    # Hallucination guard: forward-pass confidence discipline
    try:
        from chimera.chimera_hallucination_guard import apply_guard, load_diagnostics_bundle
        diag = load_diagnostics_bundle()
        diag["n_eff"] = diag.get("n_eff") or out.get("analogue_count")
        diag["interval_width"] = out.get("window_end", 0) - out.get("window_start", 0) if ("window_start" in out and "window_end" in out) else diag.get("interval_width")
        pred_for_guard = {
            "confidence_pct": out.get("confidence_pct", 50),
            "window_start": out.get("window_start"),
            "window_end": out.get("window_end"),
            "analogue_count": out.get("analogue_count"),
        }
        guarded = apply_guard(pred_for_guard, diagnostics=diag)
        if guarded["clamped"]:
            out["confidence_pct"] = int(round(guarded["confidence_after"]))
            if guarded.get("window_start") is not None and guarded.get("window_end") is not None:
                out["window_start"] = guarded["window_start"]
                out["window_end"] = guarded["window_end"]
        out["guard_state"] = guarded
    except Exception:
        out["guard_state"] = {"confidence_before": out.get("confidence_pct", 50), "confidence_after": out.get("confidence_pct", 50), "clamped": False, "reasons": []}

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
