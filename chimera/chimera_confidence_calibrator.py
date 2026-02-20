#!/usr/bin/env python3
"""
CHIMERA Confidence Calibrator — Wrapper-only. Core untouched.
Maps raw_confidence_pct → calibrated_confidence_pct via isotonic regression from calibration bins.
Applies n_eff, integrity, volatility caps. Prevents overconfidence in adversarial regimes.
"""

import json
from pathlib import Path

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

SCRIPT_DIR = _SCRIPT_DIR
DATA_DIR = SCRIPT_DIR / "cerebro_data"

CONF_LO, CONF_HI = 50, 95
N_EFF_THRESH = 5
N_EFF_MULT = 0.6
INTEGRITY_THRESH = 0.7
INTEGRITY_CAP = 70
VOL_CAP = 70
VOL_THRESH = 0.7
TIGHT_WIDTH = 3.0
TIGHT_VOL_CAP = 70  # interval_width <= 3 but vol_norm >= 0.7 → cap

WIDE_INTERVAL_THRESH = 5.0
WIDE_INTERVAL_CONF_CAP = 72  # pct; below this we consider underconfident
WIDE_INTERVAL_BOOST = 18  # pct points
WIDE_INTERVAL_MAX = 78  # pct cap


def _wide_interval_accuracy_correction(
    confidence_pct: float,
    interval_width: float | None,
    saddle_score: float | None,
) -> float:
    """
    Wide intervals get low confidence from the core formula.
    But empirically, wide-interval episodes hit at 84% not 55%.
    Correct upward when interval is wide but saddle signal is present.
    """
    if interval_width is None or interval_width < WIDE_INTERVAL_THRESH:
        return confidence_pct
    if confidence_pct >= WIDE_INTERVAL_CONF_CAP:
        return confidence_pct
    # Saddle score present and positive, or None (OECD)
    if saddle_score is not None and saddle_score < 0:
        return confidence_pct
    return min(WIDE_INTERVAL_MAX, confidence_pct + WIDE_INTERVAL_BOOST)


# Module-level isotonic model (lazy-fit from calibration_curve.json)
_isotonic_model = None
_isotonic_bins_hash = None


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _build_isotonic_model(bins: list[dict]) -> object | None:
    """
    Build isotonic regression model from bins.
    bins: [{conf_mid, empirical_hit_rate, n}, ...]. Uses conf_mid as X, empirical_hit_rate as y.
    Returns fitted model or None if insufficient data.
    """
    valid = [(b["conf_mid"], b["empirical_hit_rate"]) for b in bins if b.get("empirical_hit_rate") is not None and b.get("n", 0) > 0]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda t: t[0])
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    X = np.array([t[0] for t in valid]).reshape(-1, 1)
    y = np.array([t[1] for t in valid])
    model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    model.fit(X.ravel(), y)
    return model


def _isotonic_predict(x: float, bins: list[dict]) -> float | None:
    """
    Map raw confidence (0-1) to calibrated via isotonic regression.
    Returns calibrated value in [0,1] or None if insufficient bins.
    """
    global _isotonic_model, _isotonic_bins_hash
    bins_key = tuple((b.get("conf_mid"), b.get("empirical_hit_rate"), b.get("n")) for b in bins)
    if _isotonic_model is None or _isotonic_bins_hash != bins_key:
        _isotonic_model = _build_isotonic_model(bins)
        _isotonic_bins_hash = bins_key
    if _isotonic_model is None:
        return None
    import numpy as np
    out = _isotonic_model.predict(np.array([x]))
    return float(np.clip(out[0], 0, 1))


def calibrate(
    raw_confidence_pct: float,
    n_eff: int | None = None,
    average_integrity: float | None = None,
    vol_norm: float | None = None,
    interval_width: float | None = None,
    reason_out: list | None = None,
) -> tuple[int, str]:
    """
    Returns (calibrated_confidence_pct, reason).
    calibrated in [50, 95]. reason when capped.
    """
    reasons = []
    raw = max(0, min(100, float(raw_confidence_pct)))
    cal = raw / 100.0

    # 1. Calibration curve mapping (isotonic regression)
    cal_data = _load_json(DATA_DIR / "calibration_curve.json", {})
    bins = cal_data.get("bins", [])
    y = _isotonic_predict(raw / 100.0, bins)
    if y is not None:
        cal = y
        # Convert back to pct
        cal_pct = round(100 * cal)
    else:
        cal_pct = int(round(raw))
        reasons.append("insufficient_bins")

    # 2. n_eff cap
    if n_eff is not None and n_eff < N_EFF_THRESH:
        cal_pct = int(round(cal_pct * N_EFF_MULT))
        reasons.append("low_n_eff")

    # 3. Integrity cap
    if average_integrity is not None and average_integrity < INTEGRITY_THRESH:
        cal_pct = min(cal_pct, INTEGRITY_CAP)
        reasons.append("integrity_cap")

    # 4. Volatility cap (adversarial/noise)
    if vol_norm is not None and vol_norm >= VOL_THRESH:
        cal_pct = min(cal_pct, VOL_CAP)
        reasons.append("noise_cap")

    # 5. Tight-but-meaningless: interval_width <= 3 and vol_norm >= 0.7
    if (
        interval_width is not None
        and interval_width <= TIGHT_WIDTH
        and vol_norm is not None
        and vol_norm >= VOL_THRESH
    ):
        cal_pct = min(cal_pct, TIGHT_VOL_CAP)
        if "noise_cap" not in reasons:
            reasons.append("tight_noise_cap")

    cal_pct = max(CONF_LO, min(CONF_HI, cal_pct))
    reason = ";".join(reasons) if reasons else ""
    if reason_out is not None:
        reason_out[:] = reasons
    return (cal_pct, reason)


def calibrate_peak_window(
    pw_dict: dict,
    vol_norm: float | None = None,
    position_series: list | None = None,
    velocity_series: list | None = None,
    acceleration_series: list | None = None,
    k: int = 10,
) -> dict:
    """
    Add confidence_pct_raw, confidence_pct_calibrated to peak_window dict.
    If series provided, compute vol_norm from chimera_volatility.
    """
    out = dict(pw_dict)
    raw = out.get("confidence_pct", 50)
    out["confidence_pct_raw"] = raw

    n_eff = out.get("analogue_count", 0)
    interval_width = None
    if "window_start" in out and "window_end" in out:
        interval_width = out["window_end"] - out["window_start"]

    if position_series is not None and velocity_series is not None and acceleration_series is not None:
        try:
            from chimera import chimera_volatility
            vol_norm = chimera_volatility.compute_volatility_index(
                position_series, velocity_series, acceleration_series, k=k
            )
        except Exception:
            pass

    integrity = None
    try:
        integ = _load_json(DATA_DIR / "integrity_scores.json", {})
        integrity = integ.get("average_integrity")
    except Exception:
        pass

    cal_pct, reason = calibrate(
        raw, n_eff=n_eff, average_integrity=integrity,
        vol_norm=vol_norm, interval_width=interval_width,
    )
    # Wide-interval accuracy correction: boost underconfident wide-but-accurate episodes
    saddle_score = out.get("ring_B_score")
    cal_pct = _wide_interval_accuracy_correction(cal_pct, interval_width, saddle_score)
    out["confidence_pct"] = cal_pct
    out["confidence_pct_calibrated"] = cal_pct
    if reason:
        out["calibration_reason"] = reason
    return out
