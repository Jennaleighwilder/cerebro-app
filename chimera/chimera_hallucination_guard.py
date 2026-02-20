#!/usr/bin/env python3
"""
CHIMERA Hallucination Guard — Forward-pass confidence discipline.
Confidence must be earned. Never increases confidence. Never shrinks window.
Only clamps when: noise overconfidence, low support, drift, wide-but-confident.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_diagnostics_bundle() -> dict:
    """Assemble diagnostics from contract_report, synthetic_worlds, infinity_score, calibration."""
    bundle = {}
    contract = _load_json(DATA_DIR / "contract_report.json", {})
    synth = _load_json(DATA_DIR / "synthetic_worlds.json", {})
    infinity = _load_json(DATA_DIR / "infinity_score.json", {})
    cal = _load_json(DATA_DIR / "calibration_curve.json", {})

    bundle["n_eff"] = (
        infinity.get("diagnostics", {}).get("mean_n_eff")
        or cal.get("mean_n_eff")
        or contract.get("n_used")
    )
    bundle["interval_width"] = (
        infinity.get("diagnostics", {}).get("interval_width_mean")
        or cal.get("interval_width_mean")
    )
    bundle["synthetic_noise_conf_mean"] = (
        synth.get("noise_world_confidence_mean_calibrated")
        or synth.get("noise_world_confidence_mean")
        or infinity.get("diagnostics", {}).get("synthetic_noise_conf_cal")
    )
    bundle["drift_detected"] = bool(
        infinity.get("diagnostics", {}).get("drift_detected")
        or _load_json(DATA_DIR / "live_monitor.json", {}).get("drift_flags", {}).get("drift")
    )
    bundle["contract_status"] = contract.get("contract_status", "UNKNOWN")

    return bundle


def apply_guard(prediction: dict, diagnostics: dict | None = None) -> dict:
    """
    Apply hallucination guard. Never increases confidence. Never shrinks window.
    Returns adjusted copy with confidence_before, confidence_after, clamped, reasons, window_start, window_end.
    """
    pred = dict(prediction)
    diag = diagnostics or load_diagnostics_bundle()

    # Normalize prediction keys (confidence_pct vs confidence)
    conf = pred.get("confidence_pct") or pred.get("confidence")
    if conf is None:
        conf = 50
    conf = float(conf)
    ws = pred.get("window_start")
    we = pred.get("window_end")
    interval_width = pred.get("window_end", 0) - pred.get("window_start", 0) if (ws is not None and we is not None) else diag.get("interval_width")
    n_eff = pred.get("analogue_count") or pred.get("n_eff") or diag.get("n_eff")
    if n_eff is not None:
        n_eff = float(n_eff)
    synth_conf = diag.get("synthetic_noise_conf_mean")
    if synth_conf is not None:
        synth_conf = float(synth_conf)
    drift_detected = bool(diag.get("drift_detected", False))

    reasons = []
    new_conf = conf

    # Rule 1 — Noise overconfidence
    if synth_conf is not None and synth_conf > 0.70 and conf > 75:
        new_conf = min(new_conf, 65)
        reasons.append("noise_overconfidence")

    # Rule 2 — Low support
    if n_eff is not None and n_eff < 5 and conf > 70:
        new_conf = min(new_conf, 60)
        reasons.append("low_n_eff")

    # Rule 3 — Drift detected
    if drift_detected:
        new_conf = min(new_conf, 55)
        reasons.append("drift_detected")

    # Rule 4 — Interval too wide but confidence high
    if interval_width is not None and interval_width > 5 and conf > 70:
        new_conf = min(new_conf, 60)
        reasons.append("wide_interval_high_conf")

    clamped = new_conf < conf

    # Drift: widen window by 20%
    new_ws, new_we = ws, we
    if drift_detected and ws is not None and we is not None:
        width = we - ws
        pad = max(1, int(round(width * 0.20)))
        new_ws = ws - pad
        new_we = we + pad

    return {
        "confidence_before": round(conf, 2),
        "confidence_after": round(new_conf, 2),
        "clamped": clamped,
        "reasons": reasons,
        "window_start": new_ws,
        "window_end": new_we,
    }
