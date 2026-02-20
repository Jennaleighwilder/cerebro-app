#!/usr/bin/env python3
"""
CHIMERA Infinity Score — Composite metric: is the system getting stronger over time?
Geometric mean of normalized signals + penalties. 0–80 prototype, 80–140 operational, 140–200 strong, 200+ scary good.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_PATH = DATA_DIR / "infinity_score.json"


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_infinity_score() -> dict:
    """Compute Infinity Score from cerebro_data artifacts."""
    cal = _load_json(DATA_DIR / "calibration_curve.json", {})
    wf = _load_json(DATA_DIR / "walkforward_metrics.json", {})
    bl = _load_json(DATA_DIR / "baseline_comparison.json", {})
    stress = _load_json(DATA_DIR / "stress_test.json", {})
    integrity = _load_json(DATA_DIR / "integrity_scores.json", {})
    param_stab = _load_json(DATA_DIR / "parameter_stability.json", {})

    # Use operational or top-level
    cal_mode = cal.get("mode_operational") or cal
    brier = float(cal_mode.get("brier", cal.get("brier_score", 0.25)) or 0.25)
    coverage_80 = float(cal.get("coverage_80", cal_mode.get("coverage_80", 0.7)) or 0.7)
    mean_n_eff = float(cal.get("mean_n_eff", 7) or 7)
    interval_width_mean = float(cal.get("interval_width_mean", 5) or 5)

    walkforward_mean_error = float(wf.get("mean_error", 5) or 5)
    wf_coverage_80 = float(wf.get("coverage_80", 70) or 70) / 100.0 if wf.get("coverage_80") is not None else coverage_80

    cerebro_beats_all = bool(bl.get("cerebro_beats_all", False))
    cerebro_mae = float(bl.get("cerebro_mae", 5) or 5)

    peak_std = float(stress.get("peak_std", 2) or 2)
    window_robustness = float(stress.get("window_robustness", 80) or 80)
    instability_flag = bool(stress.get("instability_flag", False))

    average_integrity = float(integrity.get("average_integrity", 0.7) or 0.7)
    confidence_cap = str(integrity.get("confidence_cap", "HIGH") or "HIGH")

    mae_surface_variance = float(param_stab.get("mae_surface_variance", 0.1) or 0.1)
    cal_mode_strict = cal.get("mode_strict") or {}
    n_used_strict = int(cal_mode_strict.get("n_used", 20) or 20)

    # Normalize to [0,1]
    S_skill = math.exp(-walkforward_mean_error / 2.0)
    S_cal = math.exp(-brier / 0.20)
    S_cov = _clamp((coverage_80 - 0.60) / 0.40, 0, 1)
    S_neff = _clamp((mean_n_eff - 5) / 10, 0, 1)
    S_stress = _clamp(window_robustness / 100, 0, 1) * math.exp(-peak_std / 2)
    S_int = _clamp(average_integrity, 0, 1)
    S_stab = math.exp(-mae_surface_variance / 0.25)
    S_dom = 1.0 if cerebro_beats_all else 0.6

    signals = {
        "S_skill": round(S_skill, 4),
        "S_cal": round(S_cal, 4),
        "S_cov": round(S_cov, 4),
        "S_neff": round(S_neff, 4),
        "S_stress": round(S_stress, 4),
        "S_int": round(S_int, 4),
        "S_stab": round(S_stab, 4),
        "S_dom": round(S_dom, 4),
    }

    G_raw = (S_skill * S_cal * S_cov * S_neff * S_stress * S_int * S_stab * S_dom) ** (1 / 8)
    G = max(0, min(1, G_raw))

    P = 1.0
    if instability_flag:
        P *= 0.85
    if confidence_cap == "MEDIUM":
        P *= 0.90
    if n_used_strict < 10:
        P *= 0.90

    InfinityScore = 100 * math.log(1 + 25 * G * P)

    inputs_present = {
        "calibration_curve": (DATA_DIR / "calibration_curve.json").exists(),
        "walkforward_metrics": (DATA_DIR / "walkforward_metrics.json").exists(),
        "baseline_comparison": (DATA_DIR / "baseline_comparison.json").exists(),
        "stress_test": (DATA_DIR / "stress_test.json").exists(),
        "integrity_scores": (DATA_DIR / "integrity_scores.json").exists(),
        "parameter_stability": (DATA_DIR / "parameter_stability.json").exists(),
    }

    out = {
        "infinity_score": round(InfinityScore, 1),
        "G": round(G, 4),
        "penalty": round(P, 4),
        "signals": signals,
        "inputs_present": inputs_present,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out
