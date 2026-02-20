#!/usr/bin/env python3
"""
CHIMERA Infinity Score v2 — Ops-grade composite metric.
Rewards: accuracy, calibration, coverage, support, robustness.
Punishes: overconfidence, miscalibration, false positives, integrity degradation, parameter fragility.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_PATH = DATA_DIR / "infinity_score.json"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def safe_get(d: dict, path: str | list, default):
    """Get nested key: safe_get(d, 'a.b.c', 0) or safe_get(d, ['a','b','c'], 0)."""
    if isinstance(path, str):
        path = path.split(".")
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return default
    return d


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _compute_ece(bins: list[dict]) -> float:
    """ECE = Σ (n_i/N) * |empirical_i - conf_i|. Uses calibrated_conf_mid when present (post-isotonic ECE)."""
    valid = []
    for b in bins:
        c = b.get("calibrated_conf_mid") if b.get("calibrated_conf_mid") is not None else b.get("conf_mid")
        e = b.get("empirical_hit_rate")
        n = b.get("weighted_n") if b.get("weighted_n") is not None else b.get("n", 0)
        if c is not None and e is not None and n > 0:
            valid.append((c, e, float(n)))
    if not valid:
        return 0.0
    N = sum(n for _, _, n in valid)
    if N <= 0:
        return 0.0
    ece = sum((n / N) * abs(e - c) for c, e, n in valid)
    return float(ece)


def compute_infinity_score() -> dict:
    """Compute Infinity Score v2 from cerebro_data artifacts."""
    cal = _load_json(DATA_DIR / "calibration_curve.json", {})
    wf = _load_json(DATA_DIR / "walkforward_metrics.json", {})
    bl = _load_json(DATA_DIR / "baseline_comparison.json", {})
    stress = _load_json(DATA_DIR / "stress_test.json", {})
    integrity = _load_json(DATA_DIR / "integrity_scores.json", {})
    contract = _load_json(DATA_DIR / "contract_report.json", {})
    param_stab = _load_json(DATA_DIR / "parameter_stability.json", {})
    synth = _load_json(DATA_DIR / "synthetic_worlds.json", {})

    cal_mode = cal.get("mode_operational") or cal
    brier = float(cal_mode.get("brier", cal.get("brier_score", 0.25)) or 0.25)
    coverage_80 = float(cal.get("coverage_80", cal_mode.get("coverage_80", 0.7)) or 0.7)
    n_used = int(cal_mode.get("n_used", cal.get("n_used", 20)) or 20)
    mean_n_eff = float(cal.get("mean_n_eff", n_used) or n_used)
    interval_width_mean = float(cal.get("interval_width_mean", 5.0) or 5.0)

    mae_walkforward = float(wf.get("mean_error", 5) or 5)
    cerebro_beats_all = bool(bl.get("cerebro_beats_all", False))

    _ps = stress.get("peak_std")
    peak_std = float(_ps) if _ps is not None else 2.0
    _wr = stress.get("window_robustness")
    window_robustness = float(_wr) if _wr is not None else 80.0

    average_integrity = float(integrity.get("average_integrity", 0.7) or 0.7)
    mae_surface_variance = float(param_stab.get("mae_surface_variance", 0.1) or 0.1)

    fp_rate = float(synth.get("false_positive_rate", 0) or 0)
    high_noise = synth.get("worlds", {}).get("high_noise", {})
    noise_conf_cal = float(
        high_noise.get("confidence_mean_calibrated")
        or synth.get("noise_world_confidence_mean_calibrated")
        or synth.get("noise_world_confidence_mean", 0.5)
        or 0.5
    )

    # 1) S_acc — walkforward MAE
    S_acc = clamp01(1.0 - mae_walkforward / 4.0)

    # 2) S_cal — Brier + reliability (ECE)
    S_brier = clamp01(1.0 - brier / 0.25)
    ece = _compute_ece(cal.get("bins", []))
    S_rel = clamp01(1.0 - ece / 0.20)
    S_cal = math.sqrt(S_brier * S_rel) if (S_brier > 0 and S_rel > 0) else 0.0

    # 3) S_int — coverage + window width (interval honesty)
    S_cov = clamp01(1.0 - abs(coverage_80 - 0.80) / 0.30)
    S_width = clamp01(1.0 - (interval_width_mean - 3.0) / 6.0) if interval_width_mean >= 3.0 else 1.0
    S_int = math.sqrt(S_cov * S_width) if (S_cov > 0 and S_width > 0) else 0.0

    # 4) S_sup — n_eff + integrity
    S_neff = sigmoid((mean_n_eff - 10) / 2.0)
    S_intg = clamp01(average_integrity)
    S_sup = math.sqrt(S_neff * S_intg) if (S_neff > 0 and S_intg > 0) else 0.0

    # 5) S_rob — synthetic + stability + stress
    S_fp = clamp01(1.0 - fp_rate / 0.25)
    S_noise = clamp01(1.0 - max(0, noise_conf_cal - 0.70) / 0.20)
    S_stab = 1.0 / (1.0 + mae_surface_variance)
    S_peak = clamp01(1.0 - peak_std / 2.0)
    S_win = clamp01(window_robustness / 80.0)
    S_stress = math.sqrt(S_peak * S_win) if (S_peak > 0 and S_win > 0) else 0.0
    S_rob = (S_fp * S_noise * S_stab * S_stress) ** 0.25

    # Core composite
    subscores = [S_acc, S_cal, S_int, S_sup, S_rob]
    G = (S_acc * S_cal * S_int * S_sup * S_rob) ** (1 / 5) if all(s > 0 for s in subscores) else 0.0
    G = clamp01(G)

    # Baseline dominance multiplier
    M_dom = 1.05 if cerebro_beats_all else 0.90

    # Hard penalties
    P = 1.0
    penalties = []
    if average_integrity < 0.7:
        P *= 0.85
        penalties.append("integrity_low")
    if mean_n_eff < 5:
        P *= 0.80
        penalties.append("n_eff_low")
    if n_used < 30:
        P *= 0.85
        penalties.append("n_used_low")
    if noise_conf_cal > 0.8:
        P *= 0.75
        penalties.append("noise_overconfident")

    # Gates (compliance flags)
    gates = {
        "n_used_ge_30": n_used >= 30,
        "mean_n_eff_ge_10": mean_n_eff >= 10,
        "brier_le_020": brier <= 0.20,
        "ece_le_010": ece <= 0.10,
        "synthetic_noise_conf_cal_le_070": noise_conf_cal <= 0.70,
        "fp_rate_le_025": fp_rate <= 0.25,
    }

    InfinityScore = round(100 * math.log(1 + 25 * G * M_dom * P), 2)

    inputs_present = {
        "calibration_curve": (DATA_DIR / "calibration_curve.json").exists(),
        "walkforward_metrics": (DATA_DIR / "walkforward_metrics.json").exists(),
        "baseline_comparison": (DATA_DIR / "baseline_comparison.json").exists(),
        "stress_test": (DATA_DIR / "stress_test.json").exists(),
        "integrity_scores": (DATA_DIR / "integrity_scores.json").exists(),
        "parameter_stability": (DATA_DIR / "parameter_stability.json").exists(),
        "synthetic_worlds": (DATA_DIR / "synthetic_worlds.json").exists(),
    }

    out = {
        "version": 2,
        "infinity_score": InfinityScore,
        "G": round(G, 4),
        "penalty": round(P, 4),
        "signals": {
            "accuracy": round(S_acc, 4),
            "calibration": round(S_cal, 4),
            "interval": round(S_int, 4),
            "support": round(S_sup, 4),
            "robustness": round(S_rob, 4),
        },
        "subscores": {
            "accuracy": round(S_acc, 4),
            "calibration": round(S_cal, 4),
            "interval": round(S_int, 4),
            "support": round(S_sup, 4),
            "robustness": round(S_rob, 4),
        },
        "diagnostics": {
            "brier": round(brier, 4),
            "ece": round(ece, 4),
            "n_used": n_used,
            "coverage_80": round(coverage_80, 4),
            "interval_width_mean": round(interval_width_mean, 2),
            "mean_n_eff": round(mean_n_eff, 2),
            "synthetic_fp_rate": round(fp_rate, 4),
            "synthetic_noise_conf_cal": round(noise_conf_cal, 4),
            "integrity": round(average_integrity, 4),
            "mae_walkforward": round(mae_walkforward, 2),
            "peak_std": round(peak_std, 2),
            "window_robustness": round(window_robustness, 2),
            "contract_status": contract.get("contract_status", "UNKNOWN"),
            "contract_empirical_coverage": round(float(contract.get("empirical_coverage", 0) or 0), 4),
            "contract_window_widen_factor": round(float(contract.get("window_widen_factor", 1) or 1), 4),
        },
        "multipliers": {
            "dominance": M_dom,
            "penalty_product": round(P, 4),
        },
        "penalties_applied": penalties,
        "gates": gates,
        "inputs_present": inputs_present,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out
