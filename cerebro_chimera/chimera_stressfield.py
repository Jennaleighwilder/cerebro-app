#!/usr/bin/env python3
"""
CHIMERA STRESSFIELD — Adversarial perturbations
================================================
Apply perturbations, recompute honeycomb, measure stability.
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_stress_matrix.json"
MIN_TRAIN = 5


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def run_stressfield() -> dict:
    """Apply adversarial perturbations, recompute honeycomb, compute stability."""
    from cerebro_calibration import _load_episodes
    from cerebro_honeycomb import compute_honeycomb_fusion

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 2:
        return {"error": "Insufficient episodes", "version": 1}

    latest = max(episodes, key=lambda e: e.get("saddle_year", 0))
    pool = _past_only_pool(episodes, latest["saddle_year"])
    if len(pool) < MIN_TRAIN:
        return {"error": "Insufficient past", "version": 1}

    now_year = latest["saddle_year"]
    pos = latest.get("position", 0)
    vel = latest.get("velocity", 0)
    acc = latest.get("acceleration", 0)
    rb = latest.get("ring_B_score")

    # Base honeycomb
    base = compute_honeycomb_fusion(now_year, pos, vel, acc, pool, rb)
    base_peak = base.get("peak_year", now_year + 5)
    base_ws = base.get("window_start", now_year + 3)
    base_we = base.get("window_end", now_year + 10)
    base_width = max(1, base_we - base_ws)

    perturbations = [
        ("vel_plus_20", lambda p, v, a: (p, v * 1.2, a)),
        ("vel_minus_20", lambda p, v, a: (p, v * 0.8, a)),
        ("invert_acc", lambda p, v, a: (p, v, -a)),
        ("remove_25pct_analogues", None),
    ]

    stabilities = []
    peak_deviations = []
    window_inflations = []

    for name, perturb_fn in perturbations:
        if perturb_fn is not None:
            p2, v2, a2 = perturb_fn(pos, vel, acc)
            pool_pert = pool
        else:
            p2, v2, a2 = pos, vel, acc
            n_remove = max(1, int(len(pool) * 0.25))
            indices = random.sample(range(len(pool)), min(n_remove, len(pool) - MIN_TRAIN))
            if len(pool) - len(indices) < MIN_TRAIN:
                continue
            pool_pert = [e for i, e in enumerate(pool) if i not in set(indices)]

        try:
            honey = compute_honeycomb_fusion(now_year, p2, v2, a2, pool_pert, rb)
            new_peak = honey.get("peak_year", base_peak)
            new_ws = honey.get("window_start", base_ws)
            new_we = honey.get("window_end", base_we)
        except Exception:
            stabilities.append(0.0)
            peak_deviations.append(999)
            continue

        stability = 1.0 - abs(new_peak - base_peak) / base_width
        stability = max(0.0, min(1.0, stability))
        stabilities.append(stability)
        peak_deviations.append(abs(new_peak - base_peak))
        new_width = new_we - new_ws
        window_inflations.append((new_width - base_width) / base_width if base_width > 0 else 0)

    # Shock probability 2x: run forward sim with 2x shock (simulator handles this)
    # For honeycomb we don't have direct shock control; skip or approximate via acc perturbation
    # Spec says "increase shock probability 2x" - we approximate by doubling acc magnitude
    try:
        honey_shock = compute_honeycomb_fusion(now_year, pos, vel, acc * 2.0, pool, rb)
        new_peak = honey_shock.get("peak_year", base_peak)
        stability = 1.0 - abs(new_peak - base_peak) / base_width
        stability = max(0.0, min(1.0, stability))
        stabilities.append(stability)
        peak_deviations.append(abs(new_peak - base_peak))
    except Exception:
        pass

    mean_stability = sum(stabilities) / len(stabilities) if stabilities else 0.0
    worst_deviation = max(peak_deviations) if peak_deviations else 0
    inflation_rate = sum(window_inflations) / len(window_inflations) if window_inflations else 0.0

    return {
        "version": 1,
        "base_peak": base_peak,
        "base_window_width": base_width,
        "mean_stability": round(mean_stability, 4),
        "worst_case_peak_deviation": worst_deviation,
        "window_inflation_rate": round(inflation_rate, 4),
        "n_perturbations": len(stabilities),
        "stability_scores": [round(s, 4) for s in stabilities],
    }


def main():
    random.seed(42)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_stressfield()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera stressfield: stability={r.get('mean_stability')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
