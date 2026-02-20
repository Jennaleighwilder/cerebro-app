#!/usr/bin/env python3
"""
CHIMERA FAILURE MODE CLASSIFIER — Rule-based (deterministic)
============================================================
Sparse_Analogues, Undercoverage, OOD_Extrapolation, Structural_Disagreement, Fragile_System.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_failure.json"


def run_failure_modes() -> dict:
    """Classify failure mode from reconstruction, stress, distribution shift, honeycomb."""
    failure_mode = "NONE"
    severity = 0.0
    signals = {}

    # Load reconstruction
    rec_path = SCRIPT_DIR / "cerebro_data" / "chimera_reconstruction.json"
    if rec_path.exists():
        try:
            with open(rec_path) as f:
                rec = json.load(f)
            records = rec.get("records", [])
            n_effs = [r.get("n_eff", 0) for r in records if r.get("n_eff") is not None]
            mean_n_eff = sum(n_effs) / len(n_effs) if n_effs else 0
            coverage = rec.get("coverage_80_mean", 0)
            disagreement = [r.get("disagreement_std", 0) for r in records if r.get("disagreement_std") is not None]
            disagreement_std = max(disagreement) if disagreement else 0
            signals["mean_n_eff"] = round(mean_n_eff, 2)
            signals["coverage_80"] = coverage
            signals["disagreement_std"] = round(disagreement_std, 2)
        except Exception:
            mean_n_eff = 10
            coverage = 0.8
            disagreement_std = 0
    else:
        mean_n_eff = 10
        coverage = 0.8
        disagreement_std = 0

    # Load distribution shift (OOD)
    shift_path = SCRIPT_DIR / "cerebro_data" / "distribution_shift.json"
    ood_percentile = 0.5
    if shift_path.exists():
        try:
            with open(shift_path) as f:
                shift = json.load(f)
            ood_percentile = float(shift.get("percentile", 0.5))
            signals["ood_percentile"] = ood_percentile
        except Exception:
            pass

    # Load stress
    stress_path = SCRIPT_DIR / "cerebro_data" / "chimera_stress_matrix.json"
    stability = 1.0
    if stress_path.exists():
        try:
            with open(stress_path) as f:
                stress = json.load(f)
            stability = float(stress.get("mean_stability", 1.0))
            signals["stress_stability"] = stability
        except Exception:
            pass

    # Rules (priority order)
    if mean_n_eff < 7:
        failure_mode = "Sparse_Analogues"
        severity = min(1.0, 0.5 + (7 - mean_n_eff) / 10)
    if coverage < 0.65:
        s = 0.5 + (0.65 - coverage)
        if s > severity:
            failure_mode = "Undercoverage"
            severity = min(1.0, s)
    if ood_percentile > 0.97:
        s = 0.7 + (ood_percentile - 0.97) * 10
        if s > severity:
            failure_mode = "OOD_Extrapolation"
            severity = min(1.0, s)
    if disagreement_std > 4:
        s = 0.5 + (disagreement_std - 4) / 10
        if s > severity:
            failure_mode = "Structural_Disagreement"
            severity = min(1.0, s)
    if stability < 0.5:
        s = 0.6 + (0.5 - stability)
        if s > severity:
            failure_mode = "Fragile_System"
            severity = min(1.0, s)

    return {
        "version": 1,
        "failure_mode": failure_mode,
        "severity": round(severity, 4),
        "signals": signals,
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_failure_modes()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera failure: {r.get('failure_mode')} severity={r.get('severity')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
