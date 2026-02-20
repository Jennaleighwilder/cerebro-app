#!/usr/bin/env python3
"""
CHIMERA Honeycomb — Capability graph + fitness.
Cells = capabilities, edges = dependencies, fitness from artifacts or default 0.5.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def _fitness_from_artifact(name: str) -> float:
    """Compute fitness from existing artifact if present; else 0.5."""
    defaults = {
        "walkforward": 0.5,
        "calibration": 0.5,
        "stress": 0.5,
        "integrity": 0.5,
        "baselines": 0.5,
        "parameter_stability": 0.5,
        "infinity_score": 0.5,
    }
    def _fw(d):
        me = (d.get("mean_error", 10) or 0) / 10
        cov = (d.get("coverage_80", 0) or 0) / 100
        return 1.0 - min(1, me) * 0.5 + 0.5 * min(1, cov)

    def _cal(d):
        b = (d.get("brier_score", 0.25) or 0.25) / 0.25
        c = d.get("coverage_80", 0) or 0
        return 1.0 - min(1, b) * 0.5 + 0.5 * min(1, c)

    def _stress(d):
        r = (d.get("window_robustness", 0) or 0) / 100
        return (1 if not d.get("instability_flag") else 0.5) * min(1, r)

    mapping = {
        "walkforward": ("walkforward_metrics.json", _fw),
        "calibration": ("calibration_curve.json", _cal),
        "stress": ("stress_test.json", _stress),
        "integrity": ("integrity_scores.json", lambda d: float(d.get("average_integrity", 0.5) or 0.5)),
        "baselines": ("baseline_comparison.json", lambda d: 0.8 if d.get("cerebro_beats_all") else 0.5),
        "parameter_stability": ("parameter_stability.json", lambda d: 1.0 - min(1, (d.get("mae_surface_variance", 0.5) or 0.5) * 2)),
        "infinity_score": ("infinity_score.json", lambda d: min(1, (d.get("infinity_score", 0) or 0) / 200)),
    }
    if name not in mapping:
        return defaults.get(name, 0.5)
    fname, fn = mapping[name]
    p = DATA_DIR / fname
    if not p.exists():
        return defaults.get(name, 0.5)
    try:
        with open(p) as f:
            d = json.load(f)
        return max(0, min(1, float(fn(d))))
    except Exception:
        return defaults.get(name, 0.5)


def build_honeycomb() -> dict:
    """Build/update honeycomb graph with cells, edges, fitness."""
    cells = {
        "walkforward": {"fitness": _fitness_from_artifact("walkforward"), "depends_on": ["episodes", "events"]},
        "calibration": {"fitness": _fitness_from_artifact("calibration"), "depends_on": ["walkforward"]},
        "stress": {"fitness": _fitness_from_artifact("stress"), "depends_on": ["peak_window"]},
        "integrity": {"fitness": _fitness_from_artifact("integrity"), "depends_on": ["sources"]},
        "baselines": {"fitness": _fitness_from_artifact("baselines"), "depends_on": ["walkforward"]},
        "parameter_stability": {"fitness": _fitness_from_artifact("parameter_stability"), "depends_on": ["calibration", "walkforward"]},
        "infinity_score": {"fitness": _fitness_from_artifact("infinity_score"), "depends_on": ["calibration", "walkforward", "integrity", "stress", "baselines"]},
    }
    edges = []
    for cell, meta in cells.items():
        for dep in meta.get("depends_on", []):
            edges.append([dep, cell])
    return {
        "cells": cells,
        "edges": edges,
        "last_updated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
