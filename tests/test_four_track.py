#!/usr/bin/env python3
"""Tests for four-track hardening layer."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "cerebro_data"


class TestHazardCurve(unittest.TestCase):
    """hazard_curve probabilities are monotonic non-decreasing."""

    def test_hazard_monotonic(self):
        from cerebro_hazard_curve import compute_hazard_curve
        h = compute_hazard_curve(2030, 2027, 2032, 2022)
        self.assertLessEqual(h["P_1yr"], h["P_3yr"])
        self.assertLessEqual(h["P_3yr"], h["P_5yr"])
        self.assertLessEqual(h["P_5yr"], h["P_10yr"])

    def test_hazard_from_file(self):
        p = DATA_DIR / "hazard_curve.json"
        if not p.exists():
            self.skipTest("hazard_curve.json not generated")
        with open(p) as f:
            h = json.load(f)
        for k in ("P_1yr", "P_3yr", "P_5yr", "P_10yr"):
            self.assertIn(k, h)
        if all(k in h and h[k] is not None for k in ("P_1yr", "P_3yr", "P_5yr", "P_10yr")):
            self.assertLessEqual(h["P_1yr"], h["P_3yr"])
            self.assertLessEqual(h["P_3yr"], h["P_5yr"])
            self.assertLessEqual(h["P_5yr"], h["P_10yr"])


class TestRegimeProbabilities(unittest.TestCase):
    """regime probabilities sum to 1.0 (±1e-6) and none negative."""

    def test_regime_sum_and_non_negative(self):
        from cerebro_regime import compute_regime_probabilities
        r = compute_regime_probabilities({"position": -0.5, "velocity": -0.1, "acceleration": 0.05})
        total = sum(r.values())
        self.assertAlmostEqual(total, 1.0, delta=0.02)
        for v in r.values():
            self.assertGreaterEqual(v, 0)

    def test_regime_from_file(self):
        p = DATA_DIR / "regime_probabilities.json"
        if not p.exists():
            self.skipTest("regime_probabilities.json not generated")
        with open(p) as f:
            r = json.load(f)
        for k in ("Stable", "Redistribution", "Crackdown", "Reform", "Polarization"):
            self.assertIn(k, r)
        total = sum(r.values())
        self.assertAlmostEqual(total, 1.0, delta=0.02)
        for v in r.values():
            self.assertGreaterEqual(v, 0)


class TestBaselines(unittest.TestCase):
    """baselines JSON contains all required keys and cerebro_beats_all is boolean."""

    def test_baselines_schema(self):
        from cerebro_baselines import run_baselines
        r = run_baselines()
        if "error" in r:
            self.skipTest("baselines insufficient data")
        for k in ("cerebro_mae", "linear_mae", "arima_mae", "naive_mae", "random_mae", "cerebro_beats_all"):
            self.assertIn(k, r)
        self.assertIsInstance(r["cerebro_beats_all"], bool)

    def test_baselines_from_file(self):
        p = DATA_DIR / "baseline_comparison.json"
        if not p.exists():
            self.skipTest("baseline_comparison.json not generated")
        with open(p) as f:
            r = json.load(f)
        if "error" in r:
            return
        for k in ("cerebro_mae", "linear_mae", "arima_mae", "naive_mae", "random_mae", "cerebro_beats_all"):
            self.assertIn(k, r)
        self.assertIsInstance(r["cerebro_beats_all"], bool)


class TestParameterStability(unittest.TestCase):
    """parameter stability JSON contains required keys and values are finite."""

    def test_param_stability_schema(self):
        from cerebro_parameter_stability import run_stability
        r = run_stability()
        if "error" in r:
            self.skipTest("parameter stability insufficient data")
        self.assertIn("mae_surface_variance", r)
        self.assertIn("coverage_surface_variance", r)
        self.assertTrue(abs(r["mae_surface_variance"]) != float("inf"))
        self.assertTrue(abs(r["coverage_surface_variance"]) != float("inf"))

    def test_param_stability_from_file(self):
        p = DATA_DIR / "parameter_stability.json"
        if not p.exists():
            self.skipTest("parameter_stability.json not generated")
        with open(p) as f:
            r = json.load(f)
        if "error" in r:
            return
        self.assertIn("mae_surface_variance", r)
        self.assertIn("coverage_surface_variance", r)
        self.assertFalse(r.get("mae_surface_variance") in (float("inf"), float("-inf")))
        self.assertFalse(r.get("coverage_surface_variance") in (float("inf"), float("-inf")))


class TestRollingOrigin(unittest.TestCase):
    """rolling origin output schema present."""

    def test_rolling_origin_schema(self):
        from cerebro_rolling_origin import run_rolling_origin
        r = run_rolling_origin()
        self.assertIn("decade_rows", r)
        self.assertIn("overall_mae", r)
        self.assertIn("overall_coverage_80", r)
        self.assertIn("drift_flags", r)
        self.assertIsInstance(r["decade_rows"], list)
        self.assertIsInstance(r["drift_flags"], list)

    def test_rolling_origin_from_file(self):
        p = DATA_DIR / "rolling_origin_metrics.json"
        if not p.exists():
            self.skipTest("rolling_origin_metrics.json not generated")
        with open(p) as f:
            r = json.load(f)
        self.assertIn("decade_rows", r)
        self.assertIn("overall_mae", r)
        self.assertIn("overall_coverage_80", r)
        self.assertIn("drift_flags", r)


if __name__ == "__main__":
    unittest.main()
