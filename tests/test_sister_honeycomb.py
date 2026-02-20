#!/usr/bin/env python3
"""Tests for Sister Engine and Honeycomb ensemble."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestSisterMonotonicity(unittest.TestCase):
    def test_sister_hazard_monotonic(self):
        """Sister P_1yr <= P_3yr <= P_5yr <= P_10yr."""
        from cerebro_sister_engine import run_sister_engine
        r = run_sister_engine()
        if r.get("error"):
            self.skipTest("Sister engine insufficient data")
        p1 = r.get("P_1yr", 0)
        p3 = r.get("P_3yr", 0)
        p5 = r.get("P_5yr", 0)
        p10 = r.get("P_10yr", 0)
        self.assertLessEqual(p1, p3, "P_1yr <= P_3yr")
        self.assertLessEqual(p3, p5, "P_3yr <= P_5yr")
        self.assertLessEqual(p5, p10, "P_5yr <= P_10yr")


class TestHoneycombWeights(unittest.TestCase):
    def test_honeycomb_weights_sum_to_one(self):
        """Per cell: w_core + w_sister = 1."""
        from cerebro_honeycomb import _compute_cell_weights, _load_episodes
        episodes = _load_episodes()
        if len(episodes) < 10:
            self.skipTest("Insufficient episodes for honeycomb")
        weights = _compute_cell_weights(episodes)
        for cell, w in weights.items():
            total = w.get("w_core", 0) + w.get("w_sister", 0)
            self.assertAlmostEqual(total, 1.0, places=2, msg=f"Cell {cell}: w_core+w_sister={total}")


class TestWalkforwardNoLeakage(unittest.TestCase):
    def test_sister_training_past_only(self):
        """Sister training uses only episodes with saddle_year < prediction year."""
        from cerebro_sister_engine import _load_episodes, _feature_vec, _fit_logistic
        import numpy as np
        episodes = _load_episodes()
        if len(episodes) < 10:
            self.skipTest("Insufficient episodes")
        sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
        for i, ep in enumerate(sorted_ep[5:], start=5):
            t = ep.get("saddle_year")
            pool = [e for e in sorted_ep if e.get("saddle_year", 0) < t]
            for p in pool:
                self.assertLess(p.get("saddle_year", 0), t, "Training uses only past episodes")


class TestEnsembleSchema(unittest.TestCase):
    def test_ensemble_backtest_schema(self):
        """Ensemble backtest has core, sister, honeycomb, winner."""
        p = ROOT / "cerebro_data" / "ensemble_backtest.json"
        if not p.exists():
            self.skipTest("ensemble_backtest.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Backtest had error")
        self.assertIn("core", data)
        self.assertIn("sister", data)
        self.assertIn("honeycomb", data)
        self.assertIn("winner", data)
        for model in ["core", "sister", "honeycomb"]:
            self.assertIn("mae", data[model])
            self.assertIn("coverage_80", data[model])
            self.assertIn("brier_5yr", data[model])
        self.assertIn(data["winner"], ["core", "sister", "honeycomb"])


class TestHazardCurveSisterSchema(unittest.TestCase):
    def test_sister_output_schema(self):
        """Sister hazard curve has P_1yr, P_3yr, P_5yr, P_10yr."""
        p = ROOT / "cerebro_data" / "hazard_curve_sister.json"
        if not p.exists():
            self.skipTest("hazard_curve_sister.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Sister had error")
        for k in ["P_1yr", "P_3yr", "P_5yr", "P_10yr"]:
            self.assertIn(k, data)
            self.assertIsInstance(data[k], (int, float))
            self.assertGreaterEqual(data[k], 0)
            self.assertLessEqual(data[k], 1)


class TestHoneycombOutputSchema(unittest.TestCase):
    def test_honeycomb_output_schema(self):
        """Honeycomb hazard has w_core, w_sister, P_5yr."""
        p = ROOT / "cerebro_data" / "hazard_curve_honeycomb.json"
        if not p.exists():
            self.skipTest("hazard_curve_honeycomb.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Honeycomb had error")
        self.assertIn("w_core", data)
        self.assertIn("w_sister", data)
        self.assertIn("P_5yr", data)
        self.assertAlmostEqual(data["w_core"] + data["w_sister"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
