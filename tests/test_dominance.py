#!/usr/bin/env python3
"""Tests for cerebro v2 dominance modules."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestDominance(unittest.TestCase):
    def test_energy(self):
        from cerebro_energy import energy, release_risk, compute_energy_metrics
        self.assertGreater(energy(0.1, 0.5), 0)
        self.assertEqual(release_risk(False, 0.1, 0), "LOW")
        self.assertEqual(release_risk(True, 0.6, -0.1), "HIGH")
        m = compute_energy_metrics(0.5, 0.1, 0.02, True)
        self.assertIn("energy_score", m)
        self.assertIn(m["release_risk"], ("LOW", "MODERATE", "HIGH"))

    def test_coupling(self):
        from cerebro_coupling import count_saddles_active, joint_descent, systemic_load, systemic_instability_index
        clocks = [{"saddle_score": 2, "velocity": -0.1, "acceleration": -0.05}]
        self.assertEqual(count_saddles_active(clocks), 1)
        self.assertEqual(joint_descent(clocks), 1)
        self.assertLessEqual(0, systemic_load(clocks))
        self.assertLessEqual(systemic_load(clocks), 1)
        self.assertLessEqual(0, systemic_instability_index(clocks))
        self.assertLessEqual(systemic_instability_index(clocks), 1)

    def test_core(self):
        from cerebro_core import detect_saddle_canonical, weighted_median, weighted_quantile, state_distance, compute_peak_window
        is_sad, _ = detect_saddle_canonical(0, -0.1, 0.05, None)
        self.assertTrue(is_sad)
        self.assertEqual(weighted_median([1, 2, 3], [1, 1, 1]), 2)
        self.assertEqual(weighted_quantile([1, 2, 3, 4, 5], [1] * 5, 0.5), 3)
        self.assertGreater(state_distance(0, 0, 0, 1, 0, 0), 0)
        pw = compute_peak_window(2022, -1.0, -0.1, 0.05, None, [])
        self.assertIn("peak_year", pw)
        self.assertIn("window_start", pw)

    def test_peak_window_wrapper(self):
        from cerebro_peak_window import compute_peak_window, get_method_equations
        pw = compute_peak_window(2022, -1.0, -0.1, 0.05, None, [], apply_conformal=False)
        self.assertIn("peak_year", pw)
        eq = get_method_equations()
        self.assertIn("saddle_rule", eq)

    def test_integrity(self):
        from cerebro_integrity import compute_integrity
        r = compute_integrity()
        self.assertIn("average_integrity", r)
        self.assertIn(r["confidence_cap"], ("LOW", "MEDIUM", "HIGH"))


if __name__ == "__main__":
    unittest.main()
