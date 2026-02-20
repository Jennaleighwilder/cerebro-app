#!/usr/bin/env python3
"""
Unit tests for Cerebro Oracle: intent classification, saddle detection, peak window.
Run: python tests/test_oracle.py
"""

import sys
import unittest
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestOracle(unittest.TestCase):
    def test_saddle_detection(self):
        """Saddle: |v| < v_thresh AND sign(a) opposes sign(v)."""
        from cerebro_peak_window import detect_saddle_canonical

        # v negative, a positive -> opposes (deceleration toward turning point)
        is_sad, intensity = detect_saddle_canonical(0, -0.1, 0.05, None)
        self.assertTrue(is_sad)
        self.assertTrue(0 <= intensity <= 1)

        # v positive, a negative -> opposes
        is_sad, _ = detect_saddle_canonical(0, 0.1, -0.05, None)
        self.assertTrue(is_sad)

        # v and a same sign -> no saddle
        is_sad, _ = detect_saddle_canonical(0, 0.1, 0.05, None)
        self.assertFalse(is_sad)

        # |v| too large -> no saddle
        is_sad, _ = detect_saddle_canonical(0, -0.5, 0.1, None)
        self.assertFalse(is_sad)

    def test_weighted_median(self):
        """Weighted median of values."""
        from cerebro_peak_window import weighted_median

        self.assertEqual(weighted_median([1, 2, 3], [1, 1, 1]), 2)
        self.assertEqual(weighted_median([1, 2, 3], [0, 1, 0]), 2)
        self.assertEqual(weighted_median([1, 5, 10], [1, 2, 1]), 5)

    def test_weighted_quantile(self):
        """Weighted quantile."""
        from cerebro_peak_window import weighted_quantile

        vals = [1, 2, 3, 4, 5]
        weights = [1.0] * 5
        self.assertEqual(weighted_quantile(vals, weights, 0.5), 3)
        self.assertLessEqual(weighted_quantile(vals, weights, 0.25), 3)
        self.assertGreaterEqual(weighted_quantile(vals, weights, 0.75), 3)

    def test_peak_window_compute(self):
        """Peak window returns dict with required keys."""
        from cerebro_peak_window import compute_peak_window

        result = compute_peak_window(2022, -1.5, 0.08, 0.05, None, [])
        self.assertIn("peak_year", result)
        self.assertIn("window_start", result)
        self.assertIn("window_end", result)
        self.assertIn("confidence_pct", result)
        self.assertGreaterEqual(result["peak_year"], 2022)

    def test_method_equations(self):
        """Method equations are non-empty."""
        from cerebro_peak_window import get_method_equations

        eq = get_method_equations()
        self.assertIn("saddle_rule", eq)
        self.assertIn("peak_window_rule", eq)
        self.assertIn("thresholds", eq)
        self.assertIn("v_thresh", eq["thresholds"])
        self.assertGreater(len(eq["saddle_rule"]), 20)

    def test_intent_classification(self):
        """Intent classification patterns (mirror of frontend)."""
        patterns_method = [
            "What is the exact mathematical rule used to derive the 2027–2032 peak window?",
            "What thresholds are used?",
            "How is the peak computed?",
        ]
        for q in patterns_method:
            self.assertEqual(_classify_intent(q), "METHOD")
        patterns_validation = [
            "How many historical saddle events were used?",
            "What is the average forecast error in years?",
            "What is the backtest coverage?",
        ]
        for q in patterns_validation:
            self.assertEqual(_classify_intent(q), "VALIDATION")

    def test_quantile_monotonicity(self):
        """p10 <= p50 <= p90 for weighted quantiles."""
        from cerebro_peak_window import weighted_quantile

        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        weights = [1.0] * len(vals)
        p10 = weighted_quantile(vals, weights, 0.10)
        p50 = weighted_quantile(vals, weights, 0.50)
        p90 = weighted_quantile(vals, weights, 0.90)
        self.assertLessEqual(p10, p50)
        self.assertLessEqual(p50, p90)

    def test_coverage_synthetic(self):
        """Coverage calculation correctness on synthetic data."""
        from cerebro_peak_window import compute_peak_window, weighted_quantile

        # Synthetic: 5 episodes, all with dt=5. Event at saddle_year+5.
        episodes = [
            {"saddle_year": 2000, "event_year": 2005, "position": -1.0, "velocity": -0.1, "acceleration": 0.05},
            {"saddle_year": 2001, "event_year": 2006, "position": -1.2, "velocity": -0.08, "acceleration": 0.04},
            {"saddle_year": 2002, "event_year": 2007, "position": -0.9, "velocity": -0.12, "acceleration": 0.06},
            {"saddle_year": 2003, "event_year": 2008, "position": -1.1, "velocity": -0.09, "acceleration": 0.05},
            {"saddle_year": 2004, "event_year": 2009, "position": -1.0, "velocity": -0.1, "acceleration": 0.05},
        ]
        # Predict from 2004 state; others are analogues. True event 2009.
        pred = compute_peak_window(2004, -1.0, -0.1, 0.05, None, episodes[:-1], interval_alpha=0.8)
        self.assertIn("window_start", pred)
        self.assertIn("window_end", pred)
        # Event 2009 should fall in window (dt=5 is median of 5,5,5,5,5)
        self.assertLessEqual(pred["window_start"], 2009)
        self.assertGreaterEqual(pred["window_end"], 2009)

    def test_saddle_edge_cases(self):
        """Saddle edge cases: v=0, a=0, v==V_THRESH."""
        from cerebro_peak_window import detect_saddle_canonical, V_THRESH

        # v=0: no oppose (opposes requires v>0 and a<0 OR v<0 and a>0; v=0 yields False)
        is_sad, _ = detect_saddle_canonical(0, 0.0, 0.05, None)
        self.assertFalse(is_sad)

        # a=0: no oppose
        is_sad, _ = detect_saddle_canonical(0, -0.1, 0.0, None)
        self.assertFalse(is_sad)

        # v exactly at V_THRESH: below_thresh is v_abs < V_THRESH, so 0.15 < 0.15 is False
        is_sad, _ = detect_saddle_canonical(0, -V_THRESH, 0.05, None)
        self.assertFalse(is_sad)

        # v just below V_THRESH, a opposes: saddle
        is_sad, _ = detect_saddle_canonical(0, -0.14, 0.05, None)
        self.assertTrue(is_sad)

    def test_distance_weighting_sensitivity(self):
        """Closer analogues get higher weight."""
        from cerebro_peak_window import state_distance

        # Same state: dist=0, w=1/(1+0)=1
        d_same = state_distance(0, 0, 0, 0, 0, 0)
        self.assertEqual(d_same, 0.0)

        # Different state: dist > 0
        d_far = state_distance(0, 0, 0, 5, 0.5, 0.1)
        self.assertGreater(d_far, 0)

        # Closer state has smaller distance
        d_close = state_distance(0, 0, 0, 0.1, 0.01, 0.001)
        d_far2 = state_distance(0, 0, 0, 2, 0.2, 0.1)
        self.assertLess(d_close, d_far2)
        # w = 1/(1+dist): closer -> higher w
        w_close = 1.0 / (1.0 + d_close)
        w_far = 1.0 / (1.0 + d_far2)
        self.assertGreater(w_close, w_far)


def _classify_intent(q: str) -> str:
    """Mirror of frontend classifyIntent."""
    l = q.lower()
    if "exact rule" in l or "exact mathematical" in l or "mathematical rule" in l:
        return "METHOD"
    if "what threshold" in l or "how computed" in l or "how is the peak" in l:
        return "METHOD"
    if "how many" in l and ("saddle" in l or "event" in l or "historical" in l):
        return "VALIDATION"
    if "backtest" in l or "average forecast error" in l or "forecast error" in l:
        return "VALIDATION"
    return "OTHER"


if __name__ == "__main__":
    unittest.main()
