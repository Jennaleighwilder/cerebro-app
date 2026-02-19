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
        # METHOD
        patterns_method = [
            "What is the exact mathematical rule used to derive the 2027–2032 peak window?",
            "What thresholds are used?",
            "How is the peak computed?",
        ]
        for q in patterns_method:
            self.assertEqual(_classify_intent(q), "METHOD")

        # VALIDATION
        patterns_validation = [
            "How many historical saddle events were used?",
            "What is the average forecast error in years?",
            "What is the backtest coverage?",
        ]
        for q in patterns_validation:
            self.assertEqual(_classify_intent(q), "VALIDATION")


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
