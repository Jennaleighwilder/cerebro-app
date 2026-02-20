#!/usr/bin/env python3
"""
Test that causal normalization has no future leakage.
If we append an extreme value to a series, early normalized values must NOT change.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestNoLeakageNormalization(unittest.TestCase):
    def test_expanding_minmax_no_leakage(self):
        """Early index normalized value must not change when future extreme is appended."""
        from cerebro_causal_normalization import expanding_minmax_to_10pt

        # Toy series: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] — last is max
        s1 = {i: float(i) for i in range(1, 11)}
        out1 = expanding_minmax_to_10pt(s1, min_periods=5)
        # At index 5 (value 5): hist = [1,2,3,4,5], min=1, max=5 → (5-1)/(5-1)=1.0 → target range
        # Get value at key 5
        v5_before = out1.get(5)
        self.assertIsNotNone(v5_before)

        # Append extreme: 1000 at end
        s2 = dict(s1)
        s2[11] = 1000.0
        out2 = expanding_minmax_to_10pt(s2, min_periods=5)
        v5_after = out2.get(5)
        # Value at 5 must be identical — we only use data up to 5
        self.assertEqual(v5_before, v5_after, "Early normalized value must not change when future extreme is appended")

    def test_expanding_zscore_no_leakage(self):
        """Early index z-score value must not change when future extreme is appended."""
        from cerebro_causal_normalization import expanding_zscore_to_10pt

        s1 = {i: float(i) for i in range(1, 11)}
        out1 = expanding_zscore_to_10pt(s1, min_periods=5)
        v5_before = out1.get(5)
        self.assertIsNotNone(v5_before)

        s2 = dict(s1)
        s2[11] = 1000.0
        out2 = expanding_zscore_to_10pt(s2, min_periods=5)
        v5_after = out2.get(5)
        self.assertEqual(v5_before, v5_after, "Early z-score value must not change when future extreme is appended")

    def test_norm_causal_no_leakage(self):
        """norm_causal ([-1,+1]) must not leak."""
        from cerebro_causal_normalization import norm_causal

        s1 = {i: float(i) for i in range(1, 11)}
        out1 = norm_causal(s1, min_periods=5)
        v5_before = out1.get(5)
        self.assertIsNotNone(v5_before)

        s2 = dict(s1)
        s2[11] = 1000.0
        out2 = norm_causal(s2, min_periods=5)
        v5_after = out2.get(5)
        self.assertEqual(v5_before, v5_after, "Early norm_causal value must not change when future extreme is appended")


if __name__ == "__main__":
    unittest.main()
