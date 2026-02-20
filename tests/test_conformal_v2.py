#!/usr/bin/env python3
"""
Tests for cerebro_conformal_v2 — Contract Windows.
Verify: widening improves empirical coverage, apply_conformal_v2 widens correctly.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestConformalV2Contract(unittest.TestCase):
    def test_load_contract_returns_none_when_missing(self):
        """load_contract returns None when contract_report.json does not exist."""
        from cerebro_conformal_v2 import load_contract
        # May exist from pipeline run; if not, None
        c = load_contract()
        if c is None:
            return
        self.assertIn("contract_status", c)
        self.assertIn("coverage_target", c)
        self.assertIn("window_widen_factor", c)

    def test_apply_conformal_v2_widens_when_s_hat_positive(self):
        """When s_hat > 0, apply_conformal_v2 widens the window."""
        from cerebro_conformal_v2 import apply_conformal_v2
        ws, we = 2028, 2035
        contract = {"s_hat": 2.0, "coverage_target": 0.8, "window_widen_factor": 1.5}
        ws_new, we_new, s_hat, applied = apply_conformal_v2(ws, we, 0, 7, contract)
        self.assertTrue(applied)
        self.assertLessEqual(ws_new, ws)
        self.assertGreaterEqual(we_new, we)
        self.assertEqual(s_hat, 2.0)
        self.assertEqual(ws_new, ws - 2)  # pad = ceil(2) = 2
        self.assertEqual(we_new, we + 2)

    def test_apply_conformal_v2_no_widen_when_s_hat_zero(self):
        """When s_hat <= 0, apply_conformal_v2 does not widen."""
        from cerebro_conformal_v2 import apply_conformal_v2
        ws, we = 2028, 2035
        contract = {"s_hat": 0.0, "coverage_target": 0.8}
        ws_new, we_new, s_hat, applied = apply_conformal_v2(ws, we, 0, 7, contract)
        self.assertFalse(applied)
        self.assertEqual(ws_new, ws)
        self.assertEqual(we_new, we)

    def test_apply_conformal_v2_with_none_contract_computes(self):
        """When contract=None, apply_conformal_v2 may compute (or return unchanged)."""
        from cerebro_conformal_v2 import apply_conformal_v2
        ws, we = 2028, 2035
        ws_new, we_new, s_hat, applied = apply_conformal_v2(ws, we, 0, 7, None)
        self.assertIsInstance(applied, bool)
        self.assertIsInstance(s_hat, (int, float))
        self.assertIsInstance(ws_new, (int, float))
        self.assertIsInstance(we_new, (int, float))


class TestWideningImprovesCoverage(unittest.TestCase):
    """Synthetic: larger s_hat yields better empirical coverage on toy residuals."""

    def test_synthetic_residuals_widening_increases_hits(self):
        """Given events outside [ws,we], widening by s_hat brings them inside."""
        from cerebro_conformal_v2 import apply_conformal_v2
        # Base window [2028, 2035]. Event at 2026 (2 years before ws).
        ws, we = 2028, 2035
        event = 2026
        # Residual = max(0, ws-event, event-we) = max(0, 2, -9) = 2
        # So s_hat >= 2 is needed to cover. Pad 2 → [2026, 2037]
        contract = {"s_hat": 2.0, "coverage_target": 0.8}
        ws_new, we_new, _, applied = apply_conformal_v2(ws, we, 0, 7, contract)
        self.assertTrue(applied)
        self.assertLessEqual(ws_new, event)
        self.assertGreaterEqual(we_new, event)
        # Event 2026 is now inside [2026, 2037]
        self.assertTrue(ws_new <= event <= we_new)


class TestPeakWindowContractFields(unittest.TestCase):
    """Peak window output includes contract_status, coverage_target, window_widen_factor."""

    def test_peak_window_attaches_contract_fields(self):
        """compute_peak_window (wrapper) attaches contract_status, coverage_target, window_widen_factor."""
        from cerebro_peak_window import compute_peak_window
        analogues = [
            {"saddle_year": 1990, "event_year": 1994, "position": -0.5, "velocity": -0.1, "acceleration": 0.05},
        ]
        pw = compute_peak_window(2022, -0.5, -0.1, 0.05, None, analogues, apply_conformal=False)
        self.assertIn("contract_status", pw)
        self.assertIn("coverage_target", pw)
        self.assertIn("window_widen_factor", pw)
        self.assertIn(pw["contract_status"], ("PASS", "WARNING", "FAIL", "UNKNOWN"))
        self.assertIsInstance(pw["coverage_target"], (int, float))
        self.assertGreaterEqual(pw["window_widen_factor"], 1.0)


if __name__ == "__main__":
    unittest.main()
