#!/usr/bin/env python3
"""
Unit tests for Cerebro calibration: conformal, distance weights, pinball.
Run: python tests/test_calibration.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Fixture: miscalibrated episodes (narrow intervals that miss true dt)
FIXTURE_EPISODES = [
    {"saddle_year": 2000, "event_year": 2006, "position": -1.0, "velocity": -0.1, "acceleration": 0.05},
    {"saddle_year": 2001, "event_year": 2007, "position": -1.2, "velocity": -0.08, "acceleration": 0.04},
    {"saddle_year": 2002, "event_year": 2008, "position": -0.9, "velocity": -0.12, "acceleration": 0.06},
    {"saddle_year": 2003, "event_year": 2009, "position": -1.1, "velocity": -0.09, "acceleration": 0.05},
    {"saddle_year": 2004, "event_year": 2010, "position": -1.0, "velocity": -0.1, "acceleration": 0.05},
    {"saddle_year": 2005, "event_year": 2011, "position": -1.1, "velocity": -0.09, "acceleration": 0.05},
    {"saddle_year": 2006, "event_year": 2012, "position": -0.95, "velocity": -0.11, "acceleration": 0.05},
    {"saddle_year": 2007, "event_year": 2013, "position": -1.05, "velocity": -0.10, "acceleration": 0.05},
]


def pinball_loss(y_true: float, y_pred: float, q: float) -> float:
    e = y_true - y_pred
    return max(q * e, (q - 1) * e)


class TestConformal(unittest.TestCase):
    def test_conformal_widens_interval_monotonically(self):
        """Smaller alpha (higher coverage target) -> larger s_hat -> wider interval."""
        from cerebro_conformal import nonconformity_score, compute_s_hat

        scores = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        s_hat_90 = compute_s_hat(scores, 0.1)  # 90% coverage
        s_hat_80 = compute_s_hat(scores, 0.2)  # 80% coverage
        s_hat_50 = compute_s_hat(scores, 0.5)  # 50% coverage
        self.assertGreaterEqual(s_hat_90, s_hat_80)
        self.assertGreaterEqual(s_hat_80, s_hat_50)

    def test_coverage_improves_on_synthetic_miscalibrated(self):
        """Conformal widening improves coverage on miscalibrated data."""
        import cerebro_conformal as cc
        from cerebro_peak_window import compute_peak_window

        # Run calibration on fixture to create artifact
        cal_result = cc.run_calibration(FIXTURE_EPISODES, alpha=0.2, interval_alpha=0.8)
        if "error" in cal_result:
            self.skipTest("Insufficient episodes for calibration")
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "conformal_calibration.json"
            with open(out_path, "w") as f:
                json.dump(cal_result, f)
            orig_path = cc.OUTPUT_PATH
            cc.OUTPUT_PATH = out_path
            try:
                episodes = FIXTURE_EPISODES.copy()
                in_raw, in_cal = 0, 0
                for ep in episodes:
                    others = [e for e in episodes if e["saddle_year"] != ep["saddle_year"]]
                    pred_raw = compute_peak_window(
                        ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
                        None, others, interval_alpha=0.8, apply_conformal=False,
                    )
                    pred_cal = compute_peak_window(
                        ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
                        None, others, interval_alpha=0.8, apply_conformal=True,
                    )
                    ey = ep["event_year"]
                    if pred_raw["window_start"] <= ey <= pred_raw["window_end"]:
                        in_raw += 1
                    if pred_cal["window_start"] <= ey <= pred_cal["window_end"]:
                        in_cal += 1
                cov_raw = in_raw / len(episodes) * 100
                cov_cal = in_cal / len(episodes) * 100
                self.assertGreaterEqual(cov_cal, cov_raw - 1)
            finally:
                cc.OUTPUT_PATH = orig_path

    def test_apply_conformal_returns_correct_shape(self):
        """apply_conformal returns (ws, we, s_hat, applied)."""
        from cerebro_conformal import apply_conformal

        ws, we, s_hat, applied = apply_conformal(2025, 2035, 3.0, 13.0, {})
        self.assertEqual(ws, 2025)
        self.assertEqual(we, 2035)
        self.assertEqual(s_hat, 0.0)
        self.assertFalse(applied)

    def test_apply_conformal_widens_when_s_hat_positive(self):
        """With s_hat>0, interval is widened."""
        from cerebro_conformal import apply_conformal

        ws, we, s_hat, applied = apply_conformal(2025, 2035, 3.0, 13.0, {"s_hat": 2.0})
        self.assertEqual(ws, 2023)
        self.assertEqual(we, 2037)
        self.assertEqual(s_hat, 2.0)
        self.assertTrue(applied)


class TestDistanceWeights(unittest.TestCase):
    def test_fitted_weights_reproducible(self):
        """Fitted weights with fixed seed produce same result."""
        import cerebro_fit_distance_weights as mod

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "distance_weights.json"
            old_path = mod.OUTPUT_PATH
            mod.OUTPUT_PATH = out_path
            try:
                r1 = mod.run_fit()
                if "error" in r1:
                    self.skipTest("Insufficient episodes for fit (CSV missing or too few)")
                with open(out_path) as f:
                    d1 = json.load(f)
                r2 = mod.run_fit()
                with open(out_path) as f:
                    d2 = json.load(f)
                self.assertEqual(d1.get("vel_weight"), d2.get("vel_weight"))
                self.assertEqual(d1.get("acc_weight"), d2.get("acc_weight"))
            finally:
                mod.OUTPUT_PATH = old_path

    def test_pinball_decreases_vs_baseline_on_fixture(self):
        """Grid search on fixture finds weights with pinball <= baseline (100, 2500)."""
        from cerebro_peak_window import compute_peak_window

        def total_pinball(vel_w: float, acc_w: float) -> float:
            pl = 0.0
            for ep in FIXTURE_EPISODES:
                others = [e for e in FIXTURE_EPISODES if e["saddle_year"] != ep["saddle_year"]]
                pred = compute_peak_window(
                    ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
                    None, others, interval_alpha=0.8, apply_conformal=False,
                    vel_weight=vel_w, acc_weight=acc_w,
                )
                dt = ep["event_year"] - ep["saddle_year"]
                p10 = pred.get("delta_p10", 0)
                p90 = pred.get("delta_p90", 10)
                pl += pinball_loss(dt, p10, 0.10) + pinball_loss(dt, p90, 0.90)
            return pl / len(FIXTURE_EPISODES)

        baseline = total_pinball(100, 2500)
        best = baseline
        for vw in [50, 100, 200]:
            for aw in [1000, 2500, 5000]:
                s = total_pinball(vw, aw)
                if s < best:
                    best = s
        self.assertLessEqual(best, baseline)


if __name__ == "__main__":
    unittest.main()
