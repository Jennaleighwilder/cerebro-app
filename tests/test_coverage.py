#!/usr/bin/env python3
"""Extended tests for 85% coverage."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestCoreCoverage(unittest.TestCase):
    def test_weighted_median_empty(self):
        from cerebro_core import weighted_median
        self.assertEqual(weighted_median([], []), 0.0)
        self.assertEqual(weighted_median([1], []), 0.0)
        self.assertEqual(weighted_median([], [1]), 0.0)
        self.assertEqual(weighted_median([1, 2], [1]), 0.0)  # len mismatch

    def test_weighted_median_zero_total(self):
        from cerebro_core import weighted_median
        self.assertEqual(weighted_median([1, 2, 3], [0, 0, 0]), 2)  # fallback to middle

    def test_weighted_quantile_empty(self):
        from cerebro_core import weighted_quantile
        self.assertEqual(weighted_quantile([], [], 0.5), 0.0)
        self.assertEqual(weighted_quantile([1], [], 0.5), 0.0)

    def test_weighted_quantile_zero_total(self):
        from cerebro_core import weighted_quantile
        self.assertEqual(weighted_quantile([1, 2, 3], [0, 0, 0], 0.5), 1)

    def test_detect_saddle_velocity_positive(self):
        from cerebro_core import detect_saddle_canonical
        is_sad, _ = detect_saddle_canonical(0, 0.1, -0.05, None)  # v>0, a<0 opposes
        self.assertTrue(is_sad)

    def test_state_distance_custom_weights(self):
        from cerebro_core import state_distance
        d = state_distance(0, 0, 0, 1, 1, 1, vel_weight=10, acc_weight=100)
        self.assertGreater(d, 0)

    def test_compute_peak_window_with_analogues(self):
        from cerebro_core import compute_peak_window
        analogues = [
            {"saddle_year": 1990, "event_year": 1994, "position": -0.5, "velocity": -0.1, "acceleration": 0.05},
            {"saddle_year": 1985, "event_year": 1988, "position": -0.4, "velocity": -0.08, "acceleration": 0.03},
        ]
        pw = compute_peak_window(2022, -0.5, -0.1, 0.05, None, analogues, interval_alpha=0.5)
        self.assertIn("peak_year", pw)
        self.assertEqual(pw["interval_alpha"], 0.5)
        self.assertEqual(pw["window_label"], "50% window")

    def test_compute_peak_window_ring_b_boost(self):
        from cerebro_core import compute_peak_window
        analogues = [{"saddle_year": 1990, "event_year": 1994, "position": -0.5, "velocity": -0.1, "acceleration": 0.05}]
        pw = compute_peak_window(2022, -0.5, -0.1, 0.05, 0.8, analogues)  # ring_b strong
        self.assertIn("confidence_pct", pw)

    def test_compute_peak_window_single_delta(self):
        from cerebro_core import compute_peak_window
        analogues = [{"saddle_year": 1990, "event_year": 1994, "position": -0.5, "velocity": -0.1, "acceleration": 0.05}]
        pw = compute_peak_window(2022, -0.5, -0.1, 0.05, None, analogues)
        self.assertIn("delta_median", pw)

    def test_compute_peak_window_load_analogues_from_csv(self):
        """Triggers _load_analogue_episodes when analogue_episodes=None."""
        from cerebro_core import compute_peak_window
        pw = compute_peak_window(2022, -0.5, -0.1, 0.05, None, None)
        self.assertIn("peak_year", pw)
        self.assertIn("method", pw)

    def test_load_analogue_episodes_returns_list(self):
        """_load_analogue_episodes returns list of episodes (core frozen, no clock param)."""
        from cerebro_core import _load_analogue_episodes
        ep = _load_analogue_episodes()
        self.assertIsInstance(ep, list)


class TestCouplingCoverage(unittest.TestCase):
    def test_coupling_factor_high_load(self):
        from cerebro_coupling import coupling_factor, COUPLING_FACTOR_HIGH
        clocks = [
            {"saddle_score": 2.5, "velocity": -0.1, "acceleration": -0.05},
            {"saddle_score": 2.5, "velocity": -0.1, "acceleration": -0.05},
        ]
        self.assertEqual(coupling_factor(clocks, threshold=0.3), COUPLING_FACTOR_HIGH)

    def test_adjust_window(self):
        from cerebro_coupling import adjust_window
        ws, we = adjust_window(2025, 2035, 2030, 0.8)
        self.assertLess(we - ws, 10)


class TestEnergyCoverage(unittest.TestCase):
    def test_energy_derivative(self):
        from cerebro_energy import energy_derivative
        dE = energy_derivative(0.5, 0.1, 0.2, 0.3, 0.05, 0.1, dt=1.0)
        self.assertIsInstance(dE, float)

    def test_compute_energy_metrics_with_prev(self):
        from cerebro_energy import compute_energy_metrics
        m = compute_energy_metrics(0.2, 0.5, 0.1, True, 0.1, 0.4, 0.05)
        self.assertIn("energy_score", m)
        self.assertIn("release_risk", m)


class TestIntegrityCoverage(unittest.TestCase):
    def test_integrity_series_short(self):
        from cerebro_integrity import _integrity_series
        s = _integrity_series([1.0, 2.0], expected_years=5)  # n<3
        self.assertLessEqual(s, 1.0)

    def test_integrity_series_empty(self):
        from cerebro_integrity import _integrity_series
        self.assertEqual(_integrity_series([]), 0.0)

    def test_integrity_series_full(self):
        from cerebro_integrity import _integrity_series
        s = _integrity_series([1.0, 2.0, 3.0, 4.0, 5.0], expected_years=5)
        self.assertGreater(s, 0)

    def test_compute_integrity_with_csv(self):
        from cerebro_integrity import compute_integrity
        r = compute_integrity()
        self.assertIn("average_integrity", r)
        self.assertIn("sources", r)


class TestPeakWindowCoverage(unittest.TestCase):
    def test_get_method_equations(self):
        from cerebro_peak_window import get_method_equations
        eq = get_method_equations()
        self.assertIn("saddle_rule", eq)
        self.assertIn("provenance", eq)

    def test_apply_conformal_no_module(self):
        """Conformal path: import fails, falls through."""
        from cerebro_peak_window import compute_peak_window
        pw = compute_peak_window(2022, -0.5, -0.1, 0.05, None, [], apply_conformal=True)
        self.assertIn("peak_year", pw)
        self.assertFalse(pw.get("conformal_applied", True))


class TestWalkforwardCoverage(unittest.TestCase):
    def test_run_walkforward(self):
        from cerebro_walkforward import run_walkforward
        r = run_walkforward()
        self.assertIn("windows_tested", r)


class TestCalibrationCoverage(unittest.TestCase):
    def test_run_calibration(self):
        """Calibration returns bins and method. brier_score present when successful."""
        from cerebro_calibration import run_calibration
        r = run_calibration()
        self.assertIn("bins", r)
        self.assertIn("method", r)
        if not r.get("error"):
            self.assertIn("brier_score", r)

    def test_run_calibration_insufficient_episodes_skips_gracefully(self):
        """If insufficient episodes, calibration returns error dict, does not fail."""
        from cerebro_calibration import run_calibration
        r = run_calibration()
        if r.get("error"):
            self.assertIn("bins", r)
            self.assertIn("method", r)
        else:
            self.assertGreaterEqual(r.get("n_used", 0), 10, "When successful, n_used should be >= 10")


class TestStressCoverage(unittest.TestCase):
    def test_run_stress(self):
        from cerebro_stress import run_stress
        analogues = [{"saddle_year": 1990, "event_year": 1994, "position": -0.5, "velocity": -0.1, "acceleration": 0.05}]
        r = run_stress(2022, -0.5, -0.1, 0.05, analogues)
        self.assertIn("peak_mean", r)
        self.assertIn("peak_std", r)


class TestAblationCoverage(unittest.TestCase):
    def test_run_ablation(self):
        from cerebro_ablation import run_ablation
        r = run_ablation()
        self.assertIn("core_error", r)


class TestLiveMonitorCoverage(unittest.TestCase):
    def test_compute_monitor(self):
        from cerebro_live_monitor import compute_monitor
        r = compute_monitor()
        self.assertIn("predictions_active", r)


if __name__ == "__main__":
    unittest.main()
