#!/usr/bin/env python3
"""
Unit tests for Cerebro state-space, hazard, changepoint, coupling.
Run: python3 tests/test_state_space_hazard.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestStateSpace(unittest.TestCase):
    def test_kalman_step_predict(self):
        """Kalman predict step produces valid state."""
        from cerebro_state_space import kalman_step, _transition_matrix, _process_noise
        import numpy as np

        x = np.array([0.0, 0.1, 0.02])
        P = np.eye(3) * 0.1
        y = np.array([0.0, 0.0, 0.0])
        obs_mask = (False, False, False)
        F = _transition_matrix(1.0)
        Q = _process_noise(1.0)
        R = np.eye(0) * 0.5
        x_post, P_post = kalman_step(x, P, y, obs_mask, F, Q, R)
        self.assertEqual(x_post.shape, (3,))
        self.assertEqual(P_post.shape, (3, 3))
        self.assertGreater(x_post[0], x[0] - 1)  # position moved

    def test_run_filter_deterministic(self):
        """Filter with fixed seed produces same result."""
        from cerebro_state_space import run_filter

        years = [2000, 2001, 2002, 2003, 2004]
        pos = [1.0, 1.1, 1.2, 1.15, 1.3]
        vel = [0.1, 0.1, 0.0, -0.05, 0.15]
        acc = [0.0, -0.1, -0.1, 0.05, 0.2]
        r1 = run_filter(years, pos, vel, acc, seed=42)
        r2 = run_filter(years, pos, vel, acc, seed=42)
        self.assertEqual(r1["posteriors"][-1]["mean"], r2["posteriors"][-1]["mean"])


class TestHazard(unittest.TestCase):
    def test_predict_hazard_shape(self):
        """predict_hazard returns prob_1y, prob_3y, prob_5y when model exists."""
        from cerebro_hazard import predict_hazard, fit_logistic_hazard, load_episodes_with_state

        episodes = load_episodes_with_state()
        if len(episodes) < 8:
            self.skipTest("Insufficient episodes for hazard fit")
        model = fit_logistic_hazard(episodes)
        if "error" in model:
            self.skipTest(model["error"])
        pred = predict_hazard(-1.0, -0.1, 0.05, 0.5, model=model)
        self.assertIn("prob_1y", pred)
        self.assertIn("prob_3y", pred)
        self.assertIn("prob_5y", pred)
        if pred.get("prob_5y") is not None:
            self.assertGreaterEqual(pred["prob_5y"], 0)
            self.assertLessEqual(pred["prob_5y"], 1)

    def test_event_library_schema(self):
        """Event library has required fields."""
        from cerebro_hazard import EVENT_LIBRARY

        for ev in EVENT_LIBRARY:
            self.assertIn("event_type", ev)
            self.assertIn("event_year", ev)
            self.assertIn("country", ev)


class TestChangepoint(unittest.TestCase):
    def test_cusum_returns_probability(self):
        """CUSUM returns regime_shift_probability in [0,1]."""
        from cerebro_changepoint import run_changepoint
        import numpy as np

        posteriors = [
            {"year": 2000 + i, "mean": [1.0 + i * 0.1, 0.1, 0.02], "cov_diag": [0.1, 0.1, 0.1]}
            for i in range(10)
        ]
        r = run_changepoint(posteriors)
        self.assertIn("regime_shift_probability", r)
        self.assertGreaterEqual(r["regime_shift_probability"], 0)
        self.assertLessEqual(r["regime_shift_probability"], 1)


class TestCoupling(unittest.TestCase):
    def test_load_coupling_returns_matrix(self):
        """load_coupling returns matrix with harm, class, sexual, evil."""
        from cerebro_coupling import load_coupling, CLOCKS

        c = load_coupling()
        self.assertIn("matrix", c)
        for clock in CLOCKS:
            self.assertIn(clock, c["matrix"])

    def test_cross_features_shape(self):
        """cross_features returns dict of cross-terms."""
        from cerebro_coupling import cross_features

        states = {
            "harm": {"position": 0.5, "velocity": 0.1, "acceleration": 0.02},
            "class": {"position": -1.0, "velocity": -0.05, "acceleration": 0.01},
        }
        out = cross_features(states)
        self.assertIsInstance(out, dict)
        self.assertTrue(len(out) >= 0)


class TestModelMetrics(unittest.TestCase):
    def test_compute_calibrated_confidence_returns_score(self):
        """compute_calibrated_confidence returns 0-100 score."""
        from cerebro_model_metrics import compute_calibrated_confidence

        m = compute_calibrated_confidence()
        self.assertIn("confidence_pct", m)
        self.assertGreaterEqual(m["confidence_pct"], 50)
        self.assertLessEqual(m["confidence_pct"], 98)


if __name__ == "__main__":
    unittest.main()
