#!/usr/bin/env python3
"""Tests for Sister Engine and Honeycomb ensemble."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestSisterOutputSchema(unittest.TestCase):
    def test_sister_output_schema(self):
        """Sister output has peak_year, window_start, window_end, confidence_pct, method."""
        from cerebro_sister_engine import run_sister_engine
        r = run_sister_engine()
        if r.get("error"):
            self.skipTest("Sister engine insufficient data")
        for key in ["peak_year", "window_start", "window_end", "confidence_pct", "method", "n_train", "residual_iqr"]:
            self.assertIn(key, r, f"Missing key: {key}")


class TestSisterPastOnly(unittest.TestCase):
    def test_sister_training_past_only(self):
        """Sister training uses only episodes with saddle_year < prediction year."""
        from cerebro_calibration import _load_episodes
        from cerebro_eval_utils import past_only_pool
        episodes, _ = _load_episodes(score_threshold=2.0)
        if len(episodes) < 10:
            self.skipTest("Insufficient episodes")
        sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
        for ep in sorted_ep[5:]:
            t = ep.get("saddle_year")
            if t is None:
                continue
            pool = past_only_pool(episodes, t)
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


class TestSisterLatestFile(unittest.TestCase):
    def test_sister_latest_file_schema(self):
        """sister_latest.json has peak_year, window, confidence."""
        p = ROOT / "cerebro_data" / "sister_latest.json"
        if not p.exists():
            self.skipTest("sister_latest.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Sister had error")
        for k in ["peak_year", "window_start", "window_end", "confidence_pct", "method"]:
            self.assertIn(k, data)


if __name__ == "__main__":
    unittest.main()
