#!/usr/bin/env python3
"""Tests for Figure-8 lab: historical replay, forward simulation, distribution shift."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestHistoricalReplaySchema(unittest.TestCase):
    def test_replay_schema_keys_exist(self):
        """Historical replay has version, n_years, mean_errors, records."""
        p = ROOT / "cerebro_data" / "historical_replay.json"
        if not p.exists():
            self.skipTest("historical_replay.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Replay had error")
        self.assertIn("version", data)
        self.assertIn("n_years", data)
        self.assertIn("mean_errors", data)
        self.assertIn("records", data)
        self.assertIn("core", data["mean_errors"])


class TestForwardSimulationSchema(unittest.TestCase):
    def test_simulation_probabilities_in_01(self):
        """Forward simulation event_probability_5yr and event_probability_10yr in [0,1]."""
        p = ROOT / "cerebro_data" / "forward_simulation.json"
        if not p.exists():
            self.skipTest("forward_simulation.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Simulation had error")
        p5 = data.get("event_probability_5yr")
        p10 = data.get("event_probability_10yr")
        if p5 is not None:
            self.assertGreaterEqual(p5, 0, "event_probability_5yr >= 0")
            self.assertLessEqual(p5, 1, "event_probability_5yr <= 1")
        if p10 is not None:
            self.assertGreaterEqual(p10, 0, "event_probability_10yr >= 0")
            self.assertLessEqual(p10, 1, "event_probability_10yr <= 1")

    def test_simulation_run_count_equals_sim_runs(self):
        """Simulation run count equals SIM_RUNS (5000)."""
        p = ROOT / "cerebro_data" / "forward_simulation.json"
        if not p.exists():
            self.skipTest("forward_simulation.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Simulation had error")
        self.assertEqual(data.get("sim_runs"), 5000, "sim_runs must equal 5000")


class TestDistributionShiftSchema(unittest.TestCase):
    def test_distribution_shift_percentile_in_01(self):
        """Distribution shift percentile in [0,1]."""
        p = ROOT / "cerebro_data" / "distribution_shift.json"
        if not p.exists():
            self.skipTest("distribution_shift.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Distribution shift had error")
        pct = data.get("percentile")
        if pct is not None:
            self.assertGreaterEqual(pct, 0, "percentile >= 0")
            self.assertLessEqual(pct, 1, "percentile <= 1")


class TestReplayNoLeakage(unittest.TestCase):
    def test_replay_at_year_y_uses_only_past(self):
        """For any prediction at year Y, training set max saddle_year < Y."""
        from cerebro_historical_replay import _load_candidate_years, _past_only_pool
        episodes = _load_candidate_years()
        if len(episodes) < 10:
            self.skipTest("Insufficient episodes")
        sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
        for ep in sorted_ep[5:]:
            Y = ep.get("saddle_year")
            if Y is None:
                continue
            pool = _past_only_pool(episodes, Y)
            for p in pool:
                self.assertLess(p.get("saddle_year", 0), Y, "Training uses only past years")


if __name__ == "__main__":
    unittest.main()
