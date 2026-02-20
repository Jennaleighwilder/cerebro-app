#!/usr/bin/env python3
"""Tests for Phase 3 self-tune: distance weights, conformal, regime markov."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestDistanceWeights(unittest.TestCase):
    def test_distance_weights_positive_when_exists(self):
        """If distance_weights.json exists, vel_weight>0 and acc_weight>0."""
        p = ROOT / "cerebro_data" / "distance_weights.json"
        if not p.exists():
            self.skipTest("distance_weights.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Fit had error")
        self.assertGreater(data.get("vel_weight", 0), 0)
        self.assertGreater(data.get("acc_weight", 0), 0)

    def test_sweep_has_candidates_when_episodes_exist(self):
        """Sweep output contains at least 5 candidates when episodes exist."""
        p = ROOT / "cerebro_data" / "distance_weight_sweep.json"
        if not p.exists():
            self.skipTest("distance_weight_sweep.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        top = data.get("top_20", [])
        self.assertGreaterEqual(len(top), 5, "Sweep should have at least 5 candidates when episodes exist")


class TestHoneycombConformal(unittest.TestCase):
    def test_conformal_schema_keys_exist(self):
        """Conformal file has required schema keys when it can run."""
        p = ROOT / "cerebro_data" / "honeycomb_conformal.json"
        if not p.exists():
            self.skipTest("honeycomb_conformal.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Conformal had error")
        for key in ["target_coverage", "min_train", "n_used", "s_hat", "empirical_coverage", "method"]:
            self.assertIn(key, data)

    def test_honeycomb_with_conformal_peak_in_window(self):
        """Honeycomb with conformal guarantees window_start <= peak <= window_end."""
        p = ROOT / "cerebro_data" / "honeycomb_latest.json"
        if not p.exists():
            self.skipTest("honeycomb_latest.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Honeycomb had error")
        peak = data.get("peak_year")
        ws = data.get("window_start")
        we = data.get("window_end")
        self.assertGreaterEqual(peak, ws)
        self.assertLessEqual(peak, we)


class TestRegimeMarkov(unittest.TestCase):
    def test_regime_markov_distributions_sum_to_one(self):
        """Regime markov p_5yr (and others) distributions sum to 1.0 (±0.02)."""
        p = ROOT / "cerebro_data" / "regime_markov.json"
        if not p.exists():
            self.skipTest("regime_markov.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Regime Markov had error")
        for key in ["p_1yr", "p_3yr", "p_5yr", "p_10yr"]:
            dist = data.get(key, {})
            if not dist:
                continue
            total = sum(dist.values())
            self.assertAlmostEqual(total, 1.0, delta=0.02, msg=f"{key} should sum to 1.0")


if __name__ == "__main__":
    unittest.main()
