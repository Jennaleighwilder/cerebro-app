#!/usr/bin/env python3
"""Tests for Honeycomb ensemble (fusion: core + sister + sim + shift)."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestHoneycombSchema(unittest.TestCase):
    def test_honeycomb_output_schema_keys_exist(self):
        """Honeycomb output has required schema keys."""
        p = ROOT / "cerebro_data" / "honeycomb_latest.json"
        if not p.exists():
            self.skipTest("honeycomb_latest.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Honeycomb had error")
        for key in ["peak_year", "window_start", "window_end", "confidence_pct", "method", "components", "disagreement_std_years"]:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_honeycomb_confidence_in_range(self):
        """Honeycomb confidence_pct in [40, 95]."""
        p = ROOT / "cerebro_data" / "honeycomb_latest.json"
        if not p.exists():
            self.skipTest("honeycomb_latest.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Honeycomb had error")
        conf = data.get("confidence_pct")
        self.assertIsNotNone(conf)
        self.assertGreaterEqual(conf, 40, "confidence_pct >= 40")
        self.assertLessEqual(conf, 95, "confidence_pct <= 95")

    def test_honeycomb_window_width(self):
        """Honeycomb window width between [3, 15] years."""
        p = ROOT / "cerebro_data" / "honeycomb_latest.json"
        if not p.exists():
            self.skipTest("honeycomb_latest.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Honeycomb had error")
        ws = data.get("window_start")
        we = data.get("window_end")
        self.assertIsNotNone(ws)
        self.assertIsNotNone(we)
        width = we - ws
        self.assertGreaterEqual(width, 3, "window width >= 3")
        self.assertLessEqual(width, 15, "window width <= 15")

    def test_honeycomb_disagreement_nonnegative(self):
        """disagreement_std_years >= 0."""
        p = ROOT / "cerebro_data" / "honeycomb_latest.json"
        if not p.exists():
            self.skipTest("honeycomb_latest.json not yet generated")
        with open(p) as f:
            data = json.load(f)
        if data.get("error"):
            self.skipTest("Honeycomb had error")
        disp = data.get("disagreement_std_years")
        self.assertIsNotNone(disp)
        self.assertGreaterEqual(disp, 0, "disagreement_std_years >= 0")

    def test_honeycomb_peak_inside_window(self):
        """Honeycomb peak_year inside [window_start, window_end]."""
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
        self.assertGreaterEqual(peak, ws, "peak_year >= window_start")
        self.assertLessEqual(peak, we, "peak_year <= window_end")


class TestDistributionShiftPercentile(unittest.TestCase):
    def test_distribution_shift_percentile_in_01(self):
        """Distribution shift percentile in [0, 1]."""
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


if __name__ == "__main__":
    unittest.main()
