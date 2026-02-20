#!/usr/bin/env python3
"""Placebo test schema and p-value validation."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEBO_PATH = ROOT / "cerebro_data" / "placebo_test.json"
CANDIDATE_SWEEP_PATH = ROOT / "cerebro_data" / "candidate_sweep.json"


class TestPlaceboSchema(unittest.TestCase):
    def test_placebo_schema_when_exists(self):
        """Placebo output has required keys and valid p-values when file exists."""
        if not PLACEBO_PATH.exists():
            self.skipTest("placebo_test.json not yet generated (run cerebro_placebo.py)")
        with open(PLACEBO_PATH) as f:
            data = json.load(f)
        self.assertIn("real_brier", data)
        self.assertIn("real_coverage_80", data)
        self.assertIn("n_placebo", data)
        self.assertIn("p_value_brier", data)
        self.assertIn("p_value_coverage", data)
        if data.get("p_value_brier") is not None:
            self.assertGreaterEqual(data["p_value_brier"], 0)
            self.assertLessEqual(data["p_value_brier"], 1)
        if data.get("p_value_coverage") is not None:
            self.assertGreaterEqual(data["p_value_coverage"], 0)
            self.assertLessEqual(data["p_value_coverage"], 1)

    def test_candidate_sweep_schema_when_exists(self):
        """Candidate sweep has thresholds and sweep array when file exists."""
        if not CANDIDATE_SWEEP_PATH.exists():
            self.skipTest("candidate_sweep.json not yet generated (run cerebro_placebo.py)")
        with open(CANDIDATE_SWEEP_PATH) as f:
            data = json.load(f)
        self.assertIn("thresholds", data)
        self.assertIn("sweep", data)
        self.assertIsInstance(data["sweep"], list)


class TestPlaceboOutput(unittest.TestCase):
    def test_placebo_script_runs(self):
        """cerebro_placebo.main() runs and produces valid output."""
        from cerebro_placebo import _run_placebo, _run_candidate_sweep

        result, _, _ = _run_placebo()
        if result:
            self.assertIn("p_value_brier", result)
            self.assertIn("p_value_coverage", result)
            if result.get("p_value_brier") is not None:
                self.assertGreaterEqual(result["p_value_brier"], 0)
                self.assertLessEqual(result["p_value_brier"], 1)
            if result.get("p_value_coverage") is not None:
                self.assertGreaterEqual(result["p_value_coverage"], 0)
                self.assertLessEqual(result["p_value_coverage"], 1)

        sweep = _run_candidate_sweep()
        self.assertIsInstance(sweep, list)
        for row in sweep:
            self.assertIn("threshold", row)
            self.assertIn("n_used", row)
            self.assertIn("brier", row)
            self.assertIn("coverage_80", row)


if __name__ == "__main__":
    unittest.main()
