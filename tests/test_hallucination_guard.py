"""Tests for chimera_hallucination_guard."""

import pytest
from chimera.chimera_hallucination_guard import apply_guard, load_diagnostics_bundle


class TestNoiseClamp:
    """Rule 1: synthetic_noise_conf_mean > 0.70 and confidence > 75 → clamp to 65."""

    def test_noise_clamp_triggers(self):
        pred = {"confidence_pct": 85, "window_start": 2025, "window_end": 2030}
        diag = {"synthetic_noise_conf_mean": 0.75, "n_eff": 10, "drift_detected": False}
        out = apply_guard(pred, diag)
        assert out["clamped"] is True
        assert out["confidence_after"] == 65
        assert "noise_overconfidence" in out["reasons"]

    def test_noise_clamp_no_trigger_when_safe(self):
        pred = {"confidence_pct": 85, "window_start": 2025, "window_end": 2030}
        diag = {"synthetic_noise_conf_mean": 0.65, "n_eff": 10, "drift_detected": False}
        out = apply_guard(pred, diag)
        assert out["clamped"] is False
        assert out["confidence_after"] == 85


class TestLowNEff:
    """Rule 2: n_eff < 5 and confidence > 70 → clamp to 60."""

    def test_low_n_eff_clamp_triggers(self):
        pred = {"confidence_pct": 80, "window_start": 2025, "window_end": 2030}
        diag = {"n_eff": 3, "synthetic_noise_conf_mean": 0.5, "drift_detected": False}
        out = apply_guard(pred, diag)
        assert out["clamped"] is True
        assert out["confidence_after"] == 60
        assert "low_n_eff" in out["reasons"]

    def test_low_n_eff_no_trigger_when_high_n(self):
        pred = {"confidence_pct": 80, "window_start": 2025, "window_end": 2030}
        diag = {"n_eff": 10, "synthetic_noise_conf_mean": 0.5, "drift_detected": False}
        out = apply_guard(pred, diag)
        assert out["clamped"] is False


class TestDriftClamp:
    """Rule 3: drift_detected → clamp to 55 and widen window by 20%."""

    def test_drift_clamp_triggers_and_widens_window(self):
        pred = {"confidence_pct": 85, "window_start": 2025, "window_end": 2030}
        diag = {"n_eff": 10, "synthetic_noise_conf_mean": 0.5, "drift_detected": True}
        out = apply_guard(pred, diag)
        assert out["clamped"] is True
        assert out["confidence_after"] == 55
        assert "drift_detected" in out["reasons"]
        assert out["window_start"] == 2024  # 5 * 0.2 = 1, pad = 1
        assert out["window_end"] == 2031

    def test_drift_widens_by_20_percent(self):
        pred = {"confidence_pct": 90, "window_start": 2020, "window_end": 2030}
        diag = {"drift_detected": True}
        out = apply_guard(pred, diag)
        width = 10
        pad = max(1, int(round(width * 0.20)))
        assert out["window_start"] == 2020 - pad
        assert out["window_end"] == 2030 + pad


class TestWideIntervalClamp:
    """Rule 4: interval_width > 5 and confidence > 70 → clamp to 60."""

    def test_wide_interval_clamp_triggers(self):
        pred = {"confidence_pct": 80, "window_start": 2020, "window_end": 2027}
        diag = {"n_eff": 10, "synthetic_noise_conf_mean": 0.5, "drift_detected": False}
        out = apply_guard(pred, diag)
        assert out["clamped"] is True
        assert out["confidence_after"] == 60
        assert "wide_interval_high_conf" in out["reasons"]

    def test_wide_interval_no_trigger_when_narrow(self):
        pred = {"confidence_pct": 80, "window_start": 2025, "window_end": 2029}
        diag = {"n_eff": 10, "synthetic_noise_conf_mean": 0.5, "drift_detected": False}
        out = apply_guard(pred, diag)
        assert out["clamped"] is False


class TestNoClampWhenSafe:
    """No clamp when all conditions are safe."""

    def test_no_clamp_when_safe(self):
        pred = {"confidence_pct": 85, "window_start": 2025, "window_end": 2029}
        diag = {
            "n_eff": 10,
            "synthetic_noise_conf_mean": 0.6,
            "drift_detected": False,
        }
        out = apply_guard(pred, diag)
        assert out["clamped"] is False
        assert out["confidence_after"] == 85
        assert out["reasons"] == []


class TestMultipleReasons:
    """Multiple reasons accumulate correctly."""

    def test_multiple_reasons_accumulate(self):
        pred = {"confidence_pct": 90, "window_start": 2020, "window_end": 2028}
        diag = {
            "n_eff": 3,
            "synthetic_noise_conf_mean": 0.8,
            "drift_detected": True,
        }
        out = apply_guard(pred, diag)
        assert out["clamped"] is True
        assert out["confidence_after"] == 55
        assert "noise_overconfidence" in out["reasons"]
        assert "low_n_eff" in out["reasons"]
        assert "drift_detected" in out["reasons"]
        assert "wide_interval_high_conf" in out["reasons"]


class TestNeverIncreasesConfidence:
    """Guard must never increase confidence."""

    def test_never_increases(self):
        pred = {"confidence_pct": 50, "window_start": 2025, "window_end": 2030}
        diag = {"n_eff": 3, "synthetic_noise_conf_mean": 0.9, "drift_detected": True}
        out = apply_guard(pred, diag)
        assert out["confidence_after"] <= out["confidence_before"]


class TestLoadDiagnostics:
    """load_diagnostics_bundle returns dict with expected keys."""

    def test_load_diagnostics_returns_dict(self):
        bundle = load_diagnostics_bundle()
        assert isinstance(bundle, dict)
        assert "n_eff" in bundle or "synthetic_noise_conf_mean" in bundle or "contract_status" in bundle
