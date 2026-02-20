#!/usr/bin/env python3
"""Tests for auto-enable coupling and honeycomb based on data thresholds."""

import pytest


def test_coupling_disabled_below_60():
    """Assert coupling is off when n_quality_episodes < 60."""
    from cerebro_calibration import _should_use_coupling

    assert _should_use_coupling(34) is False
    assert _should_use_coupling(59) is False
    assert _should_use_coupling(60) is True
    assert _should_use_coupling(61) is True


def test_honeycomb_disabled_below_20_post1990():
    """Assert honeycomb is off when n_post1990_sister < 20; verify calibration output structure."""
    from cerebro_calibration import run_calibration, _should_use_honeycomb

    assert _should_use_honeycomb(29) is False
    assert _should_use_honeycomb(30) is True

    result = run_calibration(score_threshold=1.5)
    n_post1990 = result.get("n_post1990_sister", 0)
    assert result.get("honeycomb_enabled") == _should_use_honeycomb(n_post1990)
    assert result.get("coupling_enabled") == (result.get("n_quality_episodes", 0) >= 60)
    assert "coupling_threshold" in result
    assert "honeycomb_threshold" in result
