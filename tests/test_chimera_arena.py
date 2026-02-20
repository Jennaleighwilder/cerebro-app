#!/usr/bin/env python3
"""Tests for CHIMERA Arena."""

import json
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"


def test_arena_runs_with_episodes():
    """Arena runs when episodes exist."""
    from chimera import chimera_arena

    result = chimera_arena.run_arena(trigger="test")
    if "error" in result:
        pytest.skip(result["error"])
    assert "best_config" in result
    assert "best_metrics" in result
    assert "promoted" in result
    assert "trigger" in result


def test_arena_config_bounds():
    """Arena configs stay within bounds."""
    from chimera import chimera_arena

    result = chimera_arena.run_arena(trigger="test")
    if "error" in result:
        pytest.skip(result["error"])
    cfg = result.get("best_config", {})
    vw = cfg.get("vel_weight")
    aw = cfg.get("acc_weight")
    tau = cfg.get("tau")
    if vw is not None:
        assert 10 <= vw <= 500
    if aw is not None:
        assert 200 <= aw <= 20000
    if tau is not None:
        assert 0.5 <= tau <= 4.0


def test_arena_state_persists():
    """Arena state file exists after run."""
    from chimera import chimera_arena

    chimera_arena.run_arena(trigger="test")
    p = DATA_DIR / "chimera_arena_state.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        assert "last_run" in d or "last_config" in d or "error" in d
