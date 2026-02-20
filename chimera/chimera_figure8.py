#!/usr/bin/env python3
"""
CHIMERA Figure-8 — Backward pass (prove) + Forward pass (operate).
Backward: read artifacts, update honeycomb, compute infinity score, write chimera_export.
Forward: operational state snapshot, drift detection, drift-triggered arena.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
LOG_PATH = DATA_DIR / "chimera_log.jsonl"

# Drift thresholds for arena trigger
DRIFT_COVERAGE_THRESH = 0.70
DRIFT_BRIER_INCREASE_PCT = 0.30
DRIFT_INFINITY_DROP = 15
LOG_TAIL_FOR_DRIFT = 3


def _last_n_log_entries(n: int = LOG_TAIL_FOR_DRIFT) -> list:
    """Read last n entries from chimera_log.jsonl."""
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return entries
    except Exception:
        return []


def _drift_triggered(infinity_score: float | None = None, brier: float | None = None, coverage: float | None = None) -> tuple[bool, str]:
    """
    Check if drift warrants arena trigger.
    Triggers if: coverage_80 < 0.70 OR Brier +30% over rolling 3 OR InfinityScore drops > 15.
    """
    if brier is None or coverage is None:
        cal = {}
        p = DATA_DIR / "calibration_curve.json"
        if p.exists():
            try:
                with open(p) as f:
                    cal = json.load(f)
            except Exception:
                pass
        cal_mode = cal.get("mode_operational") or cal
        coverage = float(cal.get("coverage_80", cal_mode.get("coverage_80", 0.8)) or 0.8) if coverage is None else coverage
        brier = float(cal_mode.get("brier", cal.get("brier_score", 0.1)) or 0.1) if brier is None else brier
    if infinity_score is None:
        p = DATA_DIR / "infinity_score.json"
        if p.exists():
            try:
                with open(p) as f:
                    infinity_score = float(json.load(f).get("infinity_score", 100) or 100)
            except Exception:
                infinity_score = 100.0
        else:
            infinity_score = 100.0

    entries = _last_n_log_entries(LOG_TAIL_FOR_DRIFT)
    if coverage < DRIFT_COVERAGE_THRESH:
        return True, f"coverage_80={coverage:.2f} < {DRIFT_COVERAGE_THRESH}"

    if entries:
        briers = [e.get("backward", {}).get("brier") for e in entries if e.get("backward", {}).get("brier") is not None]
        if len(briers) >= 2:
            mean_prev = sum(briers[:-1]) / (len(briers) - 1)
            if mean_prev > 0 and brier > mean_prev * (1 + DRIFT_BRIER_INCREASE_PCT):
                return True, f"brier +{DRIFT_BRIER_INCREASE_PCT*100:.0f}% over rolling"

        scores = [e.get("backward", {}).get("infinity_score") for e in entries if e.get("backward", {}).get("infinity_score") is not None]
        if scores:
            best_prev = max(s for s in scores if s is not None)
            if infinity_score < best_prev - DRIFT_INFINITY_DROP:
                return True, f"infinity_score dropped {best_prev - infinity_score:.0f} > {DRIFT_INFINITY_DROP}"

    return False, ""


def run_backward_pass() -> dict:
    """Read existing cerebro_data/*.json, update honeycomb, compute infinity score, write chimera_export."""
    from chimera import chimera_honeycomb
    from chimera import chimera_infinity_score
    from chimera import chimera_export

    honeycomb = chimera_honeycomb.build_honeycomb()
    infinity = chimera_infinity_score.compute_infinity_score()
    export_data = chimera_export.collect_export()
    return {"honeycomb": honeycomb, "infinity_score": infinity, "export": export_data}


def run_forward_pass() -> dict:
    """Operational state snapshot. Does NOT download anything."""
    current_year = None
    integrity_cap = None
    drift_flags = None

    # Latest harm clock row
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, index_col=0)
            if not df.empty and hasattr(df.index, "max"):
                current_year = int(df.index.max()) if df.index.dtype in ("int64", "float64") else None
        except Exception:
            pass

    # Integrity cap
    p = DATA_DIR / "integrity_scores.json"
    if p.exists():
        try:
            with open(p) as f:
                d = json.load(f)
            integrity_cap = d.get("confidence_cap")
        except Exception:
            pass

    # Drift flags from live_monitor
    p = DATA_DIR / "live_monitor.json"
    if p.exists():
        try:
            with open(p) as f:
                d = json.load(f)
            drift_flags = d.get("drift_flags") or d.get("flags") or {}
        except Exception:
            drift_flags = {}

    return {
        "current_year": current_year,
        "integrity_cap": integrity_cap,
        "drift_flags": drift_flags or {},
    }


def run_figure8() -> dict:
    """Backward then forward, append entry to chimera_log.jsonl."""
    from datetime import datetime, timezone

    backward = run_backward_pass()
    forward = run_forward_pass()

    inf = backward.get("infinity_score") or {}
    cal = {}
    p = DATA_DIR / "calibration_curve.json"
    if p.exists():
        try:
            with open(p) as f:
                cal = json.load(f)
        except Exception:
            pass
    cal_mode = cal.get("mode_operational") or cal
    brier = float(cal_mode.get("brier", cal.get("brier_score", 0.1)) or 0.1)
    coverage = float(cal.get("coverage_80", cal_mode.get("coverage_80", 0.8)) or 0.8)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backward": {
            "infinity_score": inf.get("infinity_score"),
            "brier": round(brier, 4),
            "coverage_80": round(coverage, 4),
        },
        "forward": forward,
        "status": "ok",
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Drift-triggered arena
    arena_ran = False
    arena_result = None
    drift_trigger, drift_reason = _drift_triggered(
        infinity_score=inf.get("infinity_score"),
        brier=brier,
        coverage=coverage,
    )
    if drift_trigger:
        try:
            from chimera import chimera_arena
            arena_result = chimera_arena.run_arena(
                trigger=f"drift:{drift_reason}",
                infinity_score=inf.get("infinity_score"),
            )
            arena_ran = True
        except Exception as e:
            arena_result = {"error": str(e)}

    # Record run score for temporal guard
    inf_score = inf.get("infinity_score") or 0
    try:
        from chimera import chimera_arena
        chimera_arena._record_run_score(inf_score, promoted_this_run=arena_ran and (arena_result or {}).get("promoted", False))
    except Exception:
        pass

    # Temporal consistency guard (rollback if needed)
    rollback_done = False
    try:
        from chimera import chimera_arena
        rollback_done = chimera_arena.run_rollback_if_needed(inf_score)
    except Exception:
        pass

    return {
        "backward": backward,
        "forward": forward,
        "log_entry": entry,
        "arena_triggered": drift_trigger,
        "arena_ran": arena_ran,
        "arena_result": arena_result,
        "rollback_done": rollback_done,
    }
