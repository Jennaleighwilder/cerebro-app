#!/usr/bin/env python3
"""
CHIMERA Figure-8 — Backward pass (prove) + Forward pass (operate).
Backward: read artifacts, update honeycomb, compute infinity score, write chimera_export.
Forward: operational state snapshot (current year, integrity cap, drift flags).
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
LOG_PATH = DATA_DIR / "chimera_log.jsonl"


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

    entry = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backward": {"infinity_score": (backward.get("infinity_score") or {}).get("infinity_score")},
        "forward": forward,
        "status": "ok",
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"backward": backward, "forward": forward, "log_entry": entry}
