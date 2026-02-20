#!/usr/bin/env python3
"""
CHIMERA Export — Collect infinity_score, honeycomb, last N log lines.
Write compact UI-safe cerebro_data/chimera_export.json.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
EXPORT_PATH = DATA_DIR / "chimera_export.json"
LOG_PATH = DATA_DIR / "chimera_log.jsonl"
LOG_TAIL_N = 50


def collect_export() -> dict:
    """Collect latest infinity_score, honeycomb, last N log lines."""
    infinity = {}
    p = DATA_DIR / "infinity_score.json"
    if p.exists():
        try:
            with open(p) as f:
                infinity = json.load(f)
        except Exception:
            pass

    honeycomb = {}
    from chimera import chimera_honeycomb
    honeycomb = chimera_honeycomb.build_honeycomb()

    log_lines = []
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH) as f:
                lines = f.readlines()
            for line in lines[-LOG_TAIL_N:]:
                line = line.strip()
                if line:
                    try:
                        log_lines.append(json.loads(line))
                    except Exception:
                        log_lines.append({"raw": line[:200]})
        except Exception:
            pass

    out = {
        "infinity_score": infinity.get("infinity_score"),
        "G": infinity.get("G"),
        "penalty": infinity.get("penalty"),
        "signals": infinity.get("signals", {}),
        "honeycomb": honeycomb,
        "log_tail": log_lines[-LOG_TAIL_N:],
        "log_count": len(log_lines),
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out
