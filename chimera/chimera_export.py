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

    contract = {}
    p_contract = DATA_DIR / "contract_report.json"
    if p_contract.exists():
        try:
            with open(p_contract) as f:
                contract = json.load(f)
        except Exception:
            pass

    arena_state = {}
    p_arena = DATA_DIR / "chimera_arena_state.json"
    if p_arena.exists():
        try:
            with open(p_arena) as f:
                arena_state = json.load(f)
        except Exception:
            pass

    out = {
        "contract": contract,
        "infinity_score": infinity.get("infinity_score"),
        "G": infinity.get("G"),
        "penalty": infinity.get("penalty") or (infinity.get("multipliers", {}).get("penalty_product")),
        "signals": infinity.get("signals", infinity.get("subscores", {})),
        "honeycomb": honeycomb,
        "arena": {
            "last_config": arena_state.get("last_config"),
            "runs_since_promotion": arena_state.get("runs_since_promotion"),
            "promoted_at": arena_state.get("promoted_at"),
        },
        "log_tail": log_lines[-LOG_TAIL_N:],
        "log_count": len(log_lines),
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(EXPORT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out
