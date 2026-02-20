#!/usr/bin/env python3
"""
CEREBRO EVENT LOADER — Structural regime pivots from event_library.json
No hardcoded event years. Future-proof for GSS, ISSP, different waves.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVENT_LIBRARY_PATH = SCRIPT_DIR / "cerebro_data" / "event_library.json"

# Fallback if JSON missing (legacy compatibility)
FALLBACK_EVENT_YEARS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]


def load_event_years(country: str = "US") -> list[int]:
    """Load event years from event_library.json. Returns sorted list of years."""
    if not EVENT_LIBRARY_PATH.exists():
        return sorted(FALLBACK_EVENT_YEARS)
    try:
        with open(EVENT_LIBRARY_PATH) as f:
            data = json.load(f)
        events = data.get("events", [])
        years = [e["year"] for e in events if isinstance(e.get("year"), (int, float))]
        return sorted(set(int(y) for y in years)) if years else sorted(FALLBACK_EVENT_YEARS)
    except Exception:
        return sorted(FALLBACK_EVENT_YEARS)
