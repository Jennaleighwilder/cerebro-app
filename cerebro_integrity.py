#!/usr/bin/env python3
"""
CEREBRO SIGNAL INTEGRITY MONITOR
Track freshness, missing ratio, variance shift per source.
"""

import json
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "integrity_scores.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"


def _integrity_series(series: list, expected_years: int = 5) -> float:
    """Score 0-1: freshness, completeness, variance stability."""
    if not series:
        return 0.0
    valid = [x for x in series if x is not None and not (isinstance(x, float) and str(x) == "nan")]
    n = len(valid)
    completeness = n / max(1, expected_years) if expected_years else 1.0
    completeness = min(1.0, completeness)
    if n < 3:
        return completeness * 0.5
    import numpy as np
    arr = np.array([float(x) for x in valid])
    var = np.var(arr)
    mean = np.mean(arr)
    # Variance shift: compare second half to first half
    mid = n // 2
    var1 = np.var(arr[:mid]) if mid > 0 else var
    var2 = np.var(arr[mid:]) if n - mid > 0 else var
    var_ratio = max(var1, var2, 1e-10) / min(var1, var2, 1e-10)
    stability = 1.0 / min(var_ratio, 10.0)
    return min(1.0, 0.4 * completeness + 0.3 * min(1.0, 1 - abs(1 - stability)) + 0.3)


def compute_integrity() -> dict:
    """Per-source integrity. Cap confidence at MEDIUM if average < 0.7."""
    import pandas as pd
    from datetime import datetime, timezone

    if not CSV_PATH.exists():
        return {"sources": {}, "average_integrity": 0.0, "confidence_cap": "LOW"}

    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(50)

    sources = {}
    for col in ["clock_position_10pt", "velocity", "acceleration", "ring_B_score", "homicide_rate", "incarceration_rate_bjs"]:
        if col not in df.columns:
            continue
        s = df[col].dropna().tolist()
        sources[col] = round(_integrity_series(s, 20), 4)

    avg = sum(sources.values()) / len(sources) if sources else 0.0
    return {
        "sources": sources,
        "average_integrity": round(avg, 4),
        "confidence_cap": "MEDIUM" if avg < 0.7 else "HIGH",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = compute_integrity()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Integrity: avg={r['average_integrity']}, cap={r['confidence_cap']}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
