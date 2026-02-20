#!/usr/bin/env python3
"""
CEREBRO MULTI-HORIZON HAZARD CURVE
P_1yr, P_3yr, P_5yr, P_10yr — decision makers think in probability curves.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "hazard_curve.json"


def compute_hazard_curve(
    peak_year: int,
    window_start: int,
    window_end: int,
    now_year: int,
) -> dict:
    """
    Cumulative hazard: P(event by year T).
    Model: event ~ uniform over [window_start, window_end], truncated at peak_year.
    """
    span = max(1, window_end - window_start)
    # Simple model: linear ramp from window_start to window_end
    def P_by_year(T: int) -> float:
        if T < window_start:
            return 0.0
        if T >= window_end:
            return 1.0
        return min(1.0, (T - window_start) / span)

    return {
        "P_1yr": round(P_by_year(now_year + 1), 2),
        "P_3yr": round(P_by_year(now_year + 3), 2),
        "P_5yr": round(P_by_year(now_year + 5), 2),
        "P_10yr": round(P_by_year(now_year + 10), 2),
        "peak_year": peak_year,
        "window_start": window_start,
        "window_end": window_end,
        "now_year": now_year,
    }


def run_hazard_curve() -> dict:
    from cerebro_core import compute_peak_window, _load_analogue_episodes

    episodes = _load_analogue_episodes()
    if not episodes:
        return {"error": "No episodes", "P_1yr": 0, "P_3yr": 0, "P_5yr": 0, "P_10yr": 0}

    row = episodes[-1]
    pred = compute_peak_window(
        row["saddle_year"], row["position"], row["velocity"], row["acceleration"],
        row.get("ring_B_score"),
        [e for e in episodes if e["saddle_year"] != row["saddle_year"]],
        interval_alpha=0.8,
    )
    now = row["saddle_year"]
    return compute_hazard_curve(
        pred["peak_year"], pred["window_start"], pred["window_end"], now,
    )


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_hazard_curve()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Hazard curve: P_1yr={r.get('P_1yr')}, P_3yr={r.get('P_3yr')}, P_5yr={r.get('P_5yr')}, P_10yr={r.get('P_10yr')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
