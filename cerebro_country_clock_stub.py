#!/usr/bin/env python3
"""
CEREBRO COUNTRY CLOCK STUB — Minimal proxy clocks per country
=============================================================
Builds stub clocks from GLOPOP, country risk, or other local series.
Output: cerebro_data/country_clocks/{ISO}_harm_clock.csv
Columns: position, velocity, acceleration (year-indexed).
If no local data, returns error. Scaffolding for adding real country clocks later.
"""

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data" / "country_clocks"
HARM_CLOCK_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
GLOPOP_PATH = SCRIPT_DIR / "cerebro_data" / "GLOPOP_country_summary.csv"
COUNTRY_RISK_PATH = SCRIPT_DIR / "cerebro_data" / "country_risk_data.json"

# ISO to country name for output
ISO_TO_NAME = {"US": "US", "USA": "US", "UK": "UK", "GBR": "UK", "DE": "DE", "DEU": "DE", "FR": "FR", "FRA": "FR", "JP": "JP", "JPN": "JP", "CA": "CA", "CAN": "CA"}


def _gini_to_position(gini: float) -> float:
    """Map Gini (0-100) to clock position -10..+10. Higher Gini = more leftward (negative)."""
    if gini is None or gini < 20:
        return 0.0
    # Gini 20->0, 40->-4, 60->-8
    return max(-10, min(10, (gini - 35) / 3.0))


def _glopop_to_position(top10_share: float) -> float:
    """Map top10_share to position. Higher share = more inequality = more negative."""
    if top10_share is None:
        return 0.0
    return max(-10, min(10, (top10_share - 25) / 2.0))


def _build_stub_from_harm_clock() -> list[dict] | None:
    """Use main US harm clock as template for US stub."""
    if not HARM_CLOCK_PATH.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(HARM_CLOCK_PATH, index_col=0)
        df = df[df["clock_position_10pt"].notna()].tail(50)
        if len(df) < 10:
            return None
        rows = []
        for yr, row in df.iterrows():
            pos = row.get("clock_position_10pt")
            vel = row.get("velocity")
            acc = row.get("acceleration")
            if pd.isna(pos):
                continue
            rows.append({"year": int(yr), "position": float(pos), "velocity": float(vel) if not pd.isna(vel) else 0.0, "acceleration": float(acc) if not pd.isna(acc) else 0.0})
        return rows
    except Exception:
        return None


def _build_stub_from_glopop(iso: str) -> list[dict] | None:
    """Build minimal stub from GLOPOP top10_share for one country."""
    if not GLOPOP_PATH.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(GLOPOP_PATH)
        if "iso" not in df.columns and len(df.columns) > 0:
            return None
        row = df[df["iso"] == iso].iloc[0] if iso in df["iso"].values else None
        if row is None:
            return None
        top10 = float(row.get("top10_share", 25))
        pos = _glopop_to_position(top10)
        # Synthetic 30-year series with slight trend
        rows = []
        for i in range(30):
            yr = 1995 + i
            drift = 0.02 * (i - 15)
            rows.append({"year": yr, "position": pos + drift + 0.05 * (i % 5), "velocity": 0.02, "acceleration": 0.001})
        return rows
    except Exception:
        return None


def _build_stub_from_country_risk(iso: str) -> list[dict] | None:
    """Build stub from country_risk_data gini if available."""
    if not COUNTRY_RISK_PATH.exists():
        return None
    try:
        import json
        with open(COUNTRY_RISK_PATH) as f:
            data = json.load(f)
        for entry in data.get("top_10_unequal", []) + data.get("top_10_risk", []):
            if entry.get("iso") == iso and entry.get("gini") is not None:
                pos = _gini_to_position(entry["gini"])
                rows = []
                for i in range(30):
                    yr = 1995 + i
                    rows.append({"year": yr, "position": pos + 0.01 * i, "velocity": 0.01, "acceleration": 0.0})
                return rows
        return None
    except Exception:
        return None


def build_country_clock(iso: str) -> dict:
    """Build clock for country. Returns {"path": str} or {"error": str}."""
    iso_upper = iso.upper()
    if iso_upper in ("US", "USA"):
        rows = _build_stub_from_harm_clock()
    else:
        rows = _build_stub_from_glopop(iso_upper) or _build_stub_from_country_risk(iso_upper)
    if not rows or len(rows) < 10:
        return {"error": "no local data", "iso": iso_upper}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{iso_upper}_harm_clock.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "position", "velocity", "acceleration"])
        w.writeheader()
        w.writerows(rows)
    return {"path": str(out_path), "iso": iso_upper, "n_years": len(rows)}


def run_all_stubs() -> dict:
    """Build stubs for US, UK, DE, FR, JP, CA."""
    results = {}
    # Map display name -> ISO for lookup (GLOPOP uses USA, GBR, DEU, FRA, JPN, CAN)
    for name, iso in [("US", "USA"), ("UK", "GBR"), ("DE", "DEU"), ("FR", "FRA"), ("JP", "JPN"), ("CA", "CAN")]:
        r = build_country_clock(iso)
        results[name] = r
    return results


def main():
    r = run_all_stubs()
    ok = sum(1 for v in r.values() if "error" not in v)
    print(f"Country clock stubs: {ok}/6 built → {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
