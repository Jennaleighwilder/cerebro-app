#!/usr/bin/env python3
"""
CEREBRO ACLED PROTEST LOADER — Leading indicator for episode generation
======================================================================
ACLED protest event frequency precedes saddle points by 2–4 years.
Requires manual download from acleddata.com (registration).

Usage:
  1. Register at acleddata.com
  2. Data Export Tool → filter: Country=United States, Event Type=Protests
  3. Download CSV to cerebro_data/ACLED_export.csv (or set ACLED_EXPORT_PATH)
  4. Run: python cerebro_acled_loader.py

Output: cerebro_data/ACLED_protest_annual.csv (year, protest_count)
        Merged into cerebro_gathered_raw.csv by cerebro_data_gather.py
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
EXPORT_PATH = DATA_DIR / "ACLED_export.csv"
OUTPUT_PATH = DATA_DIR / "ACLED_protest_annual.csv"


def load_acled_protest_annual() -> pd.DataFrame | None:
    """
    Aggregate ACLED export to annual protest count.
    Expects CSV with columns: event_date (or event_date_parsed), event_type.
    Event types: Protests, Riots (optional — configurable).
    """
    path = Path(__file__).resolve().parent
    export = path / "cerebro_data" / "ACLED_export.csv"
    if not export.exists():
        return None
    try:
        df = pd.read_csv(export, low_memory=False)
        # ACLED columns vary; try common names
        date_col = None
        for c in ["event_date", "event_date_parsed", "EVENT_DATE", "event_date_parsed"]:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            return None
        type_col = None
        for c in ["event_type", "EVENT_TYPE", "sub_event_type", "SUB_EVENT_TYPE"]:
            if c in df.columns:
                type_col = c
                break
        df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
        df = df.dropna(subset=["year"])
        if type_col:
            # Filter to protests (and optionally riots)
            protest_mask = df[type_col].str.lower().str.contains("protest|riot", na=False, regex=True)
            df = df[protest_mask]
        annual = df.groupby("year").size().reset_index(name="protest_count")
        annual["year"] = annual["year"].astype(int)
        annual.to_csv(OUTPUT_PATH, index=False)
        return annual
    except Exception as e:
        print(f"  ✗ ACLED load failed: {e}")
        return None


def main():
    print("CEREBRO ACLED PROTEST LOADER")
    print("=" * 50)
    if not EXPORT_PATH.exists():
        print(f"  ⚠ {EXPORT_PATH} not found.")
        print("  Download from acleddata.com/data-export-tool")
        print("  Filter: Country=United States, Event Type=Protests")
        print("  Save as cerebro_data/ACLED_export.csv")
        return 1
    result = load_acled_protest_annual()
    if result is not None:
        print(f"  ✓ {OUTPUT_PATH} ({len(result)} years)")
        print(f"  Run cerebro_data_gather.py to merge into cerebro_gathered_raw.csv")
        return 0
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
