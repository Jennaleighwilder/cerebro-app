#!/usr/bin/env python3
"""Export cerebro_harm_clock_data.csv to JSON for frontend. Run after phase1 ingest."""

import json
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
OUT_JSON = SCRIPT_DIR / "public" / "cerebro_data.json"
OUT_JSON.parent.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(CSV_PATH, index_col=0)
    # Get last 10 years with full clock data
    df = df[df["clock_position_10pt"].notna()].tail(15)
    data = {
        "harm_clock": {
            "latest_year": int(df.index[-1]),
            "position": round(float(df["clock_position_10pt"].iloc[-1]), 2),
            "velocity": round(float(df["velocity"].iloc[-1]), 4),
            "acceleration": round(float(df["acceleration"].iloc[-1]), 4),
            "saddle_score": int(df["saddle_score"].iloc[-1]) if pd.notna(df["saddle_score"].iloc[-1]) else 0,
            "saddle_label": str(df["saddle_label"].iloc[-1]) if pd.notna(df["saddle_label"].iloc[-1]) else "",
            "ring_B_pct": round(float(df["ring_B_score"].iloc[-1]) * 100, 0) if pd.notna(df["ring_B_score"].iloc[-1]) else 0,
            "ring_A_pct": round(float(df["ring_A_score"].iloc[-1]) * 100, 0) if pd.notna(df["ring_A_score"].iloc[-1]) else 0,
            "ring_C_pct": round(float(df["ring_C_score"].iloc[-1]) * 100, 0) if pd.notna(df["ring_C_score"].iloc[-1]) else 0,
        },
        "raw_series": df[["clock_position_10pt", "velocity", "acceleration", "saddle_score"]].round(4).to_dict(orient="index"),
        "indicators": {
            "unemployment": round(float(df["unemployment_rate"].iloc[-1]), 1) if "unemployment_rate" in df.columns else None,
            "homicide_rate": round(float(df["homicide_rate"].iloc[-1]), 1) if "homicide_rate" in df.columns else None,
            "incarceration": int(float(df["incarceration_rate_bjs"].iloc[-1])) if "incarceration_rate_bjs" in df.columns else None,
            "overdose_rate": round(float(df["overdose_death_rate_cdc"].iloc[-1]), 1) if "overdose_death_rate_cdc" in df.columns else None,
        },
        "ring_b_loaded": bool(df["ring_B_score"].notna().any() and df["ring_B_score"].notna().sum() > 10),
    }
    # Convert to JSON-serializable types
    def to_json_val(x):
        if pd.isna(x): return None
        if isinstance(x, (bool, type(None))): return x
        if isinstance(x, (int, float)): return round(float(x), 4) if isinstance(x, float) else int(x)
        return str(x)

    data["raw_series"] = {str(k): {kk: to_json_val(vv) for kk, vv in v.items()}
                          for k, v in data["raw_series"].items()}

    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported: {OUT_JSON}")

if __name__ == "__main__":
    main()
