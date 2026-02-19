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


def _load_cultural_velocity():
    """Load L1 Google Trends cultural velocity if available."""
    csv_path = SCRIPT_DIR / "cerebro_data" / "GoogleTrends_cultural_velocity.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty or "cultural_velocity_smooth" not in df.columns:
            return None
        latest = df.iloc[-1]
        return {
            "cultural_velocity": round(float(latest.get("cultural_velocity", 0)), 2),
            "cultural_velocity_smooth": round(float(latest.get("cultural_velocity_smooth", 0)), 2),
            "reform_index": round(float(latest.get("reform", 0)), 1),
            "punitive_index": round(float(latest.get("punitive", 0)), 1),
            "year": int(latest.get("year", 0)),
            "month": int(latest.get("month", 0)),
            "lead_time_months": "3–12",
        }
    except Exception:
        return None


def _load_trends_velocity(csv_name, vel_col="velocity_smooth"):
    """Load Class or Sexual velocity from CSV."""
    p = SCRIPT_DIR / "cerebro_data" / csv_name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.empty or vel_col not in df.columns:
            return None
        latest = df.iloc[-1]
        return {"velocity_smooth": round(float(latest.get(vel_col, 0)), 2), "year": int(latest.get("year", 0)), "month": int(latest.get("month", 0))}
    except Exception:
        return None


def _load_gathered_indicators():
    """Load latest Class/Sexual/Conflict indicators from cerebro_gathered_raw."""
    p = SCRIPT_DIR / "cerebro_data" / "cerebro_gathered_raw.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p).dropna(subset=["year"]).sort_values("year")
        if df.empty:
            return {}
        out = {}
        for col, key, fmt in [
            ("gini_coefficient", "gini", lambda x: round(float(x), 3)),
            ("consumer_sentiment", "consumer_sentiment", lambda x: round(float(x), 1)),
            ("sti_combined_rate_per_100k", "sti_rate", lambda x: round(float(x), 1)),
            ("ucdp_conflict_count", "ucdp_conflicts", lambda x: int(x)),
        ]:
            if col in df.columns:
                last_valid = df[col].dropna()
                if len(last_valid) > 0:
                    out[key] = fmt(last_valid.iloc[-1])
        return out
    except Exception:
        return {}


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
        "cultural_velocity": _load_cultural_velocity(),
        "class_velocity": _load_trends_velocity("GoogleTrends_class_velocity.csv"),
        "sexual_velocity": _load_trends_velocity("GoogleTrends_sexual_velocity.csv"),
        "aux_indicators": _load_gathered_indicators(),
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
