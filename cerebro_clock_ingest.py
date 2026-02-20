#!/usr/bin/env python3
"""
CEREBRO CLASS & SEXUAL CLOCK INGEST
===================================
Builds cerebro_class_clock_data.csv and cerebro_sexual_clock_data.csv
with schema matching harm: year, clock_position_10pt, velocity, acceleration.
Adds analogue memory for class and sexual clocks (Track B symmetry).
"""

import pandas as pd
import numpy as np
from pathlib import Path

from cerebro_causal_normalization import expanding_zscore_to_10pt

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
GATHERED = DATA_DIR / "cerebro_gathered_raw.csv"
HARM_CSV = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
CLASS_CSV = SCRIPT_DIR / "cerebro_class_clock_data.csv"
SEXUAL_CSV = SCRIPT_DIR / "cerebro_sexual_clock_data.csv"


def build_class_clock() -> pd.DataFrame:
    """Class clock: GINI + wage share proxy. Velocity from L1_CLASS, acc from delta(velocity)."""
    if not GATHERED.exists():
        return pd.DataFrame()
    df = pd.read_csv(GATHERED)
    if "year" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns={c: c.strip() for c in df.columns})
    year_col = "year"
    if "gini_coefficient" not in df.columns:
        return pd.DataFrame()
    gini = pd.to_numeric(df["gini_coefficient"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    valid = gini.notna()
    pos_raw = gini[valid]
    if len(pos_raw) < 10:
        return pd.DataFrame()
    pos_10pt = expanding_zscore_to_10pt(pos_raw, min_periods=10)
    out = pd.DataFrame(index=df.loc[valid, "year"].values)
    out.index.name = "year"
    out["clock_position_10pt"] = pos_10pt.values
    vel = out["clock_position_10pt"].diff(3)
    out["velocity"] = vel
    out["acceleration"] = vel.diff(3)
    class_vel_path = DATA_DIR / "GoogleTrends_class_velocity.csv"
    if class_vel_path.exists():
        cv = pd.read_csv(class_vel_path)
        if "year" in cv.columns and "velocity_smooth" in cv.columns:
            vy = cv.groupby("year")["velocity_smooth"].mean()
            out["velocity"] = out["velocity"].fillna(vy.reindex(out.index))
    out = out.dropna(subset=["clock_position_10pt"])
    return out


def build_sexual_clock() -> pd.DataFrame:
    """Sexual clock: STI rates + inverse trust. Velocity from L1 sexual, acc from delta."""
    if not GATHERED.exists():
        return pd.DataFrame()
    df = pd.read_csv(GATHERED)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "year" not in df.columns:
        return pd.DataFrame()
    sti_col = "sti_combined_rate_per_100k"
    trust_col = "pew_trust_government_pct"
    has_sti = sti_col in df.columns
    has_trust = trust_col in df.columns
    if not has_sti and not has_trust:
        sti_path = DATA_DIR / "CDC_STI_rates.csv"
        if sti_path.exists():
            sti_df = pd.read_csv(sti_path)
            if "year" in sti_df.columns and "sti_combined_rate_per_100k" in sti_df.columns:
                df = df.merge(sti_df[["year", "sti_combined_rate_per_100k"]], on="year", how="left")
                has_sti = True
    if not has_sti:
        return pd.DataFrame()
    sti = pd.to_numeric(df[sti_col], errors="coerce")
    pos_raw = sti.copy()
    if has_trust:
        trust = pd.to_numeric(df[trust_col], errors="coerce")
        inv_trust = 100 - trust
        pos_raw = 0.7 * sti.ffill().fillna(0) + 0.3 * (inv_trust.fillna(50) / 50 - 1) * 5
    else:
        pos_raw = sti
    df["year"] = df["year"].astype(int)
    valid = pos_raw.notna()
    if valid.sum() < 10:
        return pd.DataFrame()
    pos_10pt = expanding_zscore_to_10pt(pos_raw[valid], min_periods=10)
    out = pd.DataFrame(index=df.loc[valid, "year"].values)
    out.index.name = "year"
    out["clock_position_10pt"] = pos_10pt.values
    vel = out["clock_position_10pt"].diff(3)
    out["velocity"] = vel
    out["acceleration"] = vel.diff(3)
    sexual_vel_path = DATA_DIR / "GoogleTrends_sexual_velocity.csv"
    if sexual_vel_path.exists():
        sv = pd.read_csv(sexual_vel_path)
        if "year" in sv.columns and "velocity_smooth" in sv.columns:
            vy = sv.groupby("year")["velocity_smooth"].mean()
            out["velocity"] = out["velocity"].fillna(vy.reindex(out.index))
    out = out.dropna(subset=["clock_position_10pt"])
    return out


def main():
    print("CEREBRO CLASS & SEXUAL CLOCK INGEST")
    print("=" * 50)
    class_df = build_class_clock()
    if not class_df.empty:
        class_df.to_csv(CLASS_CSV)
        print(f"  ✓ cerebro_class_clock_data.csv ({len(class_df)} years)")
    else:
        print("  ✗ Class clock: insufficient data")
    sexual_df = build_sexual_clock()
    if not sexual_df.empty:
        sexual_df.to_csv(SEXUAL_CSV)
        print(f"  ✓ cerebro_sexual_clock_data.csv ({len(sexual_df)} years)")
    else:
        print("  ✗ Sexual clock: insufficient data")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
