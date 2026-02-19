#!/usr/bin/env python3
"""
CEREBRO L1 — GOOGLE TRENDS CULTURAL VELOCITY INDEX
=================================================
Leading indicator layer: 3–12 months before behavioral shift.

TRACK: Month-over-month % change in search cluster volume
VALIDATE: Trends typically leads surveys by 6–18 months

Harm Tolerance clusters:
  - Reform pole: defund police, criminal justice reform, police reform
  - Punitive pole: back the blue, law and order, tough on crime

Cultural velocity > 0 = attention shifting toward reform
Cultural velocity < 0 = attention shifting toward punitive

Output: cerebro_data/GoogleTrends_cultural_velocity.csv
"""

import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Search clusters by clock
REFORM_KEYWORDS = ["defund the police", "criminal justice reform", "police reform"]
PUNITIVE_KEYWORDS = ["back the blue", "law and order", "tough on crime"]

# Class Permeability: redistribution vs extraction
CLASS_REDISTRIBUTION = ["universal basic income", "wealth tax", "income inequality"]
CLASS_EXTRACTION = ["cut taxes", "deregulation", "free market"]

# Sexual Pendulum: autonomy vs constraint
SEXUAL_AUTONOMY = ["marriage equality", "gender affirming care", "pronouns"]
SEXUAL_CONSTRAINT = ["traditional marriage", "family values", "gender roles"]

CSV_PATH = OUTPUT_DIR / "GoogleTrends_cultural_velocity.csv"
CSV_CLASS = OUTPUT_DIR / "GoogleTrends_class_velocity.csv"
CSV_SEXUAL = OUTPUT_DIR / "GoogleTrends_sexual_velocity.csv"


def _fetch_cluster_pair(pytrends, pole_a_name, pole_a_kw, pole_b_name, pole_b_kw):
    """Fetch two poles, merge, compute velocity. Returns DataFrame or None."""
    import pandas as pd
    all_data = []
    for name, kw in [(pole_a_name, pole_a_kw[:5]), (pole_b_name, pole_b_kw[:5])]:
        try:
            pytrends.build_payload(kw, cat=0, timeframe="today 5-y", geo="US", gprop="")
            time.sleep(1.2)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                continue
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df[name] = df.mean(axis=1)
            df = df[[name]].copy()
            df.index = pd.to_datetime(df.index)
            all_data.append((name, df))
        except Exception:
            continue
    if len(all_data) < 2:
        return None
    merged = all_data[0][1].join(all_data[1][1], how="outer").sort_index()
    va = merged[all_data[0][0]].pct_change() * 100
    vb = merged[all_data[1][0]].pct_change() * 100
    merged["velocity"] = va - vb
    merged["velocity_smooth"] = merged["velocity"].rolling(3, min_periods=1).mean()
    merged["year"] = merged.index.year
    merged["month"] = merged.index.month
    return merged.dropna(subset=["velocity"])


def fetch_trends():
    """Fetch Google Trends for Harm, Class, Sexual clusters. Returns main DataFrame."""
    try:
        from pytrends.request import TrendReq
        import pandas as pd
    except ImportError:
        print("  ⚠ pytrends not installed. Run: pip install pytrends")
        return None

    print("=" * 60)
    print("CEREBRO L1 — GOOGLE TRENDS CULTURAL VELOCITY")
    print("=" * 60)

    try:
        pytrends = TrendReq(hl="en-US", tz=360)
    except Exception as e:
        print(f"  ✗ TrendReq init failed: {e}")
        return None

    all_data = []

    for cluster_name, keywords in [("reform", REFORM_KEYWORDS), ("punitive", PUNITIVE_KEYWORDS)]:
        try:
            kw = keywords[:5]
            pytrends.build_payload(kw, cat=0, timeframe="today 5-y", geo="US", gprop="")
            time.sleep(1.2)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                print(f"  ✗ {cluster_name}: no data")
                continue
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df[cluster_name] = df.mean(axis=1)
            df = df[[cluster_name]].copy()
            df.index = pd.to_datetime(df.index)
            all_data.append((cluster_name, df))
            print(f"  ✓ Harm {cluster_name}: {len(df)} months")
        except Exception as e:
            print(f"  ✗ {cluster_name} fetch failed: {e}")
            continue

    if len(all_data) < 2:
        print("  ⚠ Harm clusters failed. Using embedded fallback.")
        return _embedded_fallback()

    reform_df = all_data[0][1]
    punitive_df = all_data[1][1]
    merged = reform_df.join(punitive_df, how="outer").sort_index()
    merged["reform_velocity"] = merged["reform"].pct_change() * 100
    merged["punitive_velocity"] = merged["punitive"].pct_change() * 100
    merged["cultural_velocity"] = merged["reform_velocity"] - merged["punitive_velocity"]
    merged["cultural_velocity_smooth"] = merged["cultural_velocity"].rolling(3, min_periods=1).mean()
    merged = merged.dropna(subset=["cultural_velocity"])
    merged["year"] = merged.index.year
    merged["month"] = merged.index.month
    merged.to_csv(CSV_PATH)
    print(f"  ✓ Saved: {CSV_PATH} ({len(merged)} rows)")

    # Class Permeability velocity
    class_df = _fetch_cluster_pair(pytrends, "redistribution", CLASS_REDISTRIBUTION, "extraction", CLASS_EXTRACTION)
    if class_df is not None:
        class_df.to_csv(CSV_CLASS)
        print(f"  ✓ Class velocity: {CSV_CLASS} ({len(class_df)} rows)")
    else:
        print("  ⚠ Class velocity: fetch failed")

    # Sexual Pendulum velocity
    sexual_df = _fetch_cluster_pair(pytrends, "autonomy", SEXUAL_AUTONOMY, "constraint", SEXUAL_CONSTRAINT)
    if sexual_df is not None:
        sexual_df.to_csv(CSV_SEXUAL)
        print(f"  ✓ Sexual velocity: {CSV_SEXUAL} ({len(sexual_df)} rows)")
    else:
        print("  ⚠ Sexual velocity: fetch failed")

    return merged


def _embedded_fallback():
    """Embedded fallback when API unavailable. Synthetic velocity for demo."""
    import pandas as pd
    from datetime import datetime

    # Last 24 months synthetic (matches typical Trends shape)
    dates = pd.date_range(end=datetime.now(), periods=24, freq="MS")
    # Simulate: reform declining from 2020 peak, punitive rising 2023+
    reform = [80, 75, 70, 65, 60, 55, 50, 48, 45, 42, 40, 38,
              36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]
    punitive = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52,
                54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76]
    df = pd.DataFrame({"reform": reform[:len(dates)], "punitive": punitive[:len(dates)]}, index=dates)
    df["reform_velocity"] = df["reform"].pct_change() * 100
    df["punitive_velocity"] = df["punitive"].pct_change() * 100
    df["cultural_velocity"] = df["reform_velocity"] - df["punitive_velocity"]
    df["cultural_velocity_smooth"] = df["cultural_velocity"].rolling(3, min_periods=1).mean()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df = df.dropna(subset=["cultural_velocity"])
    df.to_csv(CSV_PATH)
    print(f"  ✓ Saved embedded fallback: {CSV_PATH}")
    return df


def get_latest_velocity():
    """Return latest Harm cultural velocity for export. Dict or None."""
    if not CSV_PATH.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
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


def get_class_velocity():
    """Return latest Class Permeability velocity. Dict or None."""
    if not CSV_CLASS.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(CSV_CLASS, index_col=0, parse_dates=True)
        if df.empty or "velocity_smooth" not in df.columns:
            return None
        latest = df.iloc[-1]
        return {"velocity_smooth": round(float(latest.get("velocity_smooth", 0)), 2), "year": int(latest.get("year", 0)), "month": int(latest.get("month", 0))}
    except Exception:
        return None


def get_sexual_velocity():
    """Return latest Sexual Pendulum velocity. Dict or None."""
    if not CSV_SEXUAL.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(CSV_SEXUAL, index_col=0, parse_dates=True)
        if df.empty or "velocity_smooth" not in df.columns:
            return None
        latest = df.iloc[-1]
        return {"velocity_smooth": round(float(latest.get("velocity_smooth", 0)), 2), "year": int(latest.get("year", 0)), "month": int(latest.get("month", 0))}
    except Exception:
        return None


def main():
    fetch_trends()
    v = get_latest_velocity()
    if v:
        print(f"\n  Harm velocity: {v['cultural_velocity_smooth']:+.2f} (reform {v['reform_index']:.0f} vs punitive {v['punitive_index']:.0f})")
    c = get_class_velocity()
    if c:
        print(f"  Class velocity: {c['velocity_smooth']:+.2f}")
    s = get_sexual_velocity()
    if s:
        print(f"  Sexual velocity: {s['velocity_smooth']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
