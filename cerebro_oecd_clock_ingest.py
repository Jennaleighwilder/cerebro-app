#!/usr/bin/env python3
"""
CEREBRO OECD CLOCK INGEST — Multi-country harm clocks for calibration expansion
===============================================================================
Builds clocks for UK, Germany, France, Japan, Canada using:
  - World Bank: homicide, GINI, unemployment
  - Causal normalization (expanding z-score, no future leakage)
  - Phase1 saddle scoring (|v|<0.20 + sign opposes)

Output: cerebro_data/oecd/{UK,DE,FR,JP,CA}_clock.csv
Schema: year, position, velocity, acceleration, saddle_score_phase1

Does NOT touch cerebro_core. Used for calibration, rolling origin, cross-national.
"""

import json
import pandas as pd
from pathlib import Path

from cerebro_causal_normalization import (
    expanding_zscore_to_10pt,
    causal_velocity,
    causal_acceleration,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
OECD_DIR = DATA_DIR / "oecd"
OECD_DIR.mkdir(parents=True, exist_ok=True)

OECD_COUNTRIES = {
    "UK": "GBR", "DE": "DEU", "FR": "FRA", "JP": "JPN", "CA": "CAN",
    "AU": "AUS", "SE": "SWE",
    "BR": "BRA", "IN": "IND", "ZA": "ZAF", "TR": "TUR", "KR": "KOR",
    "PL": "POL", "MX": "MEX", "AR": "ARG", "NG": "NGA", "EG": "EGY",
    "ID": "IDN", "TH": "THA", "CO": "COL", "CL": "CHL", "PE": "PER",
    "HU": "HUN", "CZ": "CZE", "GR": "GRC", "PT": "PRT", "NL": "NLD",
    "BE": "BEL", "ES": "ESP", "IT": "ITA",
}

# Regime pivot years per country (for event_year assignment in calibration)
OECD_EVENT_YEARS = {
    "UK": [1979, 1997, 2010, 2016, 2020],
    "DE": [1989, 1990, 2005, 2009, 2015, 2020],
    "FR": [1981, 1995, 2002, 2008, 2017, 2020],
    "JP": [1989, 1997, 2001, 2008, 2011, 2020],
    "CA": [1984, 1993, 2006, 2015, 2020],
    "AU": [1975, 1983, 1996, 2007, 2013, 2020],
    "SE": [1976, 1991, 2006, 2014, 2020],
    "BR": [1985, 1994, 2002, 2013, 2016, 2018],
    "TR": [1980, 1997, 2002, 2013, 2016],
    "KR": [1987, 1997, 2002, 2016],
    "ZA": [1994, 2008, 2012, 2018],
    "PL": [1989, 2004, 2015, 2020],
    "AR": [1983, 1995, 2001, 2015, 2019],
    "GR": [2010, 2012, 2015],
    "ES": [2008, 2011, 2017],
    "IT": [1992, 2011, 2018],
    "IN": [1991, 2002, 2014, 2019],
    "MX": [1994, 2000, 2006, 2018],
    "CO": [1991, 2002, 2016],
    "CL": [1990, 2006, 2011, 2019],
    "HU": [1989, 2004, 2010],
    "ID": [1998, 2004, 2014],
    "PT": [1974, 1986, 2011],
    "NL": [1982, 2008, 2012],
    "BE": [1981, 2008, 2011],
    "CZ": [1989, 2004, 2013],
    "TH": [1992, 1997, 2006, 2014],
    "PE": [1990, 2000, 2006, 2016],
    "NG": [1999, 2007, 2015],
    "EG": [2011, 2013, 2014],
}

EVENT_TOLERANCE = 10
YEARS_START = 1960
YEARS_END = 2024


def load_country_raw(country_code: str) -> pd.DataFrame:
    """
    Load raw indicators from gathered data. No network fetch.
    Reads WorldBank_* and cerebro_gathered_raw if available.
    """
    wb_dir = DATA_DIR
    iso = OECD_COUNTRIES.get(country_code, country_code)

    homicide = {}
    gini = {}
    unemp = {}
    youth_unemp = {}

    # Primary: WorldBank_OECD_{iso}.csv (from cerebro_data_gather)
    p = wb_dir / f"WorldBank_OECD_{iso}.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if "year" in df.columns:
                for _, row in df.iterrows():
                    yr = int(row["year"])
                    if "homicide_rate_per_100k" in df.columns and pd.notna(row.get("homicide_rate_per_100k")):
                        homicide[yr] = float(row["homicide_rate_per_100k"])
                    if "gini" in df.columns and pd.notna(row.get("gini")):
                        gini[yr] = float(row["gini"])
                    if "unemployment_pct" in df.columns and pd.notna(row.get("unemployment_pct")):
                        unemp[yr] = float(row["unemployment_pct"])
                    if "youth_unemployment_pct" in df.columns and pd.notna(row.get("youth_unemployment_pct")):
                        youth_unemp[yr] = float(row["youth_unemployment_pct"])
        except Exception:
            pass

    # Fallback: WorldBank_homicide_{iso}.csv etc (legacy)
    if not homicide and (wb_dir / f"WorldBank_homicide_{iso}.csv").exists():
        try:
            df = pd.read_csv(wb_dir / f"WorldBank_homicide_{iso}.csv")
            homicide = dict(zip(df["year"].astype(int), df["homicide_rate_per_100k"]))
        except Exception:
            pass

    # UCDP protest: disabled — conflict counts ≠ protest intensity, contaminated signal

    years = sorted(set(range(YEARS_START, YEARS_END + 1)))
    harm_proxy = []
    unemployment = []
    youth_unemployment = []
    inequality = []

    for yr in years:
        h = homicide.get(yr)
        g = gini.get(yr)
        u = unemp.get(yr)
        uy = youth_unemp.get(yr)
        harm_proxy.append(h if h is not None and pd.notna(h) else 5.0)
        unemployment.append(u if u is not None and pd.notna(u) else 6.0)
        youth_unemployment.append(uy if uy is not None and pd.notna(uy) else 12.0)
        inequality.append(g if g is not None and pd.notna(g) else 32.0)

    return pd.DataFrame({
        "year": years,
        "harm_proxy": harm_proxy,
        "unemployment": unemployment,
        "youth_unemployment": youth_unemployment,
        "inequality": inequality,
    })


def detect_saddle_phase1(df: pd.DataFrame) -> pd.Series:
    """
    Phase1 tension rule: saddle_score = (|v|<0.20) + (sign opposes).
    Same logic as calibration expansion, not core.
    """
    v = df["velocity"]
    a = df["acceleration"]
    score = (
        (v.abs() < 0.20).astype(int) +
        ((v * a) < 0).astype(int)
    )
    return score


def build_country_clock(country: str) -> bool:
    """Build clock for one country. Returns True if successful."""
    raw = load_country_raw(country)
    if raw.empty or len(raw) < 20:
        return False

    raw = raw.dropna(subset=["harm_proxy", "unemployment", "inequality"], how="all")
    if len(raw) < 20:
        return False

    # Composite harm signal: lagging (homicide, unemp, gini) + leading (youth unemp)
    # UCDP protest: disabled — zero weight
    composite = (
        raw["harm_proxy"].fillna(5)
        + raw["unemployment"].fillna(6) * 0.4
        + raw["youth_unemployment"].fillna(12) * 0.25
        + raw["inequality"].fillna(32) * 0.1
    )
    raw = raw.set_index("year")

    pos = expanding_zscore_to_10pt(composite, min_periods=15)
    vel = causal_velocity(pos)
    acc = causal_acceleration(vel)

    df = pd.DataFrame({
        "year": raw.index,
        "position": pos.values,
        "velocity": vel.values,
        "acceleration": acc.values,
    })
    df["saddle_score_phase1"] = detect_saddle_phase1(df)

    df = df.dropna(subset=["position", "velocity", "acceleration"], how="all")
    if len(df) < 15:
        return False

    out_path = OECD_DIR / f"{country}_clock.csv"
    df.to_csv(out_path, index=False)
    return True


def _write_oecd_status():
    """Write oecd_status.json reflecting current data availability (prevents silent fallback)."""
    status = {}
    for country in OECD_COUNTRIES:
        iso = OECD_COUNTRIES[country]
        p = DATA_DIR / f"WorldBank_OECD_{iso}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                n_h = df["homicide_rate_per_100k"].notna().sum() if "homicide_rate_per_100k" in df.columns else 0
                n_g = df["gini"].notna().sum() if "gini" in df.columns else 0
                n_u = df["unemployment_pct"].notna().sum() if "unemployment_pct" in df.columns else 0
                min_n = min(n_h, n_g, n_u) if (n_h and n_g and n_u) else 0
                status[country] = {"rows": len(df), "homicide_n": int(n_h), "gini_n": int(n_g), "unemp_n": int(n_u), "status": "ok" if min_n >= 25 else "insufficient"}
            except Exception:
                status[country] = {"rows": 0, "status": "error"}
        else:
            status[country] = {"rows": 0, "status": "missing"}
    OECD_DIR.mkdir(parents=True, exist_ok=True)
    with open(OECD_DIR / "oecd_status.json", "w") as f:
        json.dump(status, f, indent=2)


def main():
    print("CEREBRO OECD Clock Ingest")
    print("=" * 50)
    _write_oecd_status()
    # Log when using fallback (prevents silent flat clocks)
    status_path = OECD_DIR / "oecd_status.json"
    if status_path.exists():
        try:
            with open(status_path) as f:
                st = json.load(f)
            fallback = [c for c, v in st.items() if v.get("status") in ("missing", "insufficient")]
            if fallback:
                print(f"  ⚠ OECD fallback for {fallback}: clocks use flat defaults (run cerebro_data_gather with network)")
        except Exception:
            pass
    built = 0
    for country in OECD_COUNTRIES:
        try:
            if build_country_clock(country):
                print(f"  ✓ {country}_clock.csv")
                built += 1
            else:
                print(f"  ✗ {country}: insufficient data")
        except Exception as e:
            print(f"  ✗ {country}: {e}")
    print(f"Built {built} OECD clocks → {OECD_DIR}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
