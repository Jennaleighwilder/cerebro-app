#!/usr/bin/env python3
"""
cerebro_ucdp_loader.py
Parses UCDP GED dataset into annual protest/civil unrest counts
in the same format as ACLED_protest_annual.csv
Output: cerebro_data/ACLED_protest_annual.csv (used by phase1 + OECD clocks)
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
UCDP_DIR = SCRIPT_DIR / "cerebro_data" / "ucdp_raw"
OUTPUT = SCRIPT_DIR / "cerebro_data" / "ACLED_protest_annual.csv"

# UCDP event types: 1=state-based, 2=non-state, 3=one-sided
# All three = organized violence / civil unrest proxy (precedes saddle points by 2-4 years)
# Type 2+3 alone yields few US events; including 1 adds state-actor events (Waco, OKC, etc.)
UNREST_TYPES = [1, 2, 3]

COUNTRY_MAP = {
    "United States": "US",
    "United States of America": "US",
    "Brazil": "BR",
    "France": "FR",
    "Spain": "ES",
    "Canada": "CA",
    "United Kingdom": "UK",
    "Germany": "DE",
    "Japan": "JP",
    "Australia": "AU",
    "Sweden": "SE",
    "Poland": "PL",
    "Hungary": "HU",
    "Nigeria": "NG",
    "Colombia": "CO",
    "Argentina": "AR",
    "Peru": "PE",
    "Mexico": "MX",
    "Indonesia": "ID",
    "India": "IN",
    "South Africa": "ZA",
    "Turkey": "TR",
    "South Korea": "KR",
    "Korea, South": "KR",
    "Netherlands": "NL",
    "Belgium": "BE",
    "Portugal": "PT",
    "Czech Republic": "CZ",
    "Czechia": "CZ",
    "Italy": "IT",
    "Greece": "GR",
    "Chile": "CL",
    "Egypt": "EG",
    "Thailand": "TH",
}


def load_ucdp() -> pd.DataFrame:
    """Find and load the UCDP GED CSV from ucdp_raw/."""
    csvs = list(UCDP_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {UCDP_DIR}. Run: curl -L https://ucdp.uu.se/downloads/ged/ged251-csv.zip -o cerebro_data/ucdp_ged.zip && unzip -o cerebro_data/ucdp_ged.zip -d cerebro_data/ucdp_raw/")
    df = pd.read_csv(csvs[0], low_memory=False)
    print(f"UCDP loaded: {len(df)} rows, columns: {list(df.columns[:10])}")
    return df


def parse_to_annual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate UCDP events to annual protest counts per country.
    Output format: country_iso, year, protest_count
    For US-only consumers (phase1): also write year, protest_count subset.
    """
    if "type_of_violence" not in df.columns or "country" not in df.columns or "year" not in df.columns:
        raise ValueError(f"UCDP GED missing required columns. Got: {list(df.columns)}")
    unrest = df[df["type_of_violence"].isin(UNREST_TYPES)].copy()
    unrest["country_iso"] = unrest["country"].map(COUNTRY_MAP)
    unrest = unrest.dropna(subset=["country_iso"])
    annual = unrest.groupby(["country_iso", "year"]).size().reset_index(name="protest_count")
    annual["year"] = annual["year"].astype(int)
    return annual


def save_annual(annual: pd.DataFrame) -> None:
    """Save to ACLED_protest_annual.csv. Phase1 expects year, protest_count for US."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(OUTPUT, index=False)
    print(f"Saved {len(annual)} country-year rows to {OUTPUT}")
    top = annual.groupby("country_iso")["protest_count"].sum().sort_values(ascending=False).head(10)
    print("Top countries by total events:", top.to_dict())
    us_rows = annual[annual["country_iso"] == "US"]
    if len(us_rows) > 0:
        print(f"US: {len(us_rows)} years, total {us_rows['protest_count'].sum()} events")


def run() -> pd.DataFrame:
    df = load_ucdp()
    annual = parse_to_annual(df)
    save_annual(annual)
    return annual


if __name__ == "__main__":
    run()
