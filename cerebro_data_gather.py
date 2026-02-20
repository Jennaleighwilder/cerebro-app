#!/usr/bin/env python3
"""
CEREBRO DATA GATHER — Aggressive automated data fetcher
======================================================
Pulls every automatable source from the Cerebro data collection guide.
Runs FRED, World Bank, UCDP, and attempts CDC WONDER.
GSS/Gallup/Pew/ACLED require manual registration — not automatable.

Run: python cerebro_data_gather.py
Output: ./cerebro_data/*.csv and *.xlsx
"""

import json
import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_DIR.mkdir(exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
WB_BASE = "https://api.worldbank.org/v2"
YEARS_START = 1960
YEARS_END = 2025

# Direct download URLs (no auth)
UCDP_ACD_URL = "https://ucdp.uu.se/downloads/replication_data/2025_ucdp-prio-acd-251.xlsx"
UCDP_BRD_URL = "https://ucdp.uu.se/downloads/replication_data/2025_ucdp-brd-dyadic-251.xlsx"
# UCDP GED (georeferenced): https://ucdp.uu.se/downloads/index.html#ged_global — manual download
# ACLED: acleddata.com/data-export-tool — requires registration
# Freedom House: freedomhouse.org/report/freedom-world — CSV tables
# UNHCR Microdata: microdata.unhcr.org — per-dataset

print("=" * 70)
print("CEREBRO DATA GATHER — AGGRESSIVE FETCH")
print("=" * 70)


# ─────────────────────────────────────────
# FRED
# ─────────────────────────────────────────

def fred_series(series_id, start=YEARS_START, end=YEARS_END):
    """Pull annual data from FRED. Returns dict {year: value}."""
    if not FRED_API_KEY:
        print(f"  ✗ FRED {series_id}: FRED_API_KEY not set (add to Vercel env vars)")
        return {}
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": f"{start}-01-01",
        "observation_end": f"{end}-12-31",
        "frequency": "a",
        "aggregation_method": "avg"
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        result = {}
        for obs in data.get("observations", []):
            year = int(obs["date"][:4])
            try:
                result[year] = float(obs["value"])
            except ValueError:
                pass
        print(f"  ✓ FRED {series_id}: {len(result)} obs")
        return result
    except Exception as e:
        print(f"  ✗ FRED {series_id}: {e}")
        return {}


# ─────────────────────────────────────────
# WORLD BANK
# ─────────────────────────────────────────

def worldbank_series(indicator, country="US", start=YEARS_START, end=YEARS_END):
    """Pull annual data from World Bank. Returns dict {year: value}."""
    url = f"{WB_BASE}/country/{country}/indicator/{indicator}"
    params = {"format": "json", "date": f"{start}:{end}", "per_page": 200}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2:
            return {}
        result = {}
        for obs in data[1]:
            if obs.get("value") is not None:
                result[int(obs["date"])] = float(obs["value"])
        print(f"  ✓ World Bank {indicator}: {len(result)} obs")
        return result
    except Exception as e:
        print(f"  ✗ World Bank {indicator}: {e}")
        return {}


# ─────────────────────────────────────────
# DIRECT DOWNLOADS
# ─────────────────────────────────────────

def download_file(url, dest_name, desc="file"):
    """Download file to OUTPUT_DIR. Returns path or None."""
    dest = OUTPUT_DIR / dest_name
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size = dest.stat().st_size
        print(f"  ✓ {desc}: {dest_name} ({size:,} bytes)")
        return dest
    except Exception as e:
        print(f"  ✗ {desc}: {e}")
        return None


# ─────────────────────────────────────────
# PEW TRUST IN GOVERNMENT (Ring B)
# ─────────────────────────────────────────

def fetch_pew_trust():
    """Fetch Pew trust-in-government 1958–present. Returns dict {year: pct_trust}."""
    url = "https://www.pewresearch.org/politics/2025/12/04/public-trust-in-government-1958-2025/"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (compatible; Cerebro/1.0)"})
        r.raise_for_status()
        import re
        # Parse table rows: | M/D/YYYY | Source | N | N |
        rows = re.findall(r'\|\s*(\d{1,2}/\d{1,2}/\d{4})\s*\|\s*[^|]+\|\s*(\d+)\s*\|\s*(\d+)\s*\|', r.text)
        if not rows:
            return _pew_trust_embedded()
        # Build year -> smoothed (last col). Use most recent poll per year.
        by_year = {}
        for date_str, _ind, smoothed in rows:
            try:
                m, d, y = date_str.split("/")
                yr = int(y)
                val = int(smoothed)
                if yr not in by_year or date_str > str(by_year.get(yr, (0, ""))[1]):
                    by_year[yr] = (val, date_str)
            except (ValueError, IndexError):
                pass
        result = {yr: v[0] for yr, v in by_year.items()}
        if len(result) >= 30:
            df = pd.DataFrame([{"year": yr, "trust_pct": v} for yr, v in sorted(result.items())])
            df.to_csv(OUTPUT_DIR / "PEW_trust_government.csv", index=False)
            print(f"  ✓ Pew trust in government: {len(result)} years")
            return result
    except Exception as e:
        print(f"  ✗ Pew trust fetch: {e}")
    return _pew_trust_embedded()


def _pew_trust_embedded():
    """Embedded Pew trust data (1958–2025) from public fact sheet."""
    # Smoothed trend: % trust "just about always" + "most of the time"
    # Source: Pew 2025 fact sheet table
    data = [
        (1958, 73), (1964, 77), (1966, 65), (1968, 62), (1970, 54), (1972, 53),
        (1974, 36), (1976, 34), (1978, 29), (1979, 30), (1980, 27), (1982, 33),
        (1984, 41), (1985, 42), (1986, 44), (1987, 43), (1988, 41), (1989, 41),
        (1990, 35), (1991, 46), (1992, 25), (1993, 25), (1994, 20), (1995, 21),
        (1996, 29), (1997, 27), (1998, 31), (1999, 34), (2000, 34), (2001, 49),
        (2002, 46), (2003, 36), (2004, 39), (2005, 31), (2006, 32), (2007, 28),
        (2008, 25), (2009, 21), (2010, 22), (2011, 18), (2012, 19), (2013, 22),
        (2014, 19), (2015, 18), (2016, 27), (2017, 19), (2018, 18), (2019, 17),
        (2020, 24), (2021, 21), (2022, 20), (2023, 19), (2024, 18), (2025, 17),
    ]
    result = dict(data)
    df = pd.DataFrame([{"year": yr, "trust_pct": v} for yr, v in sorted(result.items())])
    df.to_csv(OUTPUT_DIR / "PEW_trust_government.csv", index=False)
    print(f"  ✓ Pew trust (embedded): {len(result)} years")
    return result


# ─────────────────────────────────────────
# CDC STI (Chlamydia + Gonorrhea) — Embedded
# ─────────────────────────────────────────

def fetch_cdc_sti():
    """
    Embedded chlamydia + gonorrhea rates per 100k (1996-2023).
    CDC WONDER STD requires different XML; using CDC surveillance report data.
    Source: CDC STI Surveillance Reports (annual), Table 1.
    """
    # (year, chlamydia_rate, gonorrhea_rate) per 100k — CDC surveillance
    data = [
        (1996, 181.5, 120.9), (1997, 198.4, 122.5), (1998, 219.4, 132.2),
        (1999, 251.4, 133.2), (2000, 251.8, 131.8), (2001, 261.6, 128.7),
        (2002, 278.3, 123.0), (2003, 304.9, 116.2), (2004, 319.6, 113.5),
        (2005, 340.8, 120.9), (2006, 370.2, 120.9), (2007, 370.2, 118.9),
        (2008, 401.3, 111.6), (2009, 409.2, 98.1), (2010, 426.0, 100.8),
        (2011, 453.4, 104.2), (2012, 456.7, 106.1), (2013, 446.6, 106.1),
        (2014, 456.1, 110.7), (2015, 478.8, 123.9), (2016, 494.7, 145.8),
        (2017, 528.8, 171.9), (2018, 537.5, 178.3), (2019, 551.0, 187.8),
        (2020, 476.7, 204.5), (2021, 495.5, 214.0), (2022, 495.0, 194.4),
        (2023, 494.8, 180.4),
    ]
    result = {}
    for yr, ct, gc in data:
        result[yr] = ct + gc  # combined rate per 100k
    df = pd.DataFrame([
        {"year": yr, "chlamydia_rate": ct, "gonorrhea_rate": gc, "sti_combined_rate_per_100k": ct + gc}
        for yr, ct, gc in data
    ])
    df.to_csv(OUTPUT_DIR / "CDC_STI_rates.csv", index=False)
    print(f"  ✓ CDC STI (embedded): {len(result)} years → CDC_STI_rates.csv")
    return {yr: ct + gc for yr, ct, gc in data}


# ─────────────────────────────────────────
# CDC WONDER (Mortality — drug overdose)
# ─────────────────────────────────────────

def fetch_cdc_overdose():
    """
    CDC WONDER API for drug overdose deaths.
    Database D76 = Detailed Mortality. Requires XML request.
    """
    # ICD-10 codes for drug poisoning: X40-X44, X60-X64, X85, Y10-Y14
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<request-parameters>
    <accept_data_use_restrictions>true</accept_data_use_restrictions>
    <group_by_1>Year</group_by_1>
    <group_by_2>None</group_by_2>
    <group_by_3>None</group_by_3>
    <group_by_4>None</group_by_4>
    <icd10_113_cause_list>X40-X44,X60-X64,X85,Y10-Y14</icd10_113_cause_list>
    <year_option>year_range</year_option>
    <year_range_start>1999</year_range_start>
    <year_range_end>2023</year_range_end>
    <race_option>all</race_option>
    <hispanic_option>all</hispanic_option>
    <gender_option>all</gender_option>
    <age_option>all</age_option>
    <residence_fips>*</residence_fips>
    <urbanization_option>all</urbanization_option>
    <weekday_option>all</weekday_option>
    <authentication>false</authentication>
</request-parameters>"""
    url = "https://wonder.cdc.gov/controller/datarequest/D76"
    try:
        r = requests.post(url, data={"request_xml": xml}, timeout=60)
        r.raise_for_status()
        # Parse XML response for year + deaths
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.content)
        rows = []
        for row in root.findall(".//r"):
            year_el = row.find("c[@n='Year']")
            deaths_el = row.find("c[@n='Deaths']")
            if year_el is not None and deaths_el is not None:
                try:
                    yr = int(year_el.text)
                    deaths = int(deaths_el.text)
                    rows.append({"year": yr, "deaths": deaths})
                except (ValueError, TypeError):
                    pass
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_DIR / "CDC_overdose_deaths.csv", index=False)
            print(f"  ✓ CDC WONDER overdose: {len(rows)} years")
            return df
        print("  ✗ CDC WONDER: no data in response (may need different XML)")
        return None
    except Exception as e:
        print(f"  ✗ CDC WONDER: {e}")
        return None


# ─────────────────────────────────────────
# MAIN GATHER
# ─────────────────────────────────────────

def main():
    start_time = time.time()

    # 1. FRED
    print("\n[1/5] FRED ECONOMIC SERIES")
    fred_series_list = [
        ("UNRATE", "unemployment_rate"),
        ("CIVPART", "labor_force_participation"),
        ("MEHOINUSA672N", "median_household_income"),
        ("PSAVERT", "saving_rate"),
        ("ECIALLCIV", "employment_cost_index"),
        ("GINIALLRF", "gini_coefficient"),   # 1967+
        ("CPIAUCSL", "cpi_inflation"),       # 1947+
        ("MORTGAGE30US", "mortgage_30yr"),   # 1971+
        ("UMCSENT", "consumer_sentiment"),  # U of Michigan consumer sentiment
    ]
    fred_data = {}
    for series_id, name in fred_series_list:
        d = fred_series(series_id)
        if d:
            fred_data[name] = d
        time.sleep(0.3)  # rate limit

    # Save FRED combined
    if fred_data:
        years = sorted(set().union(*[set(d.keys()) for d in fred_data.values()]))
        df_fred = pd.DataFrame(index=years)
        for name, d in fred_data.items():
            df_fred[name] = df_fred.index.map(d)
        df_fred.index.name = "year"
        df_fred.to_csv(OUTPUT_DIR / "FRED_combined.csv")
        print(f"  → Saved FRED_combined.csv ({len(df_fred)} rows)")

    # 2. World Bank (US + OECD + Global 30)
    print("\n[2/5] WORLD BANK")
    GLOBAL_COUNTRIES = {
        "UK": "GBR", "DE": "DEU", "FR": "FRA", "JP": "JPN", "CA": "CAN",
        "AU": "AUS", "SE": "SWE",
        "BR": "BRA", "IN": "IND", "ZA": "ZAF", "TR": "TUR", "KR": "KOR",
        "PL": "POL", "MX": "MEX", "AR": "ARG", "NG": "NGA", "EG": "EGY",
        "ID": "IDN", "TH": "THA", "CO": "COL", "CL": "CHL", "PE": "PER",
        "HU": "HUN", "CZ": "CZE", "GR": "GRC", "PT": "PRT", "NL": "NLD",
        "BE": "BEL", "ES": "ESP", "IT": "ITA",
    }
    oecd_iso = {"US": "US", **GLOBAL_COUNTRIES}
    wb_homicide = worldbank_series("VC.IHR.PSRC.P5", country="US")
    if wb_homicide:
        df_wb = pd.DataFrame([
            {"year": yr, "homicide_rate_per_100k": v}
            for yr, v in sorted(wb_homicide.items())
        ])
        df_wb.to_csv(OUTPUT_DIR / "WorldBank_homicide_US.csv", index=False)

    # US leading indicators (youth unemp, tertiary enrollment, birth rate)
    print("\n[2b/5] WORLD BANK — LEADING INDICATORS (US)")
    wb_youth_unemp = worldbank_series("SL.UEM.1524.ZS", country="US")
    wb_tertiary = worldbank_series("SE.TER.ENRR", country="US")
    wb_birth_rate = worldbank_series("SP.DYN.CBRT.IN", country="US")
    if wb_youth_unemp or wb_tertiary or wb_birth_rate:
        years_li = sorted(set().union(
            set(wb_youth_unemp.keys()) if wb_youth_unemp else set(),
            set(wb_tertiary.keys()) if wb_tertiary else set(),
            set(wb_birth_rate.keys()) if wb_birth_rate else set(),
        ))
        df_li = pd.DataFrame(index=years_li)
        df_li.index.name = "year"
        if wb_youth_unemp:
            df_li["youth_unemployment_pct"] = df_li.index.map(wb_youth_unemp)
        if wb_tertiary:
            df_li["tertiary_enrollment_pct"] = df_li.index.map(wb_tertiary)
        if wb_birth_rate:
            df_li["birth_rate_per_1000"] = df_li.index.map(wb_birth_rate)
        df_li = df_li.reset_index()
        df_li.to_csv(OUTPUT_DIR / "WorldBank_leading_indicators_US.csv", index=False)
        print(f"  ✓ Leading indicators US: youth_unemp, tertiary_enrollment, birth_rate → WorldBank_leading_indicators_US.csv")
    time.sleep(0.5)

    oecd_status = {}
    for label, iso in oecd_iso.items():
        if iso == "US":
            continue
        # Core: homicide, Gini, unemployment. Leading: youth unemployment.
        h = worldbank_series("VC.IHR.PSRC.P5", country=iso)
        g = worldbank_series("SI.POV.GINI", country=iso)
        u = worldbank_series("SL.UEM.TOTL.ZS", country=iso)
        u_youth = worldbank_series("SL.UEM.1524.ZS", country=iso)
        if h or g or u or u_youth:
            years = sorted(set().union(
                set(h.keys()) if h else set(),
                set(g.keys()) if g else set(),
                set(u.keys()) if u else set(),
                set(u_youth.keys()) if u_youth else set(),
            ))
            df_oecd = pd.DataFrame(index=years)
            df_oecd.index.name = "year"
            if h:
                df_oecd["homicide_rate_per_100k"] = df_oecd.index.map(h)
            if g:
                df_oecd["gini"] = df_oecd.index.map(g)
            if u:
                df_oecd["unemployment_pct"] = df_oecd.index.map(u)
            if u_youth:
                df_oecd["youth_unemployment_pct"] = df_oecd.index.map(u_youth)
            df_oecd = df_oecd.reset_index()
            df_oecd.to_csv(OUTPUT_DIR / f"WorldBank_OECD_{iso}.csv", index=False)
            n_h = df_oecd["homicide_rate_per_100k"].notna().sum() if "homicide_rate_per_100k" in df_oecd.columns else 0
            n_g = df_oecd["gini"].notna().sum() if "gini" in df_oecd.columns else 0
            n_u = df_oecd["unemployment_pct"].notna().sum() if "unemployment_pct" in df_oecd.columns else 0
            min_rows = min(n_h, n_g, n_u) if (n_h and n_g and n_u) else 0
            status = "ok" if min_rows >= 30 else "insufficient"
            if status != "ok":
                print(f"  ⚠ World Bank OECD {label} ({iso}): {len(df_oecd)} rows, min non-null={min_rows} (need 30)")
            else:
                print(f"  ✓ World Bank OECD {label} ({iso}): {len(df_oecd)} rows")
            oecd_status[label] = {"rows": len(df_oecd), "homicide_n": int(n_h), "gini_n": int(n_g), "unemp_n": int(n_u), "status": status}
        else:
            print(f"  ✗ World Bank OECD {label} ({iso}): no data")
            oecd_status[label] = {"rows": 0, "status": "missing"}
        time.sleep(0.5)

    (OUTPUT_DIR / "oecd").mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "oecd" / "oecd_status.json", "w") as f:
        json.dump(oecd_status, f, indent=2)
    print(f"  → oecd/oecd_status.json")

    time.sleep(1)

    # 3. UCDP
    print("\n[3/6] UCDP CONFLICT DATA")
    download_file(UCDP_ACD_URL, "UCDP_armed_conflict_251.xlsx", "UCDP Armed Conflict")
    time.sleep(0.5)
    download_file(UCDP_BRD_URL, "UCDP_battle_deaths_251.xlsx", "UCDP Battle Deaths")
    download_file("https://ucdp.uu.se/downloads/ged/ged251-csv.zip", "ucdp_ged.zip", "UCDP GED")
    ucdp_ged_zip = OUTPUT_DIR / "ucdp_ged.zip"
    if ucdp_ged_zip.exists():
        import zipfile
        ucdp_raw = OUTPUT_DIR / "ucdp_raw"
        ucdp_raw.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(ucdp_ged_zip) as z:
                z.extractall(ucdp_raw)
            print(f"  ✓ UCDP GED extracted → {ucdp_raw}")
        except Exception as e:
            print(f"  ✗ UCDP GED extract: {e}")
    # Extract annual global conflict count
    ucdp_path = OUTPUT_DIR / "UCDP_armed_conflict_251.xlsx"
    if ucdp_path.exists():
        try:
            ucdp_df = pd.read_excel(ucdp_path, sheet_name="UcdpPrioConflict_v25_1")
            ucdp_annual = ucdp_df.groupby("year").size().reset_index(name="conflict_count")
            ucdp_annual.to_csv(OUTPUT_DIR / "UCDP_conflict_annual.csv", index=False)
            print(f"  ✓ UCDP conflict annual: {len(ucdp_annual)} years → UCDP_conflict_annual.csv")
        except Exception as e:
            print(f"  ✗ UCDP extract: {e}")

    # 4. Pew Trust (Ring B)
    print("\n[4/6] PEW TRUST IN GOVERNMENT")
    pew_trust = fetch_pew_trust()

    # 5. CDC
    print("\n[5/7] CDC WONDER OVERDOSE")
    fetch_cdc_overdose()
    print("\n[6/7] CDC STI (Chlamydia + Gonorrhea)")
    cdc_sti = fetch_cdc_sti()

    # 7. Build consolidated Cerebro-ready CSV
    print("\n[7/7] BUILDING CONSOLIDATED DATA")
    years = list(range(YEARS_START, YEARS_END + 1))
    df = pd.DataFrame(index=years)
    df.index.name = "year"

    # Merge FRED
    for name, d in fred_data.items():
        df[name] = df.index.map(d)

    # Merge World Bank homicide
    if wb_homicide:
        df["homicide_rate_wb"] = df.index.map(wb_homicide)

    # Merge Pew trust (Ring B)
    if pew_trust:
        df["pew_trust_government_pct"] = df.index.map(pew_trust)

    # Load CDC overdose if we got it
    cdc_path = OUTPUT_DIR / "CDC_overdose_deaths.csv"
    if cdc_path.exists():
        cdc_df = pd.read_csv(cdc_path)
        if "year" in cdc_df.columns and "deaths" in cdc_df.columns:
            cdc_dict = dict(zip(cdc_df["year"], cdc_df["deaths"]))
            df["overdose_deaths"] = df.index.map(cdc_dict)

    # Merge CDC STI
    if cdc_sti:
        df["sti_combined_rate_per_100k"] = df.index.map(cdc_sti)

    # Merge UCDP conflict annual
    ucdp_csv = OUTPUT_DIR / "UCDP_conflict_annual.csv"
    if ucdp_csv.exists():
        ucdp_df = pd.read_csv(ucdp_csv)
        if "year" in ucdp_df.columns and "conflict_count" in ucdp_df.columns:
            ucdp_dict = dict(zip(ucdp_df["year"], ucdp_df["conflict_count"]))
            df["ucdp_conflict_count"] = df.index.map(ucdp_dict)

    # Merge leading indicators (US)
    li_path = OUTPUT_DIR / "WorldBank_leading_indicators_US.csv"
    if li_path.exists():
        li_df = pd.read_csv(li_path)
        for col in ["youth_unemployment_pct", "tertiary_enrollment_pct", "birth_rate_per_1000"]:
            if col in li_df.columns and "year" in li_df.columns:
                li_dict = dict(zip(li_df["year"].astype(int), li_df[col]))
                df[col] = df.index.map(li_dict)

    # Merge ACLED protest annual (if user has placed cerebro_data/ACLED_protest_annual.csv)
    acled_path = OUTPUT_DIR / "ACLED_protest_annual.csv"
    if acled_path.exists():
        acled_df = pd.read_csv(acled_path)
        if "year" in acled_df.columns and "protest_count" in acled_df.columns:
            acled_dict = dict(zip(acled_df["year"].astype(int), acled_df["protest_count"]))
            df["acled_protest_count"] = df.index.map(acled_dict)
            print(f"  ✓ ACLED protest annual merged ({len(acled_dict)} years)")

    df.to_csv(OUTPUT_DIR / "cerebro_gathered_raw.csv")
    print(f"  ✓ cerebro_gathered_raw.csv ({len(df)} rows, {len(df.columns)} cols)")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"DONE in {elapsed:.1f}s. Output: {OUTPUT_DIR}")
    print("=" * 70)
    print("\nMANUAL SOURCES (register & download):")
    print("  • GSS: gssdataexplorer.norc.org — batch extract for Ring B")
    print("  • ISSP: gesis.org/issp — global attitudes (Ring B)")
    print("  • ACLED: acleddata.com/data-export-tool — USA 1997–present")
    print("  • UCDP GED: ucdp.uu.se/downloads — georeferenced 1946–present")
    print("  • Freedom House: freedomhouse.org/report/freedom-world")
    print("  • UNHCR: microdata.unhcr.org")
    print("  • GLOPOP-S, WDI, GBCD: see DATA_SOURCES_REGISTRY.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
