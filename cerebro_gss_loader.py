#!/usr/bin/env python3
"""
CEREBRO GSS LOADER — Parse SAS, Stata, or SPSS GSS extracts
===========================================================
Searches cerebro_data/ and workspace for *.sas7bdat, *.dta, *.sav
Extracts Ring B variables, computes annual %, saves GSS_RingB_annual.csv
"""

import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# GSS variable -> (column_name, value_to_count for "punitive/tolerance" pole)
# For binary: we want % in the "harm tolerance" or "punitive" direction
RINGB_VARS = {
    "COURTS": ("courts_not_harsh_enough_pct", None),   # % saying "not harshly enough"
    "POLHITOK": ("police_violence_ok_pct", None),      # % YES
    "CAPPUN": ("death_penalty_favor_pct", None),       # % FAVOR
    "GRASS": ("marijuana_legal_pct", None),            # % legal
    "TRUST": ("trust_people_pct", None),               # % CAN TRUST
    "FEAR": ("afraid_night_pct", None),                # % YES (afraid)
    "PREMARSX": ("premarital_ok_pct", None),           # % NOT WRONG AT ALL
    "HOMOSEX": ("homosexuality_ok_pct", None),         # % NOT WRONG AT ALL
    "ABANY": ("abortion_any_reason_pct", None),       # % YES
    "ATTEND": ("church_weekly_pct", None),             # % weekly+
    "EQWLTH": ("income_diff_ok_mean", "mean"),        # 1-7 scale, higher=ok with inequality
    "CLASS": ("working_lower_class_pct", None),       # % working + lower combined
}

# GSS value codes for "positive" response (simplified — may need codebook adjustment)
# Format: variable -> {code: label} where we want to count that code
GSS_CODES = {
    "COURTS": {2: "not harshly enough"},  # 1=too harsh, 2=not harsh enough, 3=about right
    "POLHITOK": {1: "yes"},
    "CAPPUN": {1: "favor"},
    "GRASS": {1: "legal"},
    "TRUST": {1: "can trust"},
    "FEAR": {1: "yes"},
    "PREMARSX": {4: "not wrong at all"},
    "HOMOSEX": {4: "not wrong at all"},
    "ABANY": {1: "yes"},
    "ATTEND": {8: "weekly"},  # 8=every week, 9=more than weekly
    "CLASS": {2: "working", 3: "lower"},  # 1=lower, 2=working, 3=middle, 4=upper
}


def find_gss_file():
    """Find first GSS data file in workspace."""
    search_dirs = [SCRIPT_DIR, OUTPUT_DIR, SCRIPT_DIR.parent]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in ["*.dta", "*.sav", "*.sas7bdat", "*.xpt"]:
            for f in d.rglob(ext):
                if "gss" in f.name.lower() or "GSS" in f.name or f.stat().st_size > 1000:
                    return f
    return None


def load_gss(path):
    """Load GSS file (Stata, SPSS, or SAS)."""
    ext = path.suffix.lower()
    try:
        if ext == ".dta":
            return pd.read_stata(path, convert_categoricals=False)
        elif ext == ".sav":
            return pd.read_spss(path, convert_categoricals=False)
        elif ext in (".sas7bdat", ".xpt"):
            return pd.read_sas(path, format="sas7bdat" if ext == ".sas7bdat" else "xport")
    except Exception as e:
        print(f"  ✗ Load failed: {e}")
        return None
    return None


def compute_annual_pct(df, year_col, var_col, count_codes=None):
    """Compute annual % for a variable. count_codes = set of codes to count, or None for mean."""
    if year_col not in df.columns or var_col not in df.columns:
        return {}
    df = df[[year_col, var_col]].dropna()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[df[year_col].notna()]
    result = {}
    for yr, grp in df.groupby(year_col):
        yr = int(yr)
        if count_codes is not None:
            n = grp[var_col].isin(count_codes).sum()
            total = len(grp)
            result[yr] = 100 * n / total if total > 0 else None
        else:
            result[yr] = grp[var_col].mean()
    return result


def main():
    print("=" * 60)
    print("CEREBRO GSS LOADER")
    print("=" * 60)

    path = find_gss_file()
    if path is None:
        print("\n  No GSS file found (.dta, .sav, .sas7bdat, .xpt)")
        print("  Place files in: cerebro_data/ or workspace root")
        return 0  # Not an error — just skip

    print(f"\n  Found: {path.name} ({path.stat().st_size:,} bytes)")
    df = load_gss(path)
    if df is None:
        return 1

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)[:15]}...")

    # Detect year column (YEAR, year, etc.)
    year_col = None
    for c in ["YEAR", "year", "Year"]:
        if c in df.columns:
            year_col = c
            break
    if year_col is None:
        print("  ✗ No YEAR column found")
        return 1

    # Build annual series for each variable
    annual = {}
    for gss_var, (out_name, agg) in RINGB_VARS.items():
        # Try exact match, then case-insensitive
        var_col = None
        for c in df.columns:
            if c.upper() == gss_var.upper():
                var_col = c
                break
        if var_col is None:
            continue
        codes = GSS_CODES.get(gss_var)
        if agg == "mean":
            s = compute_annual_pct(df, year_col, var_col, count_codes=None)
        else:
            count_codes = codes.keys() if isinstance(codes, dict) else None
            if gss_var == "CLASS":
                count_codes = {2, 3}  # working + lower
            s = compute_annual_pct(df, year_col, var_col, count_codes=count_codes)
        if s:
            annual[out_name] = s
            print(f"  ✓ {gss_var} -> {out_name} ({len(s)} years)")

    if not annual:
        print("  ✗ No variables extracted")
        return 1

    # Build output DataFrame
    all_years = sorted(set().union(*[set(d.keys()) for d in annual.values()]))
    out = pd.DataFrame(index=all_years)
    out.index.name = "year"
    for name, d in annual.items():
        out[name] = out.index.map(d)

    out_path = OUTPUT_DIR / "GSS_RingB_annual.csv"
    out.to_csv(out_path)
    print(f"\n  → Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
