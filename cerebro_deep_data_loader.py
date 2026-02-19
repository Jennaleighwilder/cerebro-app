#!/usr/bin/env python3
"""
CEREBRO DEEP DATA LOADER — Foundation-level academic datasets
============================================================
Adds GLOPOP-S, ISSP, GBCD, NASA, World Development Indicators.
Each dataset has Ring assignment (A/B/C/L1/L2) and confidence scoring.

Run: python cerebro_deep_data_loader.py
Output: cerebro_data/*.csv with new columns + data_sources_status.json
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Ring assignments and confidence (from DATA_SOURCES_REGISTRY)
DEEP_SOURCES = {
    "glopop_s": {"ring": "A", "confidence": 92, "clocks": ["class", "harm", "sexual", "evil"]},
    "issp": {"ring": "B", "confidence": 88, "clocks": ["class", "sexual", "harm", "evil"]},
    "gbcd": {"ring": "L2", "confidence": 82, "clocks": ["class", "evil"]},
    "nasa_socio": {"ring": "C", "confidence": 85, "clocks": ["class", "evil"]},
    "wdi": {"ring": "A", "confidence": 90, "clocks": ["class", "evil", "harm"]},
    "ucdp_ged": {"ring": "C", "confidence": 92, "clocks": ["evil"]},
    "acled_full": {"ring": "C", "confidence": 88, "clocks": ["evil", "harm"]},
    "freedom_house": {"ring": "C", "confidence": 88, "clocks": ["evil"]},
    "unhcr_micro": {"ring": "C", "confidence": 85, "clocks": ["evil"]},
}


def _write_status(sources_status):
    """Write data_sources_status.json for UI."""
    out = {
        "deep_sources": sources_status,
        "rings_loaded": {
            "A": [k for k, v in sources_status.items() if v.get("live") and DEEP_SOURCES.get(k, {}).get("ring") == "A"],
            "B": [k for k, v in sources_status.items() if v.get("live") and DEEP_SOURCES.get(k, {}).get("ring") == "B"],
            "C": [k for k, v in sources_status.items() if v.get("live") and DEEP_SOURCES.get(k, {}).get("ring") == "C"],
            "L1": [k for k, v in sources_status.items() if v.get("live") and DEEP_SOURCES.get(k, {}).get("ring") == "L1"],
            "L2": [k for k, v in sources_status.items() if v.get("live") and DEEP_SOURCES.get(k, {}).get("ring") == "L2"],
        },
    }
    with open(OUTPUT_DIR / "data_sources_status.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → data_sources_status.json")


# ─────────────────────────────────────────
# 1. GLOPOP-S — 7.3B individuals (aggregate for Class)
# ─────────────────────────────────────────

def load_glopop_s():
    """Load GLOPOP-S if available. Expects cerebro_data/GLOPOP-S/ or GLOPOP-S*.csv.
    Data: Harvard Dataverse https://doi.org/10.7910/DVN/KJC3RH
    Code: https://github.com/VU-IVM/GLOPOP-S/ (read_synthpop_data.py)"""
    glopop_dir = OUTPUT_DIR / "GLOPOP-S"
    paths = list(OUTPUT_DIR.glob("GLOPOP-S*.csv")) + list(OUTPUT_DIR.glob("glopop*.csv"))
    if glopop_dir.exists():
        paths = list(glopop_dir.rglob("*.csv")) + list(glopop_dir.rglob("*.bin")) + list(glopop_dir.rglob("*.dat.gz"))
    if not paths:
        return None, "Download from https://doi.org/10.7910/DVN/KJC3RH, use read_synthpop_data.py from github.com/VU-IVM/GLOPOP-S"
    try:
        import pandas as pd
        p0 = paths[0]
        if p0.suffix == ".gz" or ".dat.gz" in str(p0):
            # Synthpop binary format — count files as proxy for 7.3B individuals
            n_files = len(list(glopop_dir.rglob("*.dat.gz"))) if glopop_dir.exists() else 1
            return {"n_synthpop_files": n_files, "format": "synthpop"}, None
        df = pd.read_csv(p0)
        # Expect: region, income, wealth, education, household_type, etc.
        if df.empty or len(df) < 100:
            return None, "GLOPOP-S file too small"
        # Aggregate to annual Gini-like for Class clock
        if "income" in df.columns:
            gini_proxy = df["income"].std() / (df["income"].mean() + 1e-6) if df["income"].mean() else 0
            return {"gini_glopop": gini_proxy, "n_individuals": len(df)}, None
        return {"n_individuals": len(df)}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 2. ISSP Cumulations — Global attitudes
# ─────────────────────────────────────────

def load_issp():
    """Load ISSP if available. Expects cerebro_data/ISSP_*.csv, ZA4747*.dta, ZA4747*.sav, or .dta."""
    paths = (
        list(OUTPUT_DIR.glob("ISSP*.csv")) + list(OUTPUT_DIR.glob("ISSP*.dta")) + list(OUTPUT_DIR.glob("issp*.csv"))
        + list(OUTPUT_DIR.glob("ZA4747*.dta")) + list(OUTPUT_DIR.glob("ZA4747*.sav"))
    )
    if not paths:
        return None, "No ISSP file. Register at GESIS, download from https://www.gesis.org/en/issp/data-and-documentation/data-cumulations"
    try:
        import pandas as pd
        p = paths[0]
        if p.suffix.lower() == ".dta":
            df = pd.read_stata(p)
        else:
            df = pd.read_csv(p)
        if df.empty or len(df) < 500:
            return None, "ISSP file too small"
        # Key vars: redistribution (EQWLTH), religion (RELIG), trust (TRUST), etc.
        year_col = "YEAR" if "YEAR" in df.columns else "year"
        if year_col in df.columns:
            years = df[year_col].dropna().astype(int).unique()
            return {"n_respondents": len(df), "years": sorted(years)[-5:], "countries": df.get("COUNTRY", pd.Series()).nunique() if "COUNTRY" in df.columns else "?"}, None
        return {"n_respondents": len(df)}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 3. GBCD Corpus — Migration narrative
# ─────────────────────────────────────────

def load_gbcd():
    """Load GBCD from GitHub clone or local copy."""
    gbcd_dir = SCRIPT_DIR / "GBCD"  # Clone target
    if not gbcd_dir.exists():
        # Try cerebro_data
        gbcd_dir = OUTPUT_DIR / "GBCD"
    if not gbcd_dir.exists():
        return None, "Clone: git clone https://github.com/Computational-social-science/GBCD.git"
    try:
        import pandas as pd
        # Prefer brain drain/gain data (5.Geographical heterogeneity)
        pref = list(gbcd_dir.rglob("count_brain_drain.csv")) + list(gbcd_dir.rglob("country_brain_gain.csv"))
        files = pref if pref else list(gbcd_dir.rglob("*.csv")) + list(gbcd_dir.rglob("*.json"))
        if not files:
            return None, "GBCD dir exists but no CSV/JSON found"
        f = files[0]
        if f.suffix == ".csv":
            df = pd.read_csv(f, header=None, nrows=500) if "brain" in f.name.lower() else pd.read_csv(f, nrows=500)
        else:
            import json
            with open(f) as fp:
                data = json.load(fp) if f.suffix == ".json" else []
            return {"entries": len(data) if isinstance(data, list) else 1, "file": f.name}, None
        return {"rows_sampled": len(df), "file": f.name}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 4. NASA Socioeconomic
# ─────────────────────────────────────────

def load_nasa_socio():
    """Load NASA socioeconomic if available (SEDAC, INFORM, LGII Gini)."""
    paths = (list(OUTPUT_DIR.glob("NASA*.csv")) + list(OUTPUT_DIR.glob("INFORM*.csv"))
             + list(OUTPUT_DIR.glob("SEDAC*.csv")) + list(OUTPUT_DIR.glob("*LGII*.csv"))
             + list(OUTPUT_DIR.glob("*social*vulnerability*.csv"))
             + list(OUTPUT_DIR.glob("*vulnerability*.csv")))
    # Also check CIESIN SEDAC folder (common Downloads path)
    if not paths:
        for parent in [Path.home() / "Downloads", SCRIPT_DIR.parent]:
            for d in parent.glob("CIESIN_SEDAC*"):
                paths = list(d.rglob("*.csv"))
                if paths:
                    break
            if paths:
                break
    if not paths:
        return None, "Download from https://www.earthdata.nasa.gov/topics/human-dimensions/socioeconomics/data-access-tools"
    try:
        import pandas as pd
        p = paths[0]
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, nrows=100)
        else:
            df = pd.read_excel(p, nrows=100)
        return {"rows": len(df), "file": p.name}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 5. World Development Indicators (Kaggle)
# ─────────────────────────────────────────

def load_wdi():
    """Load WDI if available. Kaggle: umitka/world-development-indicators"""
    paths = (list(OUTPUT_DIR.glob("WDI*.csv")) + list(OUTPUT_DIR.glob("world_development*.csv"))
             + list(OUTPUT_DIR.glob("wdidata*.csv")) + list(OUTPUT_DIR.glob("*world*development*.csv")))
    if not paths:
        return None, "Download from Kaggle: umitka/world-development-indicators, save to cerebro_data/"
    try:
        import pandas as pd
        df = pd.read_csv(paths[0], nrows=1000)
        cols = list(df.columns)[:10]
        return {"columns_sample": cols, "file": paths[0].name}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 6. UCDP GED (expand)
# ─────────────────────────────────────────

def load_ucdp_ged():
    """Load UCDP GED georeferenced if available."""
    paths = list(OUTPUT_DIR.glob("*ged*.csv")) + list(OUTPUT_DIR.glob("*GED*.csv")) + list(OUTPUT_DIR.glob("ucdp*ged*"))
    if not paths:
        # Fall back to existing UCDP annual
        acd = OUTPUT_DIR / "UCDP_conflict_annual.csv"
        if acd.exists():
            import pandas as pd
            df = pd.read_csv(acd)
            return {"source": "UCDP_ACD_annual", "years": len(df), "expand": "GED at ucdp.uu.se/downloads"}, None
        return None, "UCDP ACD present. Expand: download GED from ucdp.uu.se"
    try:
        import pandas as pd
        df = pd.read_csv(paths[0], nrows=100)
        return {"file": paths[0].name, "has_geolocation": "lat" in str(df.columns).lower() or "lon" in str(df.columns).lower()}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 7. ACLED Full
# ─────────────────────────────────────────

def load_acled():
    """Load ACLED if available."""
    paths = (list(OUTPUT_DIR.glob("ACLED*.csv")) + list(OUTPUT_DIR.glob("acled*.csv"))
             + list(OUTPUT_DIR.glob("*acled*.csv")) + list(OUTPUT_DIR.glob("ACLED*.xlsx")))
    if not paths:
        return None, "Export from acleddata.com/data-export-tool, save to cerebro_data/"
    try:
        import pandas as pd
        p = paths[0]
        df = pd.read_excel(p, nrows=1000) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p, nrows=1000)
        return {"rows": len(df), "file": p.name}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 8. Freedom House
# ─────────────────────────────────────────

def load_freedom_house():
    """Load Freedom House disaggregated if available (CSV or Excel)."""
    paths = (list(OUTPUT_DIR.glob("Freedom*.csv")) + list(OUTPUT_DIR.glob("freedom*.csv"))
             + list(OUTPUT_DIR.glob("FH*.csv")) + list(OUTPUT_DIR.glob("*freedom*house*.csv"))
             + list(OUTPUT_DIR.glob("Freedom*.xlsx")) + list(OUTPUT_DIR.glob("FH*.xlsx"))
             + list(OUTPUT_DIR.glob("FITW*.xlsx")) + list(OUTPUT_DIR.glob("*freedom*house*.xlsx")))
    if not paths:
        return None, "Download from freedomhouse.org/report/freedom-world"
    try:
        import pandas as pd
        p = paths[0]
        df = pd.read_excel(p, nrows=100) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p, nrows=100)
        return {"file": p.name, "rows": len(df)}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# 9. UNHCR Microdata
# ─────────────────────────────────────────

def load_unhcr():
    """Load UNHCR microdata if available (e.g. SENS Nepal)."""
    paths = (list(OUTPUT_DIR.glob("UNHCR*.csv")) + list(OUTPUT_DIR.glob("unhcr*.csv"))
             + list(OUTPUT_DIR.glob("*SENS*.csv")) + list(OUTPUT_DIR.glob("UNHCR*.xlsx")))
    if not paths:
        return None, "Download from microdata.unhcr.org"
    try:
        import pandas as pd
        p = paths[0]
        df = pd.read_excel(p, nrows=100) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p, nrows=100)
        return {"file": p.name, "rows": len(df)}, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

LOADERS = [
    ("glopop_s", load_glopop_s),
    ("issp", load_issp),
    ("gbcd", load_gbcd),
    ("nasa_socio", load_nasa_socio),
    ("wdi", load_wdi),
    ("ucdp_ged", load_ucdp_ged),
    ("acled_full", load_acled),
    ("freedom_house", load_freedom_house),
    ("unhcr_micro", load_unhcr),
]


def main():
    print("=" * 70)
    print("CEREBRO DEEP DATA LOADER")
    print("=" * 70)

    sources_status = {}
    for name, loader in LOADERS:
        meta = DEEP_SOURCES.get(name, {})
        ring = meta.get("ring", "?")
        conf = meta.get("confidence", 0)
        data, err = loader()
        live = data is not None
        sources_status[name] = {
            "live": live,
            "ring": ring,
            "confidence": conf,
            "data": data,
            "error": err,
            "clocks": meta.get("clocks", []),
        }
        status = "✓" if live else "✗"
        print(f"  {status} {name} (Ring {ring}, {conf}%) — {'OK' if live else err}")

    _write_status(sources_status)
    live_count = sum(1 for s in sources_status.values() if s["live"])
    print(f"\n  Deep sources live: {live_count}/{len(LOADERS)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
