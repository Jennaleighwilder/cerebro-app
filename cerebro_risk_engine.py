#!/usr/bin/env python3
"""
CEREBRO COUNTRY RISK ENGINE — NLP layer for country-level predictions
=====================================================================
Parses country-specific queries, combines deep data (WDI, GLOPOP-S, ISSP, GBCD, UCDP)
into risk algorithm. Output: country_risk_data.json for query engine.

Risk Score = (Gini × 0.3) + (income polarization × 0.25) + (redistribution demand × 0.2)
             + (migration narrative negativity × 0.15) + (UCDP event trend × 0.1)

Run: python cerebro_risk_engine.py
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Top countries by population + geopolitical relevance (ISO3)
FOCUS_COUNTRIES = [
    "USA", "CHN", "IND", "BRA", "RUS", "MEX", "ZAF", "IDN", "TUR", "DEU",
    "GBR", "FRA", "ITA", "JPN", "KOR", "ARG", "COL", "EGY", "NGA", "PAK",
    "IRN", "THA", "UKR", "POL", "ESP", "CAN", "AUS", "SAU", "VEN", "CHL",
]

WB_BASE = "https://api.worldbank.org/v2"
YEARS_START = 2018
YEARS_END = 2023


def _fetch_wb_gini_bulk():
    """Fetch Gini for all countries from World Bank API."""
    try:
        import urllib.request
        import urllib.parse
        url = f"{WB_BASE}/country/all/indicator/SI.POV.GINI?format=json&date={YEARS_START}:{YEARS_END}&per_page=2000"
        req = urllib.request.Request(url, headers={"User-Agent": "Cerebro/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
        if len(data) < 2:
            return {}
        # data[1] = list of {countryiso3code, value, date}
        by_country = {}
        for obs in data[1]:
            c = obs.get("countryiso3code", "")
            if not c or c == "":
                continue
            try:
                val = float(obs["value"])
                yr = int(obs["date"])
                if c not in by_country:
                    by_country[c] = {}
                by_country[c][yr] = val
            except (ValueError, TypeError, KeyError):
                pass
        # Latest year per country
        result = {}
        for c, years in by_country.items():
            if years:
                latest_yr = max(years.keys())
                result[c] = {"gini": years[latest_yr], "year": latest_yr}
        return result
    except Exception as e:
        print(f"  ✗ World Bank Gini: {e}")
        return {}


def _load_glopop_by_country():
    """Load GLOPOP-S, extract income metrics by country if available."""
    paths = list(OUTPUT_DIR.glob("GLOPOP-S*.csv")) + list(OUTPUT_DIR.glob("glopop*.csv"))
    if not paths:
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(paths[0])
        if df.empty or "income" not in df.columns:
            return {}
        country_col = next((c for c in df.columns if "country" in c.lower() or "iso" in c.lower() or c == "ISO3"), None)
        if not country_col:
            # Aggregate single region
            top10_share = df["income"].quantile(0.9) / (df["income"].mean() + 1e-6) * 10 if df["income"].mean() else 0
            palma = (df["income"].quantile(0.9) / (df["income"].quantile(0.4) + 1e-6)) if len(df) > 10 else 0
            return {"_global": {"top10_share": min(100, top10_share * 10), "palma_ratio": palma, "polarization": min(100, (df["income"].std() / (df["income"].mean() + 1e-6)) * 50)}}
        out = {}
        for iso, grp in df.groupby(country_col):
            if grp["income"].mean() and len(grp) > 50:
                top10 = grp["income"].quantile(0.9) / (grp["income"].mean() + 1e-6) * 10
                palma = grp["income"].quantile(0.9) / (grp["income"].quantile(0.4) + 1e-6) if len(grp) > 10 else 0
                pol = min(100, (grp["income"].std() / (grp["income"].mean() + 1e-6)) * 50)
                out[str(iso)[:3].upper()] = {"top10_share": min(100, top10 * 10), "palma_ratio": palma, "polarization": pol}
        return out
    except Exception as e:
        print(f"  ✗ GLOPOP-S: {e}")
        return {}


def _load_issp_redistribution():
    """Load ISSP Role of Government module: 'Should government reduce income differences?'"""
    paths = (
        list(OUTPUT_DIR.glob("ISSP*Role*.csv")) + list(OUTPUT_DIR.glob("ISSP*.csv")) + list(OUTPUT_DIR.glob("issp*.csv"))
        + list(OUTPUT_DIR.glob("ZA4747*.dta")) + list(OUTPUT_DIR.glob("ZA4747*.sav"))
    )
    if not paths:
        return {}
    try:
        import pandas as pd
        p = paths[0]
        if p.suffix.lower() == ".dta":
            df = pd.read_stata(p)
        elif p.suffix.lower() == ".sav":
            try:
                df, _ = __import__("pyreadstat").read_sav(p)
            except Exception:
                df = pd.read_spss(p) if hasattr(pd, "read_spss") else pd.DataFrame()
        else:
            df = pd.read_csv(p, nrows=5000)
        # EQWLTH, GINCOME, or ZA4747 v54 ("gov reduce income differences")
        red_col = next((c for c in df.columns if "eqwlth" in str(c).lower() or "reduce" in str(c).lower() or "inequal" in str(c).lower()), None)
        if not red_col and "v54" in df.columns:
            red_col = "v54"  # ISSP Role of Gov
        country_col = next((c for c in df.columns if "country" in str(c).lower() or "iso" in str(c).lower() or c in ("V2", "COUNTRY")), None)
        if not red_col or df.empty:
            return {}
        # Score: v54 categorical 1=Definitely should, 2=Probably should, 3=Probably not, 4=Definitely not
        def score_redist(s):
            if pd.isna(s): return 50
            try:
                v = int(float(s)) if isinstance(s, (int, float)) or hasattr(s, "item") else 3
                return {1: 90, 2: 70, 3: 50, 4: 25, 5: 50}.get(v, 50)
            except: return 50
        if red_col in df.columns and str(df[red_col].dtype) == "category":
            codes = df[red_col].astype("category").cat.codes
            df["_red"] = codes.map(lambda x: {0: 90, 1: 70, 2: 50, 3: 25, 4: 50, 5: 50}.get(int(x), 50))
        else:
            df["_red"] = df[red_col].apply(score_redist)
        def to_iso(c):
            s = str(c)
            if "-" in s: return s.split("-")[1][:2].upper()
            if "." in s: return s.split(".")[1].strip()[:2].upper()
            return str(c)[:3].upper()
        if country_col:
            out = {}
            for cval, grp in df.groupby(country_col):
                iso = to_iso(cval)
                out[iso] = {"redistribution_demand": float(grp["_red"].mean())}
            return out
        return {"_global": {"redistribution_demand": float(df["_red"].mean())}}
    except Exception as e:
        print(f"  ✗ ISSP: {e}")
        return {}


def _country_to_iso3(name):
    """Map GBCD country name to ISO3. Returns None if not found."""
    if not name or not str(name).strip():
        return None
    name = str(name).strip()
    # Common GBCD name variants
    aliases = {
        "United States of America": "USA", "United States": "USA",
        "United Kingdom": "GBR", "UK": "GBR",
        "South Korea": "KOR", "Republic of Korea": "KOR",
        "North Korea": "PRK", "Democratic People's Republic of Korea": "PRK",
        "Russia": "RUS", "Russian Federation": "RUS",
        "Vietnam": "VNM", "Viet Nam": "VNM",
        "Iran": "IRN", "Iran, Islamic Republic of": "IRN",
        "Republic of Serbia": "SRB", "Serbia": "SRB",
        "Czech Republic": "CZE", "Czechia": "CZE",
        "Burma": "MMR", "Myanmar": "MMR",
        "Democratic Republic of the Congo": "COD", "DR Congo": "COD",
        "Republic of the Congo": "COG", "Congo": "COG",
        "United Republic of Tanzania": "TZA", "Tanzania": "TZA",
        "Ivory Coast": "CIV", "Cote d'Ivoire": "CIV",
        "Swaziland": "SWZ", "eSwatini": "SWZ",
        "Laos": "LAO", "Lao People's Democratic Republic": "LAO",
        "East Timor": "TLS", "Timor-Leste": "TLS",
        "Cape Verde": "CPV", "Cabo Verde": "CPV",
        "Antigua & Barbuda ": "ATG", "Antigua and Barbuda": "ATG",
        "The Bahamas": "BHS", "Bahamas": "BHS",
        "Macedonia": "MKD", "North Macedonia": "MKD",
        "Palestine": "PSE", "State of Palestine": "PSE",
        "Hong Kong": "HKG", "Macau": "MAC", "Macao": "MAC",
        "Taiwan": "TWN", "Republic of China": "TWN",
    }
    if name in aliases:
        return aliases[name]
    try:
        import pycountry
        c = pycountry.countries.search_fuzzy(name)
        if c:
            return c[0].alpha_3
    except Exception:
        pass
    return None


def _load_gbcd_sentiment():
    """Load GBCD brain drain by origin country (5.Geographical heterogeneity data)."""
    gbcd_dir = SCRIPT_DIR / "GBCD"
    if not gbcd_dir.exists():
        gbcd_dir = OUTPUT_DIR / "GBCD"
    if not gbcd_dir.exists():
        return {}
    try:
        import pandas as pd
        # Prefer brain_drain (emigration = risk proxy for origin country)
        drain_path = gbcd_dir / "5.Geographical heterogeneity" / "data" / "count_brain_drain.csv"
        if not drain_path.exists():
            candidates = list(gbcd_dir.rglob("count_brain_drain.csv")) + list(gbcd_dir.rglob("country_brain_gain.csv"))
            drain_path = candidates[0] if candidates else None
        if not drain_path or not drain_path.exists():
            return {}
        df = pd.read_csv(drain_path, header=None, names=["country", "count"])
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
        # Normalize: high brain drain -> high migration_negativity (0-100)
        total = df["count"].sum()
        if total <= 0:
            return {}
        out = {}
        for _, row in df.iterrows():
            iso = _country_to_iso3(row["country"])
            if not iso:
                continue
            # Scale by share of global drain; cap at 100
            share = row["count"] / total * 100
            neg = min(100, share * 2)  # countries with >50% share cap at 100
            out[iso] = {"migration_negativity": round(min(100, max(0, neg)), 1)}
        return out
    except Exception as e:
        print(f"  ✗ GBCD: {e}")
        return {}


def _load_ucdp_trend():
    """Load UCDP GED or annual, compute 5-year event trend by country."""
    # Try GED first (georeferenced by country)
    ged = list(OUTPUT_DIR.glob("*ged*.csv")) + list(OUTPUT_DIR.glob("*GED*.csv"))
    if ged:
        try:
            import pandas as pd
            df = pd.read_csv(ged[0])
            country_col = next((c for c in df.columns if "country" in c.lower() or "gwno" in c.lower() or "iso" in c.lower()), None)
            year_col = next((c for c in df.columns if "year" in c.lower() or "date" in c.lower()), None)
            if country_col and year_col:
                df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
                df = df.dropna(subset=[year_col])
                recent = df[df[year_col] >= YEARS_END - 5]
                by_c = recent.groupby(country_col).size()
                prev = df[(df[year_col] >= YEARS_END - 10) & (df[year_col] < YEARS_END - 5)].groupby(country_col).size()
                out = {}
                for c in by_c.index:
                    curr = by_c.get(c, 0)
                    old = prev.get(c, 0)
                    trend = (curr - old) / (old + 1) * 100  # % change
                    out[str(c)[:3].upper()] = {"ucdp_trend": min(100, max(0, (trend + 50)))}  # normalize 0-100
                return out
        except Exception as e:
            pass
    # Fallback: use global UCDP annual as proxy for all
    ann = OUTPUT_DIR / "UCDP_conflict_annual.csv"
    if ann.exists():
        try:
            import pandas as pd
            df = pd.read_csv(ann)
            if "conflict_count" in df.columns and "year" in df.columns:
                recent = df[df["year"] >= YEARS_END - 5]["conflict_count"].mean()
                old = df[(df["year"] >= YEARS_END - 10) & (df["year"] < YEARS_END - 5)]["conflict_count"].mean()
                trend = (recent - old) / (old + 1) * 100
                return {"_global": {"ucdp_trend": min(100, max(0, (trend + 50)))}}
        except Exception:
            pass
    return {}


def _to_iso3(iso):
    """Normalize to 3-char ISO for consistent lookup. Handles 2-char from ISSP."""
    if not iso or iso == "_global" or len(iso) >= 3:
        return iso
    # ISSP sometimes yields 2-char or name fragments (e.g. JA from Japan)
    m = {"US": "USA", "DE": "DEU", "GB": "GBR", "VE": "VEN", "AR": "ARG", "JP": "JPN", "JA": "JPN",
         "CH": "CHE", "CN": "CHN", "IN": "IND", "FR": "FRA", "IT": "ITA", "RU": "RUS", "BR": "BRA",
         "MX": "MEX", "AU": "AUS", "CA": "CAN", "ES": "ESP", "PL": "POL", "NL": "NLD", "SE": "SWE",
         "NO": "NOR", "KR": "KOR", "TR": "TUR", "ZA": "ZAF", "GR": "GRC", "PT": "PRT", "HU": "HUN",
         "RO": "ROU", "CZ": "CZE", "AT": "AUT", "BE": "BEL", "IE": "IRL", "NZ": "NZL", "SG": "SGP",
         "IL": "ISR", "AE": "ARE", "SA": "SAU", "CL": "CHL", "CO": "COL", "PE": "PER", "EG": "EGY",
         "NG": "NGA", "PK": "PAK", "IR": "IRN", "TH": "THA", "UA": "UKR", "ID": "IDN", "PH": "PHL",
         "VN": "VNM", "MY": "MYS", "BD": "BGD", "RS": "SRB", "BG": "BGR", "HR": "HRV", "SK": "SVK",
         "GE": "GEO", "JE": "JEY", "SW": "SWE", "CR": "CRI", "SP": "ESP", "PO": "POL", "UN": "USA"}
    if iso.upper() in m:
        return m[iso.upper()]
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=iso.upper())
        if c:
            return c.alpha_3
    except Exception:
        pass
    return iso.upper()


def _normalize(val, lo=0, hi=100):
    """Normalize value to 0-100 scale."""
    if val is None:
        return 50
    return min(100, max(0, float(val)))


def compute_risk_scores():
    """Combine all sources into country risk scores."""
    wb = _fetch_wb_gini_bulk()
    glopop = _load_glopop_by_country()
    issp = _load_issp_redistribution()
    gbcd = _load_gbcd_sentiment()
    ucdp = _load_ucdp_trend()

    # Countries: union of all with data; normalize to ISO3 for consistent lookup
    raw = set(FOCUS_COUNTRIES) | set(wb.keys()) | set(glopop.keys()) | set(issp.keys()) | set(gbcd.keys()) | set(ucdp.keys())
    all_iso = set()
    for x in raw:
        if x and x != "_global":
            all_iso.add(_to_iso3(x) if len(x) < 3 else x.upper())

    results = []
    for iso in all_iso:
        if iso == "_global":
            continue
        # Lookup keys: try both 2-char and 3-char (ISSP uses 2-char)
        iso2 = iso[:2] if len(iso) == 3 else iso
        # Get values (use _global as fallback for some)
        gini = (wb.get(iso) or wb.get(iso2) or {}).get("gini") or (wb.get("WLD") or {}).get("gini")
        gini_year = (wb.get(iso) or wb.get(iso2) or {}).get("year") or (wb.get("WLD") or {}).get("year")

        glopop_c = glopop.get(iso) or glopop.get(iso2) or glopop.get("_global", {})
        pol = glopop_c.get("polarization") or glopop_c.get("top10_share", 50)

        issp_c = issp.get(iso) or issp.get(iso2) or issp.get("_global", {})
        redist = issp_c.get("redistribution_demand", 50)

        gbcd_c = gbcd.get(iso) or gbcd.get(iso2) or {}
        mig_neg = gbcd_c.get("migration_negativity", 50)

        ucdp_c = ucdp.get(iso) or ucdp.get(iso2) or ucdp.get("_global", {})
        ucdp_tr = ucdp_c.get("ucdp_trend", 50)

        # Risk formula
        risk = (
            _normalize(gini, 20, 60) * 0.30 +
            _normalize(pol) * 0.25 +
            _normalize(redist) * 0.20 +
            _normalize(mig_neg) * 0.15 +
            _normalize(ucdp_tr) * 0.10
        )
        risk = min(100, max(0, risk))

        # Data completeness
        used = []
        if gini is not None:
            used.append("WDI")
        if glopop_c:
            used.append("GLOPOP-S")
        if issp_c:
            used.append("ISSP")
        if gbcd_c:
            used.append("GBCD")
        if ucdp_c:
            used.append("UCDP")

        completeness = len(used) / 5.0 * 100

        # Probability bucket
        if risk >= 70:
            prob = "high"
        elif risk >= 45:
            prob = "medium"
        else:
            prob = "low"

        # Drivers
        drivers = []
        if gini and gini >= 45:
            drivers.append(f"extreme Gini ({gini:.1f})")
        if redist and redist >= 65:
            drivers.append("high redistribution demand")
        if pol and pol >= 60:
            drivers.append("income polarization")
        if mig_neg and mig_neg >= 65:
            drivers.append("out-migration narrative pressure")
        if ucdp_tr and ucdp_tr >= 65:
            drivers.append("rising political violence")

        results.append({
            "iso": iso,
            "risk_score": round(risk, 1),
            "probability_2030": prob,
            "drivers": drivers[:3] or ["insufficient data"],
            "confidence_pct": round(completeness, 0),
            "data_used": {k: k in used for k in ["WDI", "GLOPOP-S", "ISSP", "GBCD", "UCDP"]},
            "gini": round(gini, 2) if gini else None,
            "gini_year": gini_year,
        })

    # Sort by risk descending
    results.sort(key=lambda x: -x["risk_score"])
    return results[:20]  # Top 20


def main():
    print("=" * 70)
    print("CEREBRO COUNTRY RISK ENGINE")
    print("=" * 70)

    results = compute_risk_scores()
    out = {
        "top_10_unequal": [r for r in results if r.get("gini")][:10],  # By Gini when available
        "top_10_risk": results[:10],
        "algorithm": "Risk = Gini×0.3 + Polarization×0.25 + Redistribution×0.2 + Migration×0.15 + UCDP×0.1",
        "sources_available": {
            "WDI": bool(results and any(r.get("gini") for r in results)),
            "GLOPOP-S": any(r.get("data_used", {}).get("GLOPOP-S") for r in results),
            "ISSP": any(r.get("data_used", {}).get("ISSP") for r in results),
            "GBCD": any(r.get("data_used", {}).get("GBCD") for r in results),
            "UCDP": any(r.get("data_used", {}).get("UCDP") for r in results),
        },
    }

    # Sort unequal by Gini
    with_gini = [r for r in results if r.get("gini")]
    with_gini.sort(key=lambda x: -(x["gini"] or 0))
    out["top_10_unequal"] = with_gini[:10]

    with open(OUTPUT_DIR / "country_risk_data.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → country_risk_data.json ({len(results)} countries)")
    print(f"  Top 5 risk: {', '.join(r['iso'] for r in results[:5])}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
