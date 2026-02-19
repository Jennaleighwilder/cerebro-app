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
    paths = list(OUTPUT_DIR.glob("ISSP*Role*.csv")) + list(OUTPUT_DIR.glob("ISSP*.csv")) + list(OUTPUT_DIR.glob("issp*.csv"))
    if not paths:
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(paths[0], nrows=5000)
        # EQWLTH, GINCOME, or similar — "agree gov should reduce"
        red_col = next((c for c in df.columns if "eqwlth" in c.lower() or "reduce" in c.lower() or "inequal" in c.lower()), None)
        country_col = next((c for c in df.columns if "country" in c.lower() or "iso" in c.lower() or c in ("V2", "COUNTRY")), None)
        if not red_col or df.empty:
            return {}
        # Score: higher = more redistribution demand (1-5 scale, 5=strongly agree)
        def score_redist(s):
            if pd.isna(s): return 50
            try:
                v = float(s)
                return min(100, max(0, (v / 5) * 100))
            except: return 50
        if country_col:
            out = {}
            for iso, grp in df.groupby(country_col):
                vals = grp[red_col].apply(score_redist)
                out[str(iso)[:3].upper()] = {"redistribution_demand": vals.mean()}
            return out
        return {"_global": {"redistribution_demand": df[red_col].apply(score_redist).mean()}}
    except Exception as e:
        print(f"  ✗ ISSP: {e}")
        return {}


def _load_gbcd_sentiment():
    """Load GBCD migration narrative sentiment by origin country."""
    gbcd_dir = SCRIPT_DIR / "GBCD" or OUTPUT_DIR / "GBCD"
    if not gbcd_dir.exists():
        return {}
    try:
        import pandas as pd
        files = list(gbcd_dir.rglob("*.csv"))
        if not files:
            return {}
        df = pd.read_csv(files[0], nrows=2000)
        # origin/destination, sentiment, volume
        origin_col = next((c for c in df.columns if "origin" in c.lower() or "from" in c.lower() or "country" in c.lower()), None)
        if not origin_col:
            return {}
        # Negativity proxy: if sentiment col exists, else use volume as risk proxy
        sent_col = next((c for c in df.columns if "sentiment" in c.lower() or "tone" in c.lower()), None)
        out = {}
        for iso, grp in df.groupby(origin_col):
            iso3 = str(iso)[:3].upper()
            if sent_col and sent_col in grp.columns:
                neg = 100 - (grp[sent_col].mean() + 1) * 50 if grp[sent_col].mean() else 50
            else:
                neg = min(100, len(grp) / 20)  # volume proxy
            out[iso3] = {"migration_negativity": min(100, max(0, neg))}
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

    # Countries: union of all with data
    all_iso = set(FOCUS_COUNTRIES) | set(wb.keys()) | set(glopop.keys()) | set(issp.keys()) | set(gbcd.keys()) | set(ucdp.keys())
    all_iso.discard("")

    results = []
    for iso in all_iso:
        if iso == "_global":
            continue
        # Get values (use _global as fallback for some)
        gini = wb.get(iso, {}).get("gini") or wb.get("WLD", {}).get("gini")
        gini_year = wb.get(iso, {}).get("year") or wb.get("WLD", {}).get("year")

        glopop_c = glopop.get(iso) or glopop.get("_global", {})
        pol = glopop_c.get("polarization") or glopop_c.get("top10_share", 50)

        issp_c = issp.get(iso) or issp.get("_global", {})
        redist = issp_c.get("redistribution_demand", 50)

        gbcd_c = gbcd.get(iso) or {}
        mig_neg = gbcd_c.get("migration_negativity", 50)

        ucdp_c = ucdp.get(iso) or ucdp.get("_global", {})
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
