#!/usr/bin/env python3
"""
CEREBRO GLOPOP-S AGGREGATOR
===========================
Reads synthpop_*.dat.gz from cerebro_data/GLOPOP-S/, extracts INCOME quintiles by country,
outputs GLOPOP_country_summary.csv for the risk engine.

Run: python cerebro_glopop_aggregator.py
Expects: cerebro_data/GLOPOP-S/*.dat.gz or cerebro_data/GLOPOP_update.zip
"""

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data"
GLOPOP_DIR = OUTPUT_DIR / "GLOPOP-S"
SUMMARY_CSV = OUTPUT_DIR / "GLOPOP_country_summary.csv"

# Binary format: 15 columns per row (from VU-IVM GLOPOP-S)
N_COLUMNS = 15
ATTR_NAMES = ['HID', 'RELATE_HEAD', 'INCOME', 'WEALTH', 'RURAL', 'AGE', 'GENDER', 'EDUC',
              'HHTYPE', 'HHSIZE_CAT', 'AGRI_OWNERSHIP', 'FLOOR', 'WALL', 'ROOF', 'SOURCE']


def _extract_zip_if_needed():
    """Extract GLOPOP_update.zip to GLOPOP-S/ if present."""
    zip_path = OUTPUT_DIR / "GLOPOP_update.zip"
    if not zip_path.exists():
        alt = Path.home() / "Downloads" / "GLOPOP_update.zip"
        if alt.exists():
            import shutil
            shutil.copy(alt, zip_path)
            print(f"  Copied GLOPOP_update.zip from Downloads → cerebro_data/")
        else:
            return
    try:
        GLOPOP_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if name.endswith('.dat.gz'):
                    base = Path(name).name
                    z.extract(name, GLOPOP_DIR)
                    extracted = GLOPOP_DIR / name
                    if extracted != GLOPOP_DIR / base:
                        (GLOPOP_DIR / base).write_bytes(extracted.read_bytes())
                        extracted.unlink(missing_ok=True)
        print(f"  Extracted {zip_path.name} → GLOPOP-S/")
    except Exception as e:
        print(f"  ⚠ GLOPOP_update.zip extract: {e}")


def _read_synthpop_file(path):
    """Read one .dat.gz file, return DataFrame with INCOME column."""
    with gzip.open(path, 'rb') as f:
        binary = f.read()
    data_np = np.frombuffer(binary, dtype=np.int32)
    n_people = data_np.size // N_COLUMNS
    data_reshaped = np.reshape(data_np, (n_people, N_COLUMNS))
    df = pd.DataFrame(data_reshaped, columns=ATTR_NAMES)
    return df


def _country_from_filename(name):
    """Extract ISO3 from synthpop_XXXrNNN.dat.gz -> XXX."""
    # synthpop_MYSr108.dat.gz -> MYS
    stem = Path(name).stem  # synthpop_MYSr108
    if '_' in stem:
        part = stem.split('_')[1]  # MYSr108
        return part[:3].upper() if len(part) >= 3 else None
    return None


def aggregate_glopop():
    """Aggregate .dat.gz files by country, output summary CSV."""
    _extract_zip_if_needed()

    paths = list(GLOPOP_DIR.rglob("*.dat.gz")) if GLOPOP_DIR.exists() else []
    if not paths:
        print("  No GLOPOP-S .dat.gz files found.")
        return

    # Limit to 2 files per country to keep runtime reasonable
    by_country = {}
    for p in paths:
        iso = _country_from_filename(p.name)
        if not iso:
            continue
        if iso not in by_country:
            by_country[iso] = []
        if len(by_country[iso]) < 2:
            by_country[iso].append(p)

    rows = []
    for iso, files in by_country.items():
        incomes = []
        for fp in files:
            try:
                df = _read_synthpop_file(fp)
                inc = df["INCOME"].astype(int)
                inc = inc[inc > 0]  # -1 = unavailable
                if len(inc) > 100:
                    incomes.extend(inc.tolist())
            except Exception as e:
                pass
        if not incomes:
            continue
        arr = np.array(incomes)
        top20 = np.mean(arr >= 5) * 100  # quintile 5 = richest 20%
        pol = min(100, float(np.std(arr)) * 25)  # polarization proxy
        palma = (np.percentile(arr, 80) / (np.percentile(arr, 40) + 1e-6)) if len(arr) > 10 else 0
        rows.append({
            "iso": iso,
            "top10_share": min(100, top20 * 0.5 + 10),  # proxy for top 10%
            "polarization": pol,
            "palma_ratio": palma,
            "n_individuals": len(arr),
        })

    if not rows:
        print("  No country-level GLOPOP data extracted.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_CSV, index=False)
    print(f"  → GLOPOP_country_summary.csv ({len(out)} countries)")


if __name__ == "__main__":
    print("=" * 60)
    print("CEREBRO GLOPOP-S AGGREGATOR")
    print("=" * 60)
    aggregate_glopop()
    print("=" * 60)
