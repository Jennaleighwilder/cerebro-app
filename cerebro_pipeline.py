#!/usr/bin/env python3
"""
CEREBRO PHASE 2 — REDUNDANT DATA PIPELINES
==========================================
PRIMARY:   Google Trends API (live)
BACKUP 1:  Cached 30-day rolling average (from CSV)
BACKUP 2:  GSS imputation (predict Trends from Ring B when available)
BACKUP 3:  Static embedded fallback

RULES:
- If any source fails, instantly swap to next backup
- Track data_source and backup_active for UI
- Confidence score adjusts by source quality
- Retries: pipeline can be re-run (run_all.sh / 60s frontend retry)
"""

import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "cerebro_data"
CSV_PATH = OUTPUT_DIR / "GoogleTrends_cultural_velocity.csv"
GSS_PATH = OUTPUT_DIR / "GSS_RingB_annual.csv"

# Source quality → confidence
SOURCE_CONFIDENCE = {
    "PRIMARY": 95,
    "BACKUP_1_CACHE": 85,
    "BACKUP_2_GSS": 72,
    "BACKUP_3_STATIC": 60,
}


def _try_primary():
    """PRIMARY: Run trends loader. Returns (data_dict, success)."""
    try:
        from cerebro_trends_loader import fetch_trends, get_latest_velocity, get_class_velocity, get_sexual_velocity
        fetch_trends()
        v = get_latest_velocity()
        cv = get_class_velocity()
        cs = get_sexual_velocity()
        if v:
            return {"cultural_velocity": v, "class_velocity": cv, "sexual_velocity": cs}, True
    except Exception as e:
        print(f"  ✗ PRIMARY (Trends) failed: {e}")
    return None, False


def _try_backup1_cache():
    """BACKUP 1: Use cached CSV, 30-day rolling average."""
    if not CSV_PATH.exists():
        return None, False
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
        if df.empty or "cultural_velocity_smooth" not in df.columns:
            return None, False
        # Use last 30 rows (≈30 months) rolling mean as proxy for "30-day"
        tail = df.tail(30)
        smooth = tail["cultural_velocity_smooth"].mean() if len(tail) > 0 else df["cultural_velocity_smooth"].iloc[-1]
        latest = df.iloc[-1]
        v = {
            "cultural_velocity": round(float(latest.get("cultural_velocity", 0)), 2),
            "cultural_velocity_smooth": round(float(smooth), 2),
            "reform_index": round(float(latest.get("reform", 0)), 1),
            "punitive_index": round(float(latest.get("punitive", 0)), 1),
            "year": int(latest.get("year", 0)),
            "month": int(latest.get("month", 0)),
            "lead_time_months": "3–12",
        }
        cv = _load_class_from_csv()
        cs = _load_sexual_from_csv()
        return {"cultural_velocity": v, "class_velocity": cv, "sexual_velocity": cs}, True
    except Exception as e:
        print(f"  ✗ BACKUP 1 (cache) failed: {e}")
    return None, False


def _load_class_from_csv():
    p = OUTPUT_DIR / "GoogleTrends_class_velocity.csv"
    if not p.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.empty or "velocity_smooth" not in df.columns:
            return None
        latest = df.iloc[-1]
        return {"velocity_smooth": round(float(latest.get("velocity_smooth", 0)), 2), "year": int(latest.get("year", 0)), "month": int(latest.get("month", 0))}
    except Exception:
        return None


def _load_sexual_from_csv():
    p = OUTPUT_DIR / "GoogleTrends_sexual_velocity.csv"
    if not p.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.empty or "velocity_smooth" not in df.columns:
            return None
        latest = df.iloc[-1]
        return {"velocity_smooth": round(float(latest.get("velocity_smooth", 0)), 2), "year": int(latest.get("year", 0)), "month": int(latest.get("month", 0))}
    except Exception:
        return None


def _try_backup2_gss():
    """BACKUP 2: GSS imputation — predict cultural velocity from Ring B composite."""
    if not GSS_PATH.exists():
        return None, False
    try:
        import pandas as pd
        df = pd.read_csv(GSS_PATH, index_col=0)
        if df.empty or len(df) < 5:
            return None, False
        # Ring B normalized -1 to +1. Scale to velocity-like: * 10 as proxy
        cols = [c for c in df.columns if "pct" in c.lower() or "score" in c.lower()]
        if not cols:
            return None, False
        composite = df[cols].mean(axis=1)
        latest = composite.iloc[-1]
        prev = composite.iloc[-2] if len(composite) > 1 else latest
        delta = (latest - prev) * 10  # proxy velocity
        v = {
            "cultural_velocity": round(float(delta), 2),
            "cultural_velocity_smooth": round(float(delta), 2),
            "reform_index": round(float(latest) * 50 + 50, 1),
            "punitive_index": round(50 - float(latest) * 50, 1),
            "year": int(df.index[-1]),
            "month": 6,
            "lead_time_months": "6–18",
        }
        return {"cultural_velocity": v, "class_velocity": None, "sexual_velocity": None}, True
    except Exception as e:
        print(f"  ✗ BACKUP 2 (GSS imputation) failed: {e}")
    return None, False


def _try_backup3_static():
    """BACKUP 3: Static embedded fallback."""
    from datetime import datetime
    v = {
        "cultural_velocity": 5.0,
        "cultural_velocity_smooth": 5.0,
        "reform_index": 35.0,
        "punitive_index": 45.0,
        "year": datetime.now().year,
        "month": datetime.now().month,
        "lead_time_months": "3–12",
    }
    return {"cultural_velocity": v, "class_velocity": {"velocity_smooth": 10.0, "year": datetime.now().year, "month": datetime.now().month}, "sexual_velocity": {"velocity_smooth": -2.0, "year": datetime.now().year, "month": datetime.now().month}}, True


def run_pipeline():
    """
    Run redundant pipeline. Returns (data, source, backup_active, confidence).
    Writes pipeline_status.json for export to read.
    """
    import json
    print("=" * 60)
    print("CEREBRO PHASE 2 — REDUNDANT DATA PIPELINE")
    print("=" * 60)

    status_path = OUTPUT_DIR / "pipeline_status.json"
    l1_data_path = OUTPUT_DIR / "pipeline_l1_data.json"

    def _write_status(source, backup_active, confidence):
        with open(status_path, "w") as f:
            json.dump({"source": source, "backup_active": backup_active, "confidence": confidence, "ts": int(time.time())}, f)

    def _write_l1(data):
        try:
            with open(l1_data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # PRIMARY
    print("\n[1/4] PRIMARY: Google Trends API...")
    data, ok = _try_primary()
    if ok:
        print("  ✓ PRIMARY active")
        _write_status("PRIMARY", False, SOURCE_CONFIDENCE["PRIMARY"])
        _write_l1(data)
        return data, "PRIMARY", False, SOURCE_CONFIDENCE["PRIMARY"]

    # BACKUP 1
    print("\n[2/4] BACKUP 1: Cached 30-day rolling average...")
    data, ok = _try_backup1_cache()
    if ok:
        print("  ✓ BACKUP 1 (cache) active")
        _write_status("BACKUP_1_CACHE", True, SOURCE_CONFIDENCE["BACKUP_1_CACHE"])
        _write_l1(data)
        return data, "BACKUP_1_CACHE", True, SOURCE_CONFIDENCE["BACKUP_1_CACHE"]

    # BACKUP 2
    print("\n[3/4] BACKUP 2: GSS imputation...")
    data, ok = _try_backup2_gss()
    if ok:
        print("  ✓ BACKUP 2 (GSS) active")
        _write_status("BACKUP_2_GSS", True, SOURCE_CONFIDENCE["BACKUP_2_GSS"])
        _write_l1(data)
        return data, "BACKUP_2_GSS", True, SOURCE_CONFIDENCE["BACKUP_2_GSS"]

    # BACKUP 3
    print("\n[4/4] BACKUP 3: Static fallback...")
    data, ok = _try_backup3_static()
    if ok:
        print("  ✓ BACKUP 3 (static) active")
        _write_status("BACKUP_3_STATIC", True, SOURCE_CONFIDENCE["BACKUP_3_STATIC"])
        _write_l1(data)
        return data, "BACKUP_3_STATIC", True, SOURCE_CONFIDENCE["BACKUP_3_STATIC"]

    print("  ✗ All sources failed")
    return None, "NONE", True, 0


def main():
    data, source, backup_active, confidence = run_pipeline()
    if data:
        print(f"\n  Source: {source} | Backup: {backup_active} | Confidence: {confidence}%")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
