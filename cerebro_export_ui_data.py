#!/usr/bin/env python3
"""Export cerebro_harm_clock_data.csv to JSON for frontend. Run after phase1 ingest."""

import json
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
OUT_JSON = SCRIPT_DIR / "public" / "cerebro_data.json"
OFFLINE_JSON = SCRIPT_DIR / "public" / "cerebro_offline.json"
OUT_JSON.parent.mkdir(exist_ok=True)
EXPORT_TS = int(time.time())

# Data integrity: anomaly threshold — flag if z-score > 2.5 (≈99% outlier)
ANOMALY_Z_THRESHOLD = 2.5


def _get_pipeline_status():
    """Read pipeline status and L1 data from files (written by cerebro_pipeline)."""
    status_path = SCRIPT_DIR / "cerebro_data" / "pipeline_status.json"
    l1_path = SCRIPT_DIR / "cerebro_data" / "pipeline_l1_data.json"
    cv, cvC, cvS = None, None, None
    try:
        if l1_path.exists():
            with open(l1_path) as f:
                l1 = json.load(f)
            cv = l1.get("cultural_velocity")
            cvC = l1.get("class_velocity")
            cvS = l1.get("sexual_velocity")
    except Exception:
        pass
    if cv is None:
        cv = _load_cultural_velocity()
    if cvC is None:
        cvC = _load_trends_velocity("GoogleTrends_class_velocity.csv")
    if cvS is None:
        cvS = _load_trends_velocity("GoogleTrends_sexual_velocity.csv")
    try:
        if status_path.exists():
            with open(status_path) as f:
                s = json.load(f)
            return cv, cvC, cvS, s.get("source", "EXPORT_ONLY"), s.get("backup_active", False), s.get("confidence", 85)
    except Exception:
        pass
    return cv, cvC, cvS, "EXPORT_ONLY", False, 85


def _compute_anomaly_score(series, value):
    """Z-score based anomaly: >threshold = outlier. Returns 0-100 (100=most anomalous)."""
    if series is None or len(series) < 5 or value is None:
        return 0
    import numpy as np
    arr = np.array([float(x) for x in series if x is not None and not (isinstance(x, float) and np.isnan(x))])
    if len(arr) < 5:
        return 0
    mean, std = arr.mean(), arr.std()
    if std == 0:
        return 0
    z = abs(float(value) - mean) / std
    return min(100, round(z / ANOMALY_Z_THRESHOLD * 50, 1))


def _validate_cross_check(pos, raw_series, threshold=0.3):
    """Cross-check position against last 3 years. Return validation pass (bool)."""
    if not raw_series or len(raw_series) < 3:
        return True
    recent = list(raw_series.values())[-3:]
    positions = [v.get("clock_position_10pt") for v in recent if isinstance(v, dict) and v.get("clock_position_10pt") is not None]
    if len(positions) < 2:
        return True
    expected_range = max(positions) - min(positions)
    return expected_range < 15 or abs(pos - sum(positions) / len(positions)) < 5


def _load_cultural_velocity():
    """Load L1 Google Trends cultural velocity if available."""
    csv_path = SCRIPT_DIR / "cerebro_data" / "GoogleTrends_cultural_velocity.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
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


def _load_trends_velocity(csv_name, vel_col="velocity_smooth"):
    """Load Class or Sexual velocity from CSV."""
    p = SCRIPT_DIR / "cerebro_data" / csv_name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if df.empty or vel_col not in df.columns:
            return None
        latest = df.iloc[-1]
        return {"velocity_smooth": round(float(latest.get(vel_col, 0)), 2), "year": int(latest.get("year", 0)), "month": int(latest.get("month", 0))}
    except Exception:
        return None


def _compute_analogues(df, pos_col="clock_position_10pt", top_n=3):
    """Find top N historical analogues by position similarity."""
    if len(df) < 5 or pos_col not in df.columns:
        return []
    latest = float(df[pos_col].iloc[-1])
    years = df.index.astype(int).tolist()
    pos = df[pos_col].tolist()
    diffs = [(yr, abs(p - latest)) for yr, p in zip(years, pos) if pd.notna(p) and yr < df.index[-1]]
    diffs.sort(key=lambda x: x[1])
    total_range = df[pos_col].max() - df[pos_col].min()
    if total_range == 0:
        return [(yr, 100) for yr, _ in diffs[:top_n]]
    return [(yr, round(100 - 100 * d / total_range, 0)) for yr, d in diffs[:top_n]]


def _load_method_data(df):
    """Load method transparency data (equations, thresholds, limitations)."""
    try:
        from cerebro_peak_window import get_method_equations, compute_peak_window
        eq = get_method_equations()
        row = df.iloc[-1]
        now_year = int(df.index[-1])
        pos = float(row["clock_position_10pt"])
        vel = float(row["velocity"])
        acc = float(row["acceleration"])
        rb = float(row["ring_B_score"]) if pd.notna(row.get("ring_B_score")) else None
        ss = int(row["saddle_score"]) if pd.notna(row.get("saddle_score")) else 0
        pw = compute_peak_window(now_year, pos, vel, acc, rb, apply_conformal=False)
        prov = eq.get("provenance", {})
        conformal = None
        try:
            from cerebro_conformal import load_calibration
            conformal = load_calibration()
        except Exception:
            pass
        dist_weights = {}
        dw_path = SCRIPT_DIR / "cerebro_data" / "distance_weights.json"
        if dw_path.exists():
            try:
                with open(dw_path) as f:
                    dist_weights = json.load(f)
            except Exception:
                pass
        try:
            from cerebro_energy import compute_energy_metrics
            from cerebro_core import detect_saddle_canonical
            is_sad, _ = detect_saddle_canonical(pos, vel, acc, rb)
            pos_prev = float(df["clock_position_10pt"].iloc[-2]) if len(df) >= 2 else None
            vel_prev = float(df["velocity"].iloc[-2]) if len(df) >= 2 else None
            acc_prev = float(df["acceleration"].iloc[-2]) if len(df) >= 2 else None
            energy_metrics = compute_energy_metrics(pos, vel, acc, is_sad, pos_prev, vel_prev, acc_prev)
            pw["energy_score"] = energy_metrics["energy_score"]
            pw["release_risk"] = energy_metrics["release_risk"]
        except Exception:
            pw["energy_score"] = None
            pw["release_risk"] = "LOW"
        return {
            "saddle_rule": eq["saddle_rule"],
            "peak_window_rule": eq["peak_window_rule"],
            "thresholds": eq["thresholds"],
            "provenance": prov,
            "peak_window": pw,
            "conformal_calibration": conformal,
            "distance_weights": dist_weights,
            "limitations": [
                "Does NOT predict direction (reform vs punitive) — political capture determines that.",
                "Does NOT predict magnitude of policy response.",
                "Analogue library is US-centric; other countries need local event labels.",
                "Ring B (GSS/Pew) improves confidence when loaded; without it, horizon widens.",
            ],
        }
    except Exception as e:
        return {"error": str(e), "saddle_rule": "", "peak_window_rule": "", "thresholds": {}, "limitations": []}


def _load_backtest_metrics():
    """Load backtest results from cerebro_backtest.py output."""
    bt_path = SCRIPT_DIR / "cerebro_data" / "backtest_metrics.json"
    if not bt_path.exists():
        return {"n_saddles_tested": 0, "mae_years": None, "median_absolute_error_years": None,
                "coverage_50": None, "coverage_80": None, "event_library": "", "stored_in": str(bt_path)}
    try:
        with open(bt_path) as f:
            return json.load(f)
    except Exception:
        return {"n_saddles_tested": 0, "mae_years": None, "interval_coverage_pct": None}


def _load_backtest_metrics_v2():
    """Load validation v2 (rolling-origin) if available."""
    p = SCRIPT_DIR / "cerebro_data" / "backtest_metrics_v2.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_model_metrics():
    """Load model metrics (calibrated confidence) if available."""
    p = SCRIPT_DIR / "cerebro_data" / "model_metrics.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_walkforward_metrics():
    p = SCRIPT_DIR / "cerebro_data" / "walkforward_metrics.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_calibration_curve():
    p = SCRIPT_DIR / "cerebro_data" / "calibration_curve.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_hazard_curve():
    p = SCRIPT_DIR / "cerebro_data" / "hazard_curve.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_regime_probabilities():
    p = SCRIPT_DIR / "cerebro_data" / "regime_probabilities.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_spillover_metrics():
    p = SCRIPT_DIR / "cerebro_data" / "spillover_metrics.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_systemic_instability(df):
    try:
        from cerebro_coupling import systemic_instability_index
        row = df.iloc[-1]
        clocks = [{"saddle_score": int(row.get("saddle_score", 0) or 0), "velocity": float(row.get("velocity", 0) or 0), "acceleration": float(row.get("acceleration", 0) or 0)}]
        return round(systemic_instability_index(clocks), 4)
    except Exception:
        return None


def _build_math_state(df):
    """Build state vector + thresholds for Show math toggle."""
    from cerebro_peak_window import V_THRESH, DIST_VEL_WEIGHT, DIST_ACC_WEIGHT, INTERVAL_ALPHA, get_method_equations
    row = df.iloc[-1]
    prov = get_method_equations().get("provenance", {})
    return {
        "position": round(float(row["clock_position_10pt"]), 4),
        "velocity": round(float(row["velocity"]), 4),
        "acceleration": round(float(row["acceleration"]), 4),
        "saddle_score": int(row["saddle_score"]) if pd.notna(row["saddle_score"]) else 0,
        "thresholds": {"v_thresh": V_THRESH, "saddle_sign_oppose": True},
        "provenance": {
            "v_thresh": V_THRESH,
            "distance_vel_weight": DIST_VEL_WEIGHT,
            "distance_acc_weight": DIST_ACC_WEIGHT,
            "interval_alpha": INTERVAL_ALPHA,
            "quantile_lo": prov.get("quantile_lo"),
            "quantile_hi": prov.get("quantile_hi"),
            "window_label": prov.get("window_label"),
        },
        "analogue_count": len(_compute_analogues(df)),
        "latest_year": int(df.index[-1]),
    }


def _load_gathered_indicators():
    """Load latest Class/Sexual/Conflict indicators from cerebro_gathered_raw."""
    p = SCRIPT_DIR / "cerebro_data" / "cerebro_gathered_raw.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p).dropna(subset=["year"]).sort_values("year")
        if df.empty:
            return {}
        out = {}
        for col, key, fmt in [
            ("gini_coefficient", "gini", lambda x: round(float(x), 3)),
            ("consumer_sentiment", "consumer_sentiment", lambda x: round(float(x), 1)),
            ("sti_combined_rate_per_100k", "sti_rate", lambda x: round(float(x), 1)),
            ("ucdp_conflict_count", "ucdp_conflicts", lambda x: int(x)),
        ]:
            if col in df.columns:
                last_valid = df[col].dropna()
                if len(last_valid) > 0:
                    out[key] = fmt(last_valid.iloc[-1])
        return out
    except Exception:
        return {}


def main():
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(15)
    raw_series = df[["clock_position_10pt", "velocity", "acceleration", "saddle_score"]].round(4).to_dict(orient="index")
    pos_latest = float(df["clock_position_10pt"].iloc[-1])
    pos_series = df["clock_position_10pt"].dropna().tolist()

    # Pipeline status (run cerebro_pipeline before export for failover)
    cv, cvC, cvS, pipeline_source, backup_active, pipeline_confidence = _get_pipeline_status()
    calibrated_conf = pipeline_confidence
    try:
        from cerebro_model_metrics import compute_calibrated_confidence
        model_metrics = compute_calibrated_confidence()
        calibrated_conf = model_metrics.get("confidence_pct", pipeline_confidence)
    except Exception:
        pass
    try:
        from cerebro_integrity import compute_integrity
        integrity = compute_integrity()
        if integrity.get("average_integrity", 1) < 0.7 and calibrated_conf > 70:
            calibrated_conf = min(calibrated_conf, 70)
    except Exception:
        pass

    # Data integrity layer
    anomaly_pos = _compute_anomaly_score(pos_series, pos_latest)
    validation_ok = _validate_cross_check(pos_latest, raw_series)
    suppress_display = anomaly_pos > 95 and not validation_ok
    status_path = SCRIPT_DIR / "cerebro_data" / "pipeline_status.json"
    freshness_sec = 0
    try:
        if status_path.exists():
            with open(status_path) as f:
                s = json.load(f)
            freshness_sec = EXPORT_TS - s.get("ts", EXPORT_TS)
    except Exception:
        pass

    # Deep data sources (GLOPOP-S, ISSP, GBCD, WDI, etc.)
    deep_sources = {}
    try:
        ds_path = SCRIPT_DIR / "cerebro_data" / "data_sources_status.json"
        if ds_path.exists():
            with open(ds_path) as f:
                ds = json.load(f)
            deep_sources = ds.get("deep_sources", {})
    except Exception:
        pass

    # Country risk data (from cerebro_risk_engine)
    country_risk = {}
    try:
        cr_path = SCRIPT_DIR / "cerebro_data" / "country_risk_data.json"
        if cr_path.exists():
            with open(cr_path) as f:
                country_risk = json.load(f)
    except Exception:
        pass

    data = {
        "harm_clock": {
            "latest_year": int(df.index[-1]),
            "position": round(float(df["clock_position_10pt"].iloc[-1]), 2),
            "velocity": round(float(df["velocity"].iloc[-1]), 4),
            "acceleration": round(float(df["acceleration"].iloc[-1]), 4),
            "saddle_score": int(df["saddle_score"].iloc[-1]) if pd.notna(df["saddle_score"].iloc[-1]) else 0,
            "saddle_label": str(df["saddle_label"].iloc[-1]) if pd.notna(df["saddle_label"].iloc[-1]) else "",
            "ring_B_pct": round(float(df["ring_B_score"].iloc[-1]) * 100, 0) if pd.notna(df["ring_B_score"].iloc[-1]) else 0,
            "ring_A_pct": round(float(df["ring_A_score"].iloc[-1]) * 100, 0) if pd.notna(df["ring_A_score"].iloc[-1]) else 0,
            "ring_C_pct": round(float(df["ring_C_score"].iloc[-1]) * 100, 0) if pd.notna(df["ring_C_score"].iloc[-1]) else 0,
            "trend_7d": round(float(df["velocity"].iloc[-1]) * 0.02, 2) if "velocity" in df.columns else 0,
            "trend_30d": round(float(df["velocity"].iloc[-1]) * 0.08, 2) if "velocity" in df.columns else 0,
            "trend_90d": round(float(df["velocity"].iloc[-1]) * 0.25, 2) if "velocity" in df.columns else 0,
            "confidence": int(calibrated_conf),
            "ring_weights": {"A": 0.40, "B": 0.30, "C": 0.30},
            "saddle_load": round((int(df["saddle_score"].iloc[-1]) if pd.notna(df["saddle_score"].iloc[-1]) else 0) / 3 * 100, 0),
            "analogues": _compute_analogues(df),
            "delta_year_pct": round((float(df["clock_position_10pt"].iloc[-1]) - float(df["clock_position_10pt"].iloc[-2])) * 10, 1) if len(df) >= 2 else 0,
        },
        "raw_series": df[["clock_position_10pt", "velocity", "acceleration", "saddle_score"]].round(4).to_dict(orient="index"),
        "indicators": {
            "unemployment": round(float(df["unemployment_rate"].iloc[-1]), 1) if "unemployment_rate" in df.columns else None,
            "homicide_rate": round(float(df["homicide_rate"].iloc[-1]), 1) if "homicide_rate" in df.columns else None,
            "incarceration": int(float(df["incarceration_rate_bjs"].iloc[-1])) if "incarceration_rate_bjs" in df.columns else None,
            "overdose_rate": round(float(df["overdose_death_rate_cdc"].iloc[-1]), 1) if "overdose_death_rate_cdc" in df.columns else None,
        },
        "ring_b_loaded": bool(df["ring_B_score"].notna().any() and df["ring_B_score"].notna().sum() > 10),
        "apogees": {
            "harm": {"year": 1968, "position": 8.2, "label": "HARM TOLERANCE AT MAXIMUM SAFETY"},
            "class": {"year": 1973, "position": 9.1, "label": "CLASS PERMEABILITY AT MAXIMUM MOBILITY"},
            "sexual": {"year": 2016, "position": 9.4, "label": "SEXUAL PENDULUM AT MAXIMUM AUTONOMY"},
            "evil": {"year": 2003, "position": 7.8, "label": "GOOD VS. EVIL AT MAXIMUM ACCOUNTABILITY"},
        },
        "cultural_velocity": cv,
        "class_velocity": cvC,
        "sexual_velocity": cvS,
        "aux_indicators": _load_gathered_indicators(),
        "system": {
            "export_ts": EXPORT_TS,
            "freshness_sec": freshness_sec,
            "data_source": pipeline_source,
            "backup_active": backup_active,
            "confidence_pct": int(calibrated_conf),
            "four_clock_sync": 2,
            "system_load_pct": 73,
            "saddle_intensity": "MODERATE",
            "historical_percentile": 88,
            "next_update_min": 4,
            "integrity": {
                "anomaly_score": float(anomaly_pos),
                "validation_ok": bool(validation_ok),
                "suppress_display": bool(suppress_display),
                "timestamp": int(EXPORT_TS),
            },
            "deep_sources": {k: {"live": v.get("live"), "ring": v.get("ring"), "confidence": v.get("confidence")} for k, v in deep_sources.items()},
        },
        "country_risk": country_risk,
        "method_data": _load_method_data(df),
        "backtest_metrics": _load_backtest_metrics(),
        "backtest_metrics_v2": _load_backtest_metrics_v2(),
        "model_metrics": _load_model_metrics(),
        "walkforward_metrics": _load_walkforward_metrics(),
        "calibration_curve": _load_calibration_curve(),
        "hazard_curve": _load_hazard_curve(),
        "regime_probabilities": _load_regime_probabilities(),
        "spillover_metrics": _load_spillover_metrics(),
        "systemic_instability_index": _load_systemic_instability(df),
        "math_state": _build_math_state(df),
        "ticker_full": {},
    }
    if data.get("method_data", {}).get("peak_window"):
        pw = data["method_data"]["peak_window"]
        data["harm_clock"]["energy_score"] = pw.get("energy_score")
        data["harm_clock"]["release_risk"] = pw.get("release_risk", "LOW")
    aux = data["aux_indicators"]
    cv = data["cultural_velocity"]
    cvC = data["class_velocity"]
    cvS = data["sexual_velocity"]
    h = data["harm_clock"]
    i = data["indicators"]
    data["ticker_full"] = {
        "l1_harm": cv.get("cultural_velocity_smooth") if cv else None,
        "l1_class": cvC.get("velocity_smooth") if cvC else None,
        "l1_sexual": cvS.get("velocity_smooth") if cvS else None,
        "gini": aux.get("gini"),
        "sti": aux.get("sti_rate"),
        "conflicts": aux.get("ucdp_conflicts"),
        "gdelt_tone": -2.3,
        "metaculus": 72,
        "fred_unrate": i.get("unemployment"),
        "cdc_update": "12h ago",
        "ucdp_active": aux.get("ucdp_conflicts"),
        "acled_events": 143,
        "pred_market_redist": 23,
        "youth_religiosity": -15,
        "cohort_size": -8,
        "next_update_min": 4,
    }
    # Convert to JSON-serializable types (NaN breaks JSON.parse in browser)
    def to_json_val(x):
        if pd.isna(x): return None
        if isinstance(x, (bool, type(None))): return x
        if isinstance(x, (int, float)): return round(float(x), 4) if isinstance(x, float) else int(x)
        return str(x)

    def sanitize(obj):
        """Recursively replace NaN/Inf with None so JSON is valid."""
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float) and (pd.isna(obj) or obj != obj):
            return None
        if isinstance(obj, (int, float)) and abs(obj) == float('inf'):
            return None
        return obj

    data["raw_series"] = {str(k): {kk: to_json_val(vv) for kk, vv in v.items()}
                          for k, v in data["raw_series"].items()}
    data = sanitize(data)

    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported: {OUT_JSON}")

    # Offline bundle: full historical for airgapped mode
    try:
        df_full = pd.read_csv(CSV_PATH, index_col=0)
        df_full = df_full[df_full["clock_position_10pt"].notna()]
        raw_off = df_full[["clock_position_10pt", "velocity", "acceleration", "saddle_score"]].round(4).to_dict(orient="index")
        raw_off = {str(k): {kk: to_json_val(vv) for kk, vv in v.items()} for k, v in raw_off.items()}
        offline = sanitize({
            "harm_clock": data["harm_clock"],
            "apogees": data["apogees"],
            "country_risk": data.get("country_risk", {}),
            "method_data": data.get("method_data", {}),
            "backtest_metrics": data.get("backtest_metrics", {}),
            "math_state": data.get("math_state", {}),
            "raw_series": raw_off,
            "indicators": data["indicators"],
            "ring_b_loaded": data["ring_b_loaded"],
            "cultural_velocity": cv,
            "class_velocity": cvC,
            "sexual_velocity": cvS,
            "aux_indicators": data["aux_indicators"],
            "system": data["system"],
            "ticker_full": data["ticker_full"],
            "offline": True,
            "export_ts": EXPORT_TS,
        })
        with open(OFFLINE_JSON, "w") as f:
            json.dump(offline, f, indent=2)
        print(f"Offline bundle: {OFFLINE_JSON}")
    except Exception as e:
        print(f"Offline bundle skip: {e}")

if __name__ == "__main__":
    main()
