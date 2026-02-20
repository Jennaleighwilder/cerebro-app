#!/usr/bin/env python3
"""
CHIMERA Synthetic Adversarial Worlds — Stress-test saddle detection and peak window.
Generates controlled worlds where we know ground truth: no regime shift, abrupt shift,
smooth trend, high noise, oscillatory. Runs detect_saddle + compute_peak_window,
records false_positive_rate, confidence_mean, window metrics.
Prevents OECD inflation from fooling Infinity Score.
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import json
import numpy as np
import pandas as pd

SCRIPT_DIR = _SCRIPT_DIR
DATA_DIR = SCRIPT_DIR / "cerebro_data"
OUTPUT_PATH = DATA_DIR / "synthetic_worlds.json"

N_YEARS = 100
YEAR_START = 1925
np.random.seed(42)


def _causal_velocity(s: pd.Series) -> pd.Series:
    return s.diff(1)


def _causal_acceleration(v: pd.Series) -> pd.Series:
    return v.diff(1)


def _gen_linear_trend() -> pd.DataFrame:
    """Linear trend: no regime shift, no saddle."""
    years = np.arange(YEAR_START, YEAR_START + N_YEARS)
    position = 0.02 * (years - years[0]) + np.random.normal(0, 0.05, N_YEARS)
    df = pd.DataFrame({"year": years, "position": position})
    df["velocity"] = _causal_velocity(df["position"])
    df["acceleration"] = _causal_acceleration(df["velocity"])
    return df


def _gen_cyclical() -> pd.DataFrame:
    """Cyclical: oscillatory, no single regime shift."""
    years = np.arange(YEAR_START, YEAR_START + N_YEARS)
    t = (years - years[0]) / 20.0
    position = 3 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, N_YEARS)
    df = pd.DataFrame({"year": years, "position": position})
    df["velocity"] = _causal_velocity(df["position"])
    df["acceleration"] = _causal_acceleration(df["velocity"])
    return df


def _gen_high_noise() -> pd.DataFrame:
    """High-noise white noise: no structure."""
    years = np.arange(YEAR_START, YEAR_START + N_YEARS)
    position = np.cumsum(np.random.normal(0, 0.3, N_YEARS))
    df = pd.DataFrame({"year": years, "position": position})
    df["velocity"] = _causal_velocity(df["position"])
    df["acceleration"] = _causal_acceleration(df["velocity"])
    return df


def _gen_sudden_regime_jump() -> pd.DataFrame:
    """Abrupt shift at year 50 with no prior warning."""
    years = np.arange(YEAR_START, YEAR_START + N_YEARS)
    position = np.zeros(N_YEARS)
    jump_year = YEAR_START + 50
    for i, y in enumerate(years):
        if y < jump_year:
            position[i] = 2.0 + np.random.normal(0, 0.05)
        else:
            position[i] = 5.0 + np.random.normal(0, 0.05)
    df = pd.DataFrame({"year": years, "position": position})
    df["velocity"] = _causal_velocity(df["position"])
    df["acceleration"] = _causal_acceleration(df["velocity"])
    return df


def _gen_near_saddle_illusion() -> pd.DataFrame:
    """Velocity mimics saddle (|v|<0.20, sign opposes) but no real regime shift."""
    years = np.arange(YEAR_START, YEAR_START + N_YEARS)
    # Create a shallow dip that produces saddle-like v,a without real transition
    t = (years - years[0]) / float(N_YEARS)
    position = 3.0 + 0.5 * np.sin(4 * np.pi * t) * np.exp(-((t - 0.5) ** 2) * 20)
    position += np.random.normal(0, 0.08, N_YEARS)
    df = pd.DataFrame({"year": years, "position": position})
    df["velocity"] = _causal_velocity(df["position"])
    df["acceleration"] = _causal_acceleration(df["velocity"])
    return df


def _run_world(name: str, df: pd.DataFrame, has_true_event: bool, event_year: int | None) -> dict:
    """Run saddle detection and peak window on one world. Return metrics (raw + calibrated)."""
    from cerebro_core import detect_saddle_canonical
    from cerebro_peak_window import compute_peak_window

    df = df.dropna(subset=["position", "velocity", "acceleration"], how="all")
    if len(df) < 20:
        return {"error": "insufficient_rows", "n": len(df)}

    saddle_detections = []
    confidence_vals_raw = []
    confidence_vals_cal = []
    window_widths = []

    # Load real episodes for compute_peak_window (or use empty)
    pool = []
    try:
        from cerebro_calibration import _load_episodes as load_ep
        raw, _ = load_ep(score_threshold=2.0)
        pool = [e for e in raw if e.get("saddle_year") and e.get("event_year")]
    except Exception:
        pass

    for i, row in df.iterrows():
        y = int(row["year"])
        pos = float(row["position"])
        vel = float(row["velocity"])
        acc = float(row["acceleration"])
        if pd.isna(pos) or pd.isna(vel) or pd.isna(acc):
            continue
        is_saddle, _ = detect_saddle_canonical(pos, vel, acc, None)
        saddle_detections.append((y, is_saddle))

        if is_saddle:
            past = df[df["year"] <= y]
            pos_ser = past["position"].tolist()
            vel_ser = past["velocity"].tolist()
            acc_ser = past["acceleration"].tolist()
            pred = compute_peak_window(
                y, pos, vel, acc, None,
                analogue_episodes=pool if pool else [],
                interval_alpha=0.8, vel_weight=100, acc_weight=2500,
                position_series=pos_ser, velocity_series=vel_ser, acceleration_series=acc_ser,
            )
            conf_raw = pred.get("confidence_pct_raw", pred.get("confidence_pct", 50)) / 100.0
            conf_cal = pred.get("confidence_pct", 50) / 100.0
            ww = pred.get("window_end", y + 10) - pred.get("window_start", y + 3)
            confidence_vals_raw.append(conf_raw)
            confidence_vals_cal.append(conf_cal)
            window_widths.append(ww)

    n_saddles = sum(1 for _, s in saddle_detections if s)
    n_total = len(saddle_detections)
    false_positive_rate = n_saddles / n_total if n_total else 0.0
    if has_true_event:
        # For regime jump: we want low FP elsewhere; saddles near event are OK
        false_positive_rate = n_saddles / n_total if n_total else 0.0
    # For no-event worlds, every saddle is a false positive
    if not has_true_event:
        false_positive_rate = n_saddles / n_total if n_total else 0.0

    return {
        "false_positive_rate": round(false_positive_rate, 4),
        "n_saddles": n_saddles,
        "n_total": n_total,
        "confidence_mean": round(float(np.mean(confidence_vals_cal)), 4) if confidence_vals_cal else 0.5,
        "confidence_mean_raw": round(float(np.mean(confidence_vals_raw)), 4) if confidence_vals_raw else 0.5,
        "confidence_mean_calibrated": round(float(np.mean(confidence_vals_cal)), 4) if confidence_vals_cal else 0.5,
        "confidence_std": round(float(np.std(confidence_vals_cal)), 4) if len(confidence_vals_cal) > 1 else 0.0,
        "window_width_mean": round(float(np.mean(window_widths)), 2) if window_widths else 7.0,
        "window_width_std": round(float(np.std(window_widths)), 2) if len(window_widths) > 1 else 0.0,
    }


def run_synthetic_worlds() -> dict:
    """Generate all worlds, run detection, write cerebro_data/synthetic_worlds.json."""
    worlds = {
        "linear_trend": (_gen_linear_trend(), False, None),
        "cyclical": (_gen_cyclical(), False, None),
        "high_noise": (_gen_high_noise(), False, None),
        "sudden_regime_jump": (_gen_sudden_regime_jump(), True, YEAR_START + 50),
        "near_saddle_illusion": (_gen_near_saddle_illusion(), False, None),
    }

    results = {}
    for name, (df, has_event, event_year) in worlds.items():
        results[name] = _run_world(name, df, has_event, event_year)
        if "error" not in results[name]:
            r = results[name]
            cal = r.get("confidence_mean_calibrated", r.get("confidence_mean", 0.5))
            print(f"  {name}: fp_rate={r['false_positive_rate']:.2f} conf_cal={cal:.2f}")

    # Aggregate for Infinity Score (use calibrated for penalties)
    no_event = ["linear_trend", "cyclical", "high_noise", "near_saddle_illusion"]
    fp_rates = [results[w]["false_positive_rate"] for w in no_event if "error" not in results[w]]
    fp_agg = round(float(np.mean(fp_rates)), 4) if fp_rates else 0.0
    high_noise = results.get("high_noise", {})
    agg = {
        "worlds": results,
        "false_positive_rate": fp_agg,
        "aggregate": {"false_positive_rate": fp_agg},
        "noise_world_confidence_mean": high_noise.get("confidence_mean_raw", high_noise.get("confidence_mean", 0.5)),
        "noise_world_confidence_mean_calibrated": high_noise.get("confidence_mean_calibrated", high_noise.get("confidence_mean", 0.5)),
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(agg, f, indent=2)
    return agg


def main():
    print("CHIMERA Synthetic Adversarial Worlds")
    print("=" * 45)
    run_synthetic_worlds()
    print(f"  → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
