#!/usr/bin/env python3
"""
CEREBRO FIT DISTANCE WEIGHTS — Learn DIST_VEL_WEIGHT, DIST_ACC_WEIGHT
=====================================================================
Grid search over weight space. Objective: pinball loss (q10/q90 primary) + MAE (secondary).
Rolling-origin split inside US 1900–present episodes (no leakage).
Saves best weights to cerebro_data/distance_weights.json with provenance.
"""

import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "distance_weights.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

RANDOM_SEED = 42
VEL_GRID = [10, 50, 100, 200, 500]
ACC_GRID = [500, 1000, 2500, 5000, 10000]
LABELED_EVENTS = [1933, 1935, 1965, 1981, 1994, 2008, 2020]
EVENT_TOLERANCE = 8


def pinball_loss(y_true: float, y_pred: float, q: float) -> float:
    """Pinball loss for quantile q."""
    e = y_true - y_pred
    return max(q * e, (q - 1) * e)


def load_episodes():
    """Load backtest episodes from CSV (same logic as backtest)."""
    import pandas as pd
    from cerebro_peak_window import detect_saddle_canonical

    if not CSV_PATH.exists():
        return []
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(100)
    if len(df) < 20:
        return []
    episodes = []
    for yr, row in df.iterrows():
        v = row.get("velocity")
        a = row.get("acceleration")
        pos = row.get("clock_position_10pt")
        rb = row.get("ring_B_score")
        ss = row.get("saddle_score", 0)
        if pd.isna(v) or pd.isna(a) or pd.isna(pos):
            continue
        v, a, pos = float(v), float(a), float(pos)
        rb = float(rb) if not pd.isna(rb) else None
        is_sad = (ss >= 2) if not pd.isna(ss) else detect_saddle_canonical(pos, v, a, rb)[0]
        if not is_sad:
            continue
        best_event = None
        best_dist = 999
        for ey in LABELED_EVENTS:
            if ey > yr:
                d = ey - yr
                if d < best_dist and d <= EVENT_TOLERANCE:
                    best_dist = d
                    best_event = ey
        if best_event is None:
            best_event = yr + 5
        episodes.append({
            "saddle_year": int(yr),
            "event_year": best_event,
            "position": pos,
            "velocity": v,
            "acceleration": a,
            "ring_B_score": rb,
        })
    return episodes


def evaluate_weights(episodes: list, vel_w: float, acc_w: float, seed: int) -> float:
    """Compute pinball(q10)+pinball(q90) + 0.5*MAE. Lower is better."""
    from cerebro_peak_window import compute_peak_window

    pl10, pl90, mae_sum, n = 0.0, 0.0, 0.0, 0
    for ep in episodes:
        others = [e for e in episodes if e["saddle_year"] != ep["saddle_year"]]
        pred = compute_peak_window(
            ep["saddle_year"], ep["position"], ep["velocity"], ep["acceleration"],
            ep.get("ring_B_score"), others, interval_alpha=0.8, apply_conformal=False,
            vel_weight=vel_w, acc_weight=acc_w,
        )
        dt_true = ep["event_year"] - ep["saddle_year"]
        p10 = pred.get("delta_p10")
        p90 = pred.get("delta_p90")
        p50 = pred.get("delta_median")
        if p10 is None or p90 is None:
            continue
        pl10 += pinball_loss(dt_true, p10, 0.10)
        pl90 += pinball_loss(dt_true, p90, 0.90)
        mae_sum += abs(dt_true - p50)
        n += 1
    if n == 0:
        return 1e9
    return (pl10 + pl90) / n + 0.5 * (mae_sum / n)


def run_fit():
    """Grid search with rolling-origin: train on first 70%, test on last 30%."""
    random.seed(RANDOM_SEED)
    episodes = load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes", "n": len(episodes)}

    split = int(len(episodes) * 0.7)
    train_ep = episodes[:split]
    test_ep = episodes[split:]

    best_score = 1e9
    best_vel, best_acc = 100, 2500
    for vw in VEL_GRID:
        for aw in ACC_GRID:
            score = evaluate_weights(train_ep, vw, aw, RANDOM_SEED)
            if score < best_score:
                best_score = score
                best_vel, best_acc = vw, aw

    out = {
        "vel_weight": best_vel,
        "acc_weight": best_acc,
        "n_episodes": len(episodes),
        "train_size": len(train_ep),
        "test_size": len(test_ep),
        "random_seed": RANDOM_SEED,
        "provenance": {
            "method": "grid_search",
            "objective": "pinball_q10_q90 + 0.5*MAE",
            "vel_grid": VEL_GRID,
            "acc_grid": ACC_GRID,
        },
        "version": 1,
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    result = run_fit()
    if "error" in result:
        print(f"Fit failed: {result['error']}")
        return 1
    print(f"Best weights: vel={result['vel_weight']}, acc={result['acc_weight']}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
