#!/usr/bin/env python3
"""
CHIMERA COUPLING — Cross-clock correlation and saddle synchronization
=====================================================================
Harm, class, sexual clocks. Cross-correlation, lead-lag, saddle sync.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_coupling.json"


def _load_clock_csv(name: str) -> list[tuple[int, float, float, float]]:
    """Load clock CSV. Returns [(year, position, velocity, acceleration), ...]."""
    p = SCRIPT_DIR / f"cerebro_{name}_clock_data.csv"
    if not p.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(p, index_col=0)
        df = df[df.index.notna()]
        rows = []
        for yr, row in df.iterrows():
            pos = row.get("clock_position_10pt")
            vel = row.get("velocity")
            acc = row.get("acceleration")
            if pd.isna(pos):
                continue
            vel = float(vel) if not pd.isna(vel) else 0.0
            acc = float(acc) if not pd.isna(acc) else 0.0
            rows.append((int(yr), float(pos), vel, acc))
        return rows
    except Exception:
        return []


def _align_series(series: dict[int, float], years: list[int]) -> np.ndarray:
    """Extract aligned values. Fill missing with nan for correlation."""
    return np.array([series.get(y, np.nan) for y in years])


def run_coupling() -> dict:
    """Compute cross-correlation, lead-lag, saddle synchronization."""
    harm = _load_clock_csv("harm")
    class_ = _load_clock_csv("class")
    sexual = _load_clock_csv("sexual")

    if len(harm) < 10:
        return {"error": "Insufficient harm clock data", "version": 1}

    # Build year-indexed series for position
    harm_pos = {y: p for y, p, v, a in harm}
    class_pos = {y: p for y, p, v, a in class_} if class_ else {}
    sexual_pos = {y: p for y, p, v, a in sexual} if sexual else {}

    years = sorted(set(harm_pos.keys()))
    if len(years) < 10:
        return {"error": "Too few years", "version": 1}

    # Cross-correlation matrix (position only for simplicity)
    clocks = [("harm", harm_pos), ("class", class_pos), ("sexual", sexual_pos)]
    clocks = [(n, s) for n, s in clocks if len(s) >= 5]
    n_c = len(clocks)
    corr_matrix = {}
    lead_lag = {}

    for i, (ni, si) in enumerate(clocks):
        for j, (nj, sj) in enumerate(clocks):
            if i >= j:
                continue
            a = _align_series(si, years)
            b = _align_series(sj, years)
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 5:
                corr_matrix[f"{ni}_{nj}"] = None
                lead_lag[f"{ni}_{nj}"] = None
                continue
            a_clean = a[mask]
            b_clean = b[mask]
            corr = np.corrcoef(a_clean, b_clean)[0, 1] if len(a_clean) > 1 else 0
            corr_matrix[f"{ni}_{nj}"] = round(float(corr), 4) if not np.isnan(corr) else None

            # Lead-lag: max correlation at lags -5 to +5
            best_lag = 0
            best_c = 0
            for lag in range(-5, 6):
                if lag <= 0:
                    a_l = a_clean[:lag] if lag != 0 else a_clean
                    b_l = b_clean[-lag:] if lag != 0 else b_clean
                else:
                    a_l = a_clean[lag:]
                    b_l = b_clean[:-lag]
                if len(a_l) < 5 or len(b_l) < 5:
                    continue
                c = np.corrcoef(a_l, b_l)[0, 1] if len(a_l) > 1 else 0
                if not np.isnan(c) and abs(c) > abs(best_c):
                    best_c = c
                    best_lag = lag
            lead_lag[f"{ni}_{nj}"] = {"lag": best_lag, "corr": round(float(best_c), 4)}

    # Saddle synchronization: harm saddles vs class/sexual
    # Saddle = |v| < 0.2 and sign(a) opposes sign(v) — approximate from harm
    harm_saddles = set()
    for y, p, v, a in harm:
        if abs(v) < 0.25 and ((v > 0 and a < 0) or (v < 0 and a > 0)):
            harm_saddles.add(y)

    class_saddles = set()
    for y, p, v, a in class_:
        if abs(v) < 0.25 and ((v > 0 and a < 0) or (v < 0 and a > 0)):
            class_saddles.add(y)

    sexual_saddles = set()
    for y, p, v, a in sexual:
        if abs(v) < 0.25 and ((v > 0 and a < 0) or (v < 0 and a > 0)):
            sexual_saddles.add(y)

    sync_events = 0
    total_harm = len(harm_saddles)
    for y in harm_saddles:
        if (y in class_saddles or any(abs(y - ys) <= 2 for ys in class_saddles)) or \
           (y in sexual_saddles or any(abs(y - ys) <= 2 for ys in sexual_saddles)):
            sync_events += 1
    saddle_sync_rate = sync_events / total_harm if total_harm > 0 else 0.0

    # Variance inflation (simplified: variance of harm position over time)
    harm_vals = [p for y, p, v, a in harm if p is not None]
    var_inflation = float(np.var(harm_vals)) if len(harm_vals) > 1 else 0.0

    cross_corr_strength = np.nanmean([abs(v) for v in corr_matrix.values() if v is not None]) or 0.0
    systemic_instability = 0.4 * cross_corr_strength + 0.4 * saddle_sync_rate + 0.2 * min(1.0, var_inflation)

    return {
        "version": 1,
        "cross_correlation_matrix": corr_matrix,
        "lead_lag_matrix": lead_lag,
        "saddle_sync_rate": round(saddle_sync_rate, 4),
        "variance_inflation": round(var_inflation, 4),
        "systemic_instability_index": round(float(systemic_instability), 4),
        "harm_saddle_count": total_harm,
        "clocks_loaded": [n for n, _ in clocks],
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_coupling()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera coupling: systemic_instability={r.get('systemic_instability_index')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
