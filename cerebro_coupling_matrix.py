#!/usr/bin/env python3
"""
CEREBRO COUPLING MATRIX — Cross-dimensional lead-lag between clocks
====================================================================
Computes cross-correlations between harm, sexual, class, evil clocks.
Uses first-differenced (velocity) series. Positive lag = clock i leads clock j.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
CLOCKS = ["harm", "sexual", "class", "evil"]
MAX_LAG = 5  # years

CLOCK_FILES = {
    "harm": SCRIPT_DIR / "cerebro_harm_clock_data.csv",
    "sexual": SCRIPT_DIR / "cerebro_sexual_clock_data.csv",
    "class": SCRIPT_DIR / "cerebro_class_clock_data.csv",
    "evil": SCRIPT_DIR / "cerebro_evil_clock_data.csv",  # may not exist
}


def load_clock_series() -> dict[str, pd.Series]:
    """
    Load position series for all four clocks.
    Returns dict of {clock_name: pd.Series indexed by year}.
    Missing clocks get empty Series.
    """
    result = {}
    for name, path in CLOCK_FILES.items():
        if not path.exists():
            result[name] = pd.Series(dtype=float)
            continue
        try:
            df = pd.read_csv(path)
            if "year" in df.columns:
                pos_col = "clock_position_10pt" if "clock_position_10pt" in df.columns else "position"
                if pos_col not in df.columns:
                    result[name] = pd.Series(dtype=float)
                    continue
                df = df.dropna(subset=[pos_col])
                if len(df) < 15:
                    result[name] = pd.Series(dtype=float)
                    continue
                series = df.set_index("year")[pos_col].astype(float)
                result[name] = series
            else:
                df = pd.read_csv(path, index_col=0)
                pos_col = "clock_position_10pt" if "clock_position_10pt" in df.columns else "position"
                if pos_col not in df.columns:
                    result[name] = pd.Series(dtype=float)
                    continue
                df = df[df[pos_col].notna()].tail(100)
                if len(df) < 15:
                    result[name] = pd.Series(dtype=float)
                    continue
                result[name] = df[pos_col].astype(float)
        except Exception:
            result[name] = pd.Series(dtype=float)
    return result


def compute_lead_lag_matrix(
    clock_series: dict[str, pd.Series],
    max_lag: int = MAX_LAG,
) -> dict:
    """
    For each pair (i, j), compute cross-correlation at lags 1..max_lag.
    Returns coupling_matrix[i][j] = (best_lag, correlation_coefficient)
    where positive lag means clock i leads clock j.
    Uses Pearson correlation on first-differenced series (velocity) not levels.
    """
    matrix = {}
    for i in CLOCKS:
        matrix[i] = {}
        si = clock_series.get(i)
        if si is None or len(si) < 20:
            for j in CLOCKS:
                matrix[i][j] = {"best_lag": 0, "correlation": 0.0}
            continue
        vel_i = si.diff().dropna()
        if len(vel_i) < 15:
            for j in CLOCKS:
                matrix[i][j] = {"best_lag": 0, "correlation": 0.0}
            continue
        for j in CLOCKS:
            if i == j:
                matrix[i][j] = {"best_lag": 0, "correlation": 1.0}
                continue
            sj = clock_series.get(j)
            if sj is None or len(sj) < 20:
                matrix[i][j] = {"best_lag": 0, "correlation": 0.0}
                continue
            vel_j = sj.diff().dropna()
            common_idx = vel_i.index.intersection(vel_j.index)
            if len(common_idx) < 15:
                matrix[i][j] = {"best_lag": 0, "correlation": 0.0}
                continue
            vi = vel_i.reindex(common_idx).dropna()
            vj = vel_j.reindex(common_idx).dropna()
            common = vi.index.intersection(vj.index)
            vi, vj = vi.loc[common], vj.loc[common]
            if len(vi) < 15:
                matrix[i][j] = {"best_lag": 0, "correlation": 0.0}
                continue
            best_lag = 0
            best_corr = 0.0
            for lag in range(1, max_lag + 1):
                if lag >= len(vi) - 1:
                    break
                # i leads j: corr(vel_i[t], vel_j[t+lag])
                vi_lag = vi.iloc[:-lag].values
                vj_lead = vj.iloc[lag:].values
                if len(vi_lag) < 10:
                    continue
                corr = np.corrcoef(vi_lag, vj_lead)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            matrix[i][j] = {"best_lag": best_lag, "correlation": round(float(best_corr), 4)}
    return matrix


def compute_coupling_correction(
    current_state: dict[str, dict],
    coupling_matrix: dict,
    dampening: float = 0.10,
) -> dict[str, float]:
    """
    For each clock dimension d, compute coupling correction:
    correction[d] = dampening * sum over other clocks c of:
        correlation(c->d) * velocity[c] * (1 / best_lag[c->d])

    Returns dict of {clock_name: correction_value} (in years).
    """
    corrections = {c: 0.0 for c in CLOCKS}
    for d in CLOCKS:
        total = 0.0
        for c in CLOCKS:
            if c == d:
                continue
            pair = coupling_matrix.get(c, {}).get(d, {})
            corr = pair.get("correlation", 0.0)
            lag = pair.get("best_lag", 1)
            if lag <= 0:
                lag = 1
            vel_c = current_state.get(c, {}).get("velocity", 0.0)
            if vel_c is None:
                vel_c = 0.0
            total += corr * float(vel_c) * (1.0 / lag)
        corrections[d] = round(dampening * total, 4)
    return corrections


def save_coupling_matrix(
    matrix: dict,
    path: str | Path = None,
) -> Path:
    if path is None:
        path = DATA_DIR / "coupling_matrix.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(matrix, f, indent=2)
    return path


def load_coupling_matrix(
    path: str | Path = None,
) -> dict:
    if path is None:
        path = DATA_DIR / "coupling_matrix.json"
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


SHADOW_LOG_PATH = DATA_DIR / "coupling_shadow_log.jsonl"


def _build_bins_for_ece(episodes: list, hit_key: str = "hit", n_bins: int = 10) -> list:
    """Build calibration bins from episodes. hit_key: 'hit' for base, 'shadow_hit' for shadow."""
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        subset = [e for e in episodes if lo <= e.get("confidence", 0) < hi]
        if not subset:
            bins.append({"conf_mid": round((lo + hi) / 2, 2), "empirical_hit_rate": None, "n": 0})
            continue
        hit_rate = sum(1 for e in subset if e.get(hit_key, False)) / len(subset)
        bins.append({"conf_mid": round((lo + hi) / 2, 2), "empirical_hit_rate": round(hit_rate, 4), "n": len(subset)})
    return bins


def _compute_ece_from_bins(bins: list) -> float:
    """ECE = Σ (n_i/N) * |empirical_i - conf_mid_i|."""
    valid = [(b["conf_mid"], b["empirical_hit_rate"], b["n"]) for b in bins if b.get("empirical_hit_rate") is not None and b.get("n", 0) > 0]
    if not valid:
        return 0.0
    N = sum(n for _, _, n in valid)
    if N <= 0:
        return 0.0
    return float(sum((n / N) * abs(e - c) for c, e, n in valid))


def shadow_run(simulate_coupling: bool = True) -> dict:
    """
    Shadow-run coupling: compute coupling corrections but do not apply them.
    Log to coupling_shadow_log.jsonl. Report shadow_brier, shadow_ece, delta_vs_base.
    If delta is positive coupling helps; if negative it hurts (threshold should be raised to 70).
    """
    from cerebro_calibration import get_calibration_setup
    from cerebro_eval_utils import walkforward_predictions, past_only_pool
    from cerebro_honeycomb import compute_honeycomb_fusion, _load_current_clock_velocities
    from cerebro_forward_simulation import run_forward_simulation

    raw, countries_included, component_weights = get_calibration_setup(score_threshold=1.5)
    if len(raw) < 8:
        return {"error": "Insufficient episodes", "n_episodes": len(raw)}

    # Ensure coupling matrix exists
    series = load_clock_series()
    matrix = compute_lead_lag_matrix(series)
    if not matrix or not any(matrix.get(c, {}).get(d, {}).get("correlation") for c in CLOCKS for d in CLOCKS if c != d):
        save_coupling_matrix(matrix)
    coupling = load_coupling_matrix()

    # Run walkforward with use_coupling=False (base)
    episodes_op = []
    for country in countries_included:
        country_raw = [e for e in raw if e.get("country") == country]
        if len(country_raw) >= 6:
            ep_country = walkforward_predictions(
                country_raw, interval_alpha=0.8, min_train=3,
                use_honeycomb=True, component_weights=component_weights,
                use_coupling=False,
            )
            episodes_op.extend(ep_country)
    if len(episodes_op) < 8:
        raw_us = [e for e in raw if e.get("country") == "US"]
        episodes_op = walkforward_predictions(
            raw_us, interval_alpha=0.8, min_train=3,
            use_honeycomb=True, component_weights=component_weights,
            use_coupling=False,
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SHADOW_LOG_PATH
    with open(log_path, "w") as f:
        pass  # truncate

    records = []
    for ep in episodes_op:
        country = ep.get("country", "US")
        saddle_year = ep.get("saddle_year", 0)
        event_year = ep.get("event_year", saddle_year + 5)
        base_peak = ep.get("pred_peak_year", saddle_year + 5)
        base_ws = ep.get("pred_window_start", base_peak - 2)
        base_we = ep.get("pred_window_end", base_peak + 2)
        confidence = ep.get("confidence", 0.7)
        base_hit = ep.get("hit", False)

        coupling_correction = 0.0
        if simulate_coupling and coupling:
            try:
                current_state = _load_current_clock_velocities(
                    saddle_year,
                    ep.get("velocity", 0),
                    ep.get("position", 0),
                    ep.get("acceleration", 0),
                )
                corrections = compute_coupling_correction(current_state, coupling, dampening=0.10)
                coupling_correction = corrections.get("harm", 0.0)
            except Exception:
                pass

        shadow_peak = base_peak + coupling_correction
        shadow_ws = base_ws + int(round(coupling_correction))
        shadow_we = base_we + int(round(coupling_correction))
        shadow_hit = shadow_ws <= event_year <= shadow_we

        base_error = abs(base_peak - event_year)
        shadow_error = abs(shadow_peak - event_year)

        rec = {
            "episode": f"{country}_{saddle_year}",
            "base_prediction": base_peak,
            "coupling_correction": round(coupling_correction, 2),
            "shadow_prediction": round(shadow_peak, 1),
            "actual": event_year,
            "base_error": base_error,
            "shadow_error": round(shadow_error, 2),
            "delta": round(coupling_correction, 2),
        }
        records.append(rec)
        ep["shadow_hit"] = shadow_hit

        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # Base metrics
    base_brier = sum((ep.get("confidence", 0.7) - (1.0 if ep.get("hit") else 0.0)) ** 2 for ep in episodes_op) / len(episodes_op)
    base_bins = _build_bins_for_ece(episodes_op, hit_key="hit")
    base_ece = _compute_ece_from_bins(base_bins)

    # Shadow metrics
    shadow_brier = sum((ep.get("confidence", 0.7) - (1.0 if ep.get("shadow_hit") else 0.0)) ** 2 for ep in episodes_op) / len(episodes_op)
    shadow_bins = _build_bins_for_ece(episodes_op, hit_key="shadow_hit")
    shadow_ece = _compute_ece_from_bins(shadow_bins)

    delta_brier = shadow_brier - base_brier
    delta_ece = shadow_ece - base_ece
    mean_base_error = sum(r["base_error"] for r in records) / len(records)
    mean_shadow_error = sum(r["shadow_error"] for r in records) / len(records)
    delta_vs_base = mean_shadow_error - mean_base_error

    result = {
        "n_episodes": len(episodes_op),
        "base_brier": round(base_brier, 4),
        "shadow_brier": round(shadow_brier, 4),
        "base_ece": round(base_ece, 4),
        "shadow_ece": round(shadow_ece, 4),
        "delta_brier": round(delta_brier, 4),
        "delta_ece": round(delta_ece, 4),
        "delta_vs_base": round(delta_vs_base, 4),
        "mean_base_error": round(mean_base_error, 2),
        "mean_shadow_error": round(mean_shadow_error, 2),
        "log_path": str(log_path),
    }
    return result


def main():
    series = load_clock_series()
    available = [c for c in CLOCKS if len(series.get(c, pd.Series())) >= 15]
    print(f"Loaded clocks: {available}")
    matrix = compute_lead_lag_matrix(series)
    save_coupling_matrix(matrix)
    print(f"Saved coupling matrix → cerebro_data/coupling_matrix.json")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "shadow":
        r = shadow_run(simulate_coupling=True)
        if "error" in r:
            print(f"Shadow run error: {r['error']}")
            sys.exit(1)
        print(f"Shadow run: n={r['n_episodes']}")
        print(f"  base_brier={r['base_brier']}, shadow_brier={r['shadow_brier']}, delta_brier={r['delta_brier']}")
        print(f"  base_ece={r['base_ece']}, shadow_ece={r['shadow_ece']}, delta_ece={r['delta_ece']}")
        print(f"  delta_vs_base={r['delta_vs_base']} (negative=coupling helps)")
        print(f"  → {r['log_path']}")
        sys.exit(0)
    sys.exit(main())
