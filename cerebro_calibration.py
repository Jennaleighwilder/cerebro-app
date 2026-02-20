#!/usr/bin/env python3
"""
CEREBRO CONFIDENCE CALIBRATION — Walk-forward (past-only analogues)
==================================================================
Bin predictions by confidence decile, compute empirical hit rate.
Each episode's prediction uses only past episodes. No future leakage.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "calibration_curve.json"
CSV_PATH = SCRIPT_DIR / "cerebro_harm_clock_data.csv"

EVENT_TOLERANCE = 10
MIN_TRAIN_OPERATIONAL = 3
MIN_TRAIN_STRICT = 5
NEAR_SADDLE_V_THRESH = 0.20  # optional: include |v| < 0.20 + opposes for calibration only


def _get_labeled_events():
    from cerebro_event_loader import load_event_years
    return load_event_years(country="US", include_global=False)


def _is_candidate_year(row, include_near_saddle: bool = True, score_threshold: float = 2.0) -> tuple[bool, str]:
    """
    Candidate = saddle_score >= score_threshold (primary) OR near-saddle (secondary).
    Returns (is_candidate, source) where source in ("core", "score", "near_saddle").
    """
    from cerebro_core import detect_saddle_canonical

    v = row.get("velocity")
    a = row.get("acceleration")
    pos = row.get("clock_position_10pt")
    rb = row.get("ring_B_score")
    score = row.get("saddle_score")

    import pandas as pd
    if any(x is None or (hasattr(x, "__float__") and pd.isna(x)) for x in [v, a, pos]):
        return False, ""

    v, a = float(v), float(a)
    pos = float(pos)
    rb = float(rb) if rb is not None and not pd.isna(rb) else None

    # Core saddle (production rule)
    is_core, _ = detect_saddle_canonical(pos, v, a, rb)
    if is_core:
        return True, "core"

    # Phase1 saddle_score >= score_threshold (calibration-only expansion)
    if score is not None and not (hasattr(score, "__float__") and pd.isna(score)):
        try:
            sc = float(score)
            if sc >= score_threshold:
                return True, "score"
        except (TypeError, ValueError):
            pass

    # Near-saddle: |v| < 0.20 AND sign(a) opposes sign(v)
    if include_near_saddle:
        opposes = (v > 0 and a < 0) or (v < 0 and a > 0)
        if abs(v) < NEAR_SADDLE_V_THRESH and opposes:
            return True, "near_saddle"

    return False, ""


def _load_episodes(score_threshold: float = 2.0):
    import pandas as pd

    if not CSV_PATH.exists():
        return [], {"candidate_years_total": 0, "core_saddles_used": 0, "score_saddles_used": 0, "near_saddles_used": 0}
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(100)
    if len(df) < 20:
        return [], {"candidate_years_total": 0, "core_saddles_used": 0, "score_saddles_used": 0, "near_saddles_used": 0}

    event_years = _get_labeled_events()
    max_event_year = max(event_years) if event_years else 9999

    raw = []
    core_count = score_count = near_count = 0
    for yr, row in df.iterrows():
        yr = int(yr)
        # Exclude years >= max_event_year (no observable future event)
        if yr >= max_event_year:
            continue
        is_cand, source = _is_candidate_year(row, include_near_saddle=True, score_threshold=score_threshold)
        if not is_cand:
            continue
        if source == "core":
            core_count += 1
        elif source == "score":
            score_count += 1
        else:
            near_count += 1

        v, a, pos = row.get("velocity"), row.get("acceleration"), row.get("clock_position_10pt")
        rb = row.get("ring_B_score")
        v, a, pos = float(v), float(a), float(pos)
        rb = float(rb) if rb is not None and not pd.isna(rb) else None

        best_event = None
        best_d = 999
        for ey in event_years:
            if ey > yr and ey - yr <= EVENT_TOLERANCE and ey - yr < best_d:
                best_d = ey - yr
                best_event = ey
        if best_event is None:
            best_event = yr + 5
        raw.append({
            "saddle_year": yr,
            "event_year": best_event,
            "position": pos,
            "velocity": v,
            "acceleration": a,
            "ring_B_score": rb,
        })

    diag = {
        "candidate_years_total": len(raw),
        "core_saddles_used": core_count,
        "score_saddles_used": score_count,
        "near_saddles_used": near_count,
    }
    return raw, diag


def _run_mode(raw, min_train: int) -> dict:
    """Run calibration for one min_train mode."""
    from cerebro_eval_utils import walkforward_predictions

    episodes = walkforward_predictions(raw, interval_alpha=0.8, min_train=min_train)
    if len(episodes) < 8:
        return {"min_train": min_train, "brier": None, "coverage_80": None, "n_used": 0}

    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        subset = [e for e in episodes if lo <= e["confidence"] < hi]
        if not subset:
            bins.append({"conf_mid": (lo + hi) / 2, "empirical_hit_rate": None, "n": 0})
            continue
        hit_rate = sum(1 for e in subset if e["hit"]) / len(subset)
        bins.append({"conf_mid": round((lo + hi) / 2, 2), "empirical_hit_rate": round(hit_rate, 4), "n": len(subset)})

    brier = sum((e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e in episodes) / len(episodes)
    in_80 = sum(1 for e in episodes if e["hit"])
    coverage_80 = in_80 / len(episodes)
    return {"min_train": min_train, "brier": round(brier, 4), "coverage_80": round(coverage_80, 4), "n_used": len(episodes), "bins": bins}


def _wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for coverage. Returns (lower, upper)."""
    try:
        from statsmodels.stats.proportion import proportion_confint
        lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
        return (round(float(lo), 2), round(float(hi), 2))
    except Exception:
        return (None, None)


def run_calibration(score_threshold: float = 2.0) -> dict:
    from cerebro_eval_utils import walkforward_predictions

    raw, diag = _load_episodes(score_threshold=score_threshold)
    if len(raw) < 8:
        return {
            "error": "Insufficient episodes",
            "mode_operational": {},
            "mode_strict": {},
            "method": "walkforward",
            "candidate_years_total": diag.get("candidate_years_total", 0),
            "core_saddles_used": diag.get("core_saddles_used", 0),
            "score_saddles_used": diag.get("score_saddles_used", 0),
            "near_saddles_used": diag.get("near_saddles_used", 0),
        }

    mode_op = _run_mode(raw, MIN_TRAIN_OPERATIONAL)
    mode_strict = _run_mode(raw, MIN_TRAIN_STRICT)

    # Use operational for primary output (bins, etc.)
    episodes_op = walkforward_predictions(raw, interval_alpha=0.8, min_train=MIN_TRAIN_OPERATIONAL)
    bins = mode_op.get("bins", [])
    brier = mode_op.get("brier")
    coverage_80 = mode_op.get("coverage_80")
    n_used = mode_op.get("n_used", 0)

    # Wilson interval for coverage (do not claim 100%)
    coverage_80_ci_lower, coverage_80_ci_upper = None, None
    mean_n_eff = None
    n_eff_interpretation = None
    interval_width_mean = None
    interval_width_std = None
    if n_used > 0 and episodes_op:
        in_80 = sum(1 for e in episodes_op if e["hit"])
        coverage_80_ci_lower, coverage_80_ci_upper = _wilson_interval(in_80, n_used)
        n_effs = [e.get("n_eff") for e in episodes_op if "n_eff" in e and e["n_eff"] is not None]
        mean_n_eff = round(sum(n_effs) / len(n_effs), 2) if n_effs else None
        # n_eff interpretation: < 3 tiny, 3–7 fragile, 7–10 moderate, > 10 stable (core quality signal)
        if mean_n_eff is not None:
            if mean_n_eff < 3:
                n_eff_interpretation = "tiny"
            elif mean_n_eff < 7:
                n_eff_interpretation = "fragile"
            elif mean_n_eff < 10:
                n_eff_interpretation = "moderate"
            else:
                n_eff_interpretation = "stable"
        # Interval width: low coverage + wide intervals = misspecification; narrow = overconfidence
        widths = [e.get("interval_width") for e in episodes_op if "interval_width" in e and e["interval_width"] is not None]
        if widths:
            import numpy as np
            arr = np.array(widths, dtype=float)
            interval_width_mean = round(float(np.mean(arr)), 2)
            interval_width_std = round(float(np.std(arr)), 2) if len(widths) > 1 else 0.0

    candidate_years_total = diag.get("candidate_years_total", len(raw))
    candidate_years_used = n_used

    return {
        "mode_operational": {
            "min_train": MIN_TRAIN_OPERATIONAL,
            "brier": brier,
            "coverage_80": coverage_80,
            "n_used": n_used,
        },
        "mode_strict": {
            "min_train": MIN_TRAIN_STRICT,
            "brier": mode_strict.get("brier"),
            "coverage_80": mode_strict.get("coverage_80"),
            "n_used": mode_strict.get("n_used", 0),
        },
        "bins": bins,
        "brier_score": brier,
        "coverage_80": coverage_80,
        "coverage_80_ci_lower": coverage_80_ci_lower,
        "coverage_80_ci_upper": coverage_80_ci_upper,
        "interval_width_mean": interval_width_mean,
        "interval_width_std": interval_width_std,
        "mean_n_eff": mean_n_eff,
        "n_eff_interpretation": n_eff_interpretation,
        "n_episodes": n_used,
        "n_used": n_used,
        "min_train": MIN_TRAIN_OPERATIONAL,
        "method": "walkforward",
        "candidate_years_total": candidate_years_total,
        "candidate_years_used": candidate_years_used,
        "core_saddles_used": diag.get("core_saddles_used", 0),
        "score_saddles_used": diag.get("score_saddles_used", 0),
        "near_saddles_used": diag.get("near_saddles_used", 0),
        "eval_only_candidate_expansion": True,
        "production_saddle_rule": "detect_saddle_canonical(|v|<0.15 AND sign(a) opposes sign(v))",
        "calibration_candidate_rule": "saddle_score>=2 OR near-saddle threshold; eval-only",
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_calibration()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    # Write backtest_metrics.json with walk-forward labels (no silent overwrite)
    bt_path = SCRIPT_DIR / "cerebro_data" / "backtest_metrics.json"
    if "error" not in r and r.get("method") == "walkforward":
        mo = r.get("mode_operational", {})
        bt = {
            "brier_walkforward": r.get("brier_score"),
            "coverage_80_walkforward": r.get("coverage_80"),
            "coverage_80_ci_lower": r.get("coverage_80_ci_lower"),
            "coverage_80_ci_upper": r.get("coverage_80_ci_upper"),
            "interval_width_mean": r.get("interval_width_mean"),
            "interval_width_std": r.get("interval_width_std"),
            "mean_n_eff": r.get("mean_n_eff"),
            "n_eff_interpretation": r.get("n_eff_interpretation"),
            "n_used": r.get("n_used"),
            "min_train": r.get("min_train"),
            "candidate_years_total": r.get("candidate_years_total"),
            "candidate_years_used": r.get("candidate_years_used"),
            "core_saddles_used": r.get("core_saddles_used"),
            "score_saddles_used": r.get("score_saddles_used"),
            "near_saddles_used": r.get("near_saddles_used"),
            "mode_operational": mo,
            "mode_strict": r.get("mode_strict", {}),
            "method": "walkforward",
            "stored_in": str(bt_path),
        }
        with open(bt_path, "w") as f:
            json.dump(bt, f, indent=2)
    print(f"Calibration: Brier={r.get('brier_score')}, bins={len(r.get('bins', []))}, method={r.get('method')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
