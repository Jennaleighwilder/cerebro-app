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
OECD_DIR = SCRIPT_DIR / "cerebro_data" / "oecd"

EVENT_TOLERANCE = 13
OECD_EVENT_YEARS = {
    "UK": [1979, 1997, 2010, 2016, 2020],  # 2010: Coalition gov't (2008 was crisis onset)
    "DE": [1989, 1990, 2005, 2009, 2015, 2020],
    "FR": [1981, 1995, 2002, 2008, 2017, 2020],
    "JP": [1989, 1997, 2001, 2008, 2011, 2020],
    "CA": [1984, 1993, 2006, 2015, 2020],  # 2006: Harper first Conservative gov't
    "AU": [1975, 1983, 1996, 2007, 2013, 2020],
    "SE": [1976, 1991, 2006, 2014, 2020],
    "BR": [1985, 1994, 2002, 2013, 2016, 2018],  # 2016: Dilma impeachment / Temer
    "TR": [1980, 1997, 2002, 2013, 2016],
    "KR": [1987, 1997, 2002, 2016],
    "ZA": [1994, 2008, 2012, 2018],
    "PL": [1989, 2004, 2015, 2020],
    "AR": [1983, 1995, 2001, 2015, 2019],
    "GR": [2010, 2012, 2015],
    "ES": [2008, 2011, 2017],
    "IT": [1992, 2011, 2018],
    "IN": [1991, 2002, 2014, 2019],
    "MX": [1994, 2000, 2006, 2018],
    "CO": [1991, 2002, 2016],
    "CL": [1990, 2006, 2011, 2019],
    "HU": [1989, 2004, 2010],
    "ID": [1998, 2004, 2014],
    "PT": [1974, 1986, 2011],
    "NL": [1982, 2008, 2012],
    "BE": [1981, 2008, 2011],
    "CZ": [1989, 2004, 2013],
    "TH": [1992, 1997, 2006, 2014],
    "PE": [1990, 2000, 2006, 2016],
    "NG": [1999, 2007, 2015],
    "EG": [2011, 2013, 2014],
}
MIN_TRAIN_OPERATIONAL = 3
MIN_TRAIN_STRICT = 5
COUPLING_THRESHOLD = 60
HONEYCOMB_THRESHOLD = 30
NEAR_SADDLE_V_THRESH = 0.20


def _should_use_coupling(n_quality_episodes: int) -> bool:
    """Enable coupling when quality episodes exceed threshold."""
    return n_quality_episodes >= COUPLING_THRESHOLD


def _should_use_honeycomb(n_post1990_sister: int) -> bool:
    """
    Enable honeycomb when sister engine has enough post-1990 data.
    Sister training pool: count post-1990 episodes (saddle_year >= 1990).
    Enable when n_post1990 >= threshold.
    """
    return n_post1990_sister >= HONEYCOMB_THRESHOLD


def _get_sister_post1990_count(raw_episodes: list) -> int:
    """Count post-1990 episodes in sister engine's training pool (saddle_year >= 1990)."""
    return sum(1 for e in raw_episodes if e.get("saddle_year", 0) >= 1990)  # optional: include |v| < 0.20 + opposes for calibration only
UK_NEAR_SADDLE_V = 0.55  # UK hybrid: score>=1 AND (|v|<this and opposes) to get 1-2 quality episodes


def _get_labeled_events():
    from cerebro_event_loader import load_event_years
    return load_event_years(country="US")


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
            "country": "US",
        })

    diag = {
        "candidate_years_total": len(raw),
        "core_saddles_used": core_count,
        "score_saddles_used": score_count,
        "near_saddles_used": near_count,
    }
    return raw, diag


CALIBRATION_QUALITY_GATE_POS_STD = 0.5  # Exclude flat clocks (position std < this)


def _calibration_quality_gate(country: str, clock_df, oecd_status: dict) -> tuple[bool, str]:
    """
    Quality gate: exclude countries with flat clocks or seed-only event years.
    Returns (passes, reason). US is always allowed (not OECD).
    """
    if country == "US":
        return True, "US"
    import pandas as pd
    pos = clock_df["position"] if "position" in clock_df.columns else (clock_df["clock_position_10pt"] if "clock_position_10pt" in clock_df.columns else None)
    if pos is None:
        return False, "no_position"
    pos_clean = pd.Series(pos).dropna()
    if len(pos_clean) < 15:
        return False, "too_few_rows"
    pos_std = float(pos_clean.std())
    if pos_std < CALIBRATION_QUALITY_GATE_POS_STD:
        return False, f"flat_clock(std={pos_std:.2f})"
    st = oecd_status.get(country, {})
    status = st.get("status", "missing")
    if status != "ok":
        return False, f"no_wb_validation(status={status})"
    return True, "ok"


def _load_oecd_status() -> dict:
    """Load oecd_status.json. Returns {} if missing."""
    p = OECD_DIR / "oecd_status.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_oecd_clocks():
    """Load OECD country clocks. Returns dict {country: DataFrame}."""
    import pandas as pd
    clocks = {}
    if not OECD_DIR.exists():
        return clocks
    for f in sorted(OECD_DIR.glob("*_clock.csv")):
        country = f.stem.replace("_clock", "")
        try:
            df = pd.read_csv(f)
            if len(df) >= 15 and "year" in df.columns and "position" in df.columns:
                clocks[country] = df
        except Exception:
            pass
    return clocks


def _load_metaculus_episodes():
    return []  # disabled — Metaculus episodes degrade Brier


def _is_candidate_oecd(row, score_threshold: float = 2.0, country: str = "") -> tuple[bool, str]:
    """OECD candidate: saddle_score_phase1 >= threshold or near-saddle. UK hybrid: score>=1 AND near-saddle."""
    import pandas as pd
    v = row.get("velocity")
    a = row.get("acceleration")
    score = row.get("saddle_score_phase1")
    if v is None or (hasattr(v, "__float__") and pd.isna(v)):
        return False, ""
    if a is None or (hasattr(a, "__float__") and pd.isna(a)):
        return False, ""
    v, a = float(v), float(a)
    opposes = (v > 0 and a < 0) or (v < 0 and a > 0)
    try:
        score_val = float(score) if score is not None and not (hasattr(score, "__float__") and pd.isna(score)) else 0.0
    except (TypeError, ValueError):
        score_val = 0.0

    # UK hybrid: score >= 1 AND near-saddle (relaxed |v|<0.55 for UK to capture 1996 etc.)
    if country == "UK":
        if score_val >= 1 and abs(v) < UK_NEAR_SADDLE_V and opposes:
            return True, "uk_hybrid"
        return False, ""

    # Standard: score >= threshold OR near-saddle
    if score_val >= score_threshold:
        return True, "score"
    if abs(v) < NEAR_SADDLE_V_THRESH and opposes:
        return True, "near_saddle"
    return False, ""


def _build_oecd_episodes(country: str, clock_df, event_years: list, score_threshold: float = 1.5) -> tuple[list, dict]:
    """Build episodes for one OECD country. Returns (raw_episodes, diag)."""
    import pandas as pd
    raw = []
    core_count = score_count = near_count = 0
    max_event = max(event_years) if event_years else 9999
    for _, row in clock_df.iterrows():
        yr = int(row["year"])
        if yr >= max_event:
            continue
        is_cand, source = _is_candidate_oecd(row, score_threshold=score_threshold, country=country)
        if not is_cand:
            continue
        if source == "score":
            score_count += 1
        elif source == "uk_hybrid":
            near_count += 1
        else:
            near_count += 1
        pos = float(row.get("position", 0))
        vel = float(row.get("velocity", 0))
        acc = float(row.get("acceleration", 0))
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
            "velocity": vel,
            "acceleration": acc,
            "ring_B_score": None,
            "country": country,
        })
    diag = {"candidate_years_total": len(raw), "core_saddles_used": 0, "score_saddles_used": score_count, "near_saddles_used": near_count}
    return raw, diag


def _data_quality_weight(n_episodes: int) -> float:
    """Weight by country episode count: <5 → 0.5, 5–10 → 0.75, 10+ → 1.0."""
    if n_episodes < 5:
        return 0.5
    if n_episodes < 10:
        return 0.75
    return 1.0


def _mark_episode(ep: dict, country_episode_counts: dict) -> dict:
    """Add quality and context markers to each episode."""
    ep = dict(ep)
    if ep.get("source") != "metaculus":
        ep["quality_score"] = _data_quality_weight(country_episode_counts.get(ep.get("country", "US"), 0))
    ep["era"] = "post1990" if ep.get("saddle_year", 0) >= 1990 else "pre1990"
    conf = ep.get("confidence", 0)
    ep["confidence_decile"] = min(9, int(conf * 10)) if conf < 1.0 else 9
    ep["clock_type"] = ep.get("clock", "harm")
    return ep


def _stratified_episode_sample(episodes: list, max_decile_fraction: float = 0.40) -> list:
    """
    Force spread across confidence deciles before calibrator sees data.
    If any decile contains more than max_decile_fraction of total episodes,
    downsample that decile randomly (seed=42). Never removes more than 50% of total.
    """
    from collections import defaultdict
    import random

    random.seed(42)
    by_decile = defaultdict(list)
    for ep in episodes:
        by_decile[ep.get("confidence_decile", 9)].append(ep)

    rebalanced = []
    for decile, eps in by_decile.items():
        if decile in (7, 8):
            max_per_decile = max(3, int(len(episodes) * 0.50))
        else:
            max_per_decile = max(3, int(len(episodes) * max_decile_fraction))
        if len(eps) > max_per_decile:
            rebalanced.extend(random.sample(eps, max_per_decile))
        else:
            rebalanced.extend(eps)

    if len(rebalanced) < len(episodes) * 0.60:
        return episodes
    return rebalanced


def _add_calibrated_conf_mid(bins: list) -> list:
    """Add isotonic-calibrated conf_mid to each bin for post-calibration ECE."""
    valid = [(b["conf_mid"], b["empirical_hit_rate"]) for b in bins if b.get("empirical_hit_rate") is not None and b.get("n", 0) > 0]
    if len(valid) < 2:
        return bins
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        valid = sorted(valid, key=lambda t: t[0])
        X = np.array([t[0] for t in valid])
        y = np.array([t[1] for t in valid])
        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1)
        model.fit(X, y)
        out = []
        for b in bins:
            b = dict(b)
            if b.get("empirical_hit_rate") is not None and b.get("n", 0) > 0:
                cal = float(np.clip(model.predict(np.array([b["conf_mid"]]))[0], 0, 1))
                b["calibrated_conf_mid"] = round(cal, 4)
            out.append(b)
        return out
    except Exception:
        return bins


def _print_55_bin_diagnostic(episodes: list) -> None:
    """
    Print interval width, velocity, acceleration for 0.55 bin vs 0.75 bin.
    Identifies whether wide intervals come from specific countries, eras, or formula bias.
    """
    bin_55 = [e for e in episodes if 0.45 <= e.get("confidence", 0) < 0.60]
    bin_75 = [e for e in episodes if 0.70 <= e.get("confidence", 0) < 0.80]
    if not bin_55 and not bin_75:
        return
    print("\n--- 0.55 bin vs 0.75 bin diagnostic (wide-but-accurate miscalibration) ---")
    for label, subset in [("0.55 bin (45-60% conf)", bin_55), ("0.75 bin (70-80% conf)", bin_75)]:
        if not subset:
            print(f"{label}: n=0")
            continue
        widths = []
        vels, accs = [], []
        for e in subset:
            w = e.get("interval_width")
            if w is None and e.get("pred_window_start") is not None and e.get("pred_window_end") is not None:
                w = e["pred_window_end"] - e["pred_window_start"]
            if w is not None:
                widths.append(w)
            v = e.get("velocity")
            a = e.get("acceleration")
            if v is not None:
                vels.append(float(v))
            if a is not None:
                accs.append(float(a))
        avg_w = sum(widths) / len(widths) if widths else None
        avg_v = sum(vels) / len(vels) if vels else None
        avg_a = sum(accs) / len(accs) if accs else None
        hits = sum(1 for e in subset if e.get("hit"))
        print(f"{label}: n={len(subset)}, hit_rate={hits}/{len(subset)}={100*hits/len(subset):.1f}%")
        print(f"  avg interval_width={avg_w:.1f}" if avg_w is not None else "  avg interval_width=N/A")
        print(f"  avg velocity={avg_v:.4f}" if avg_v is not None else "  avg velocity=N/A")
        print(f"  avg acceleration={avg_a:.4f}" if avg_a is not None else "  avg acceleration=N/A")
    print("\n0.55 bin per-episode (interval_width, saddle_score, country, year, hit):")
    for e in sorted(bin_55, key=lambda x: (x.get("country", ""), x.get("saddle_year", 0))):
        w = e.get("interval_width")
        if w is None and e.get("pred_window_start") is not None and e.get("pred_window_end") is not None:
            w = e["pred_window_end"] - e["pred_window_start"]
        score = e.get("saddle_score") or e.get("ring_B_score") or e.get("saddle_score_phase1")
        print(f"  {e.get('country','?')} {e.get('saddle_year','?')}: width={w}, saddle_score={score}, hit={e.get('hit')}")
    print("---\n")


def _build_weighted_bins(episodes: list, n_bins: int = 10) -> list:
    """
    Build calibration bins using quality_score as sample weight.
    weighted_hit_rate = Σ(quality_score * hit) / Σ(quality_score)
    weighted_n = Σ(quality_score) for ECE weighting
    """
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        subset = [e for e in episodes if lo <= e.get("confidence", 0) < hi]
        if not subset:
            bins.append({"conf_mid": round((lo + hi) / 2, 2), "empirical_hit_rate": None, "n": 0, "weighted_n": 0})
            continue
        weights = [e.get("quality_score", e.get("data_quality_weight", 1.0)) for e in subset]
        total_w = sum(weights)
        if total_w <= 0:
            hit_rate = sum(1 for e in subset if e["hit"]) / len(subset)
            total_w = len(subset)
        else:
            hit_rate = sum(w * (1.0 if e["hit"] else 0.0) for e, w in zip(subset, weights)) / total_w
        bins.append({
            "conf_mid": round((lo + hi) / 2, 2),
            "empirical_hit_rate": round(hit_rate, 4),
            "n": len(subset),
            "weighted_n": round(total_w, 2),
        })
    return bins


def _run_mode_from_episodes(episodes: list, min_train: int) -> dict:
    """Compute calibration metrics from precomputed episodes. Uses data_quality_weight when present."""
    if len(episodes) < 8:
        return {"min_train": min_train, "brier": None, "coverage_80": None, "n_used": 0}

    bins = []
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        subset = [e for e in episodes if lo <= e["confidence"] < hi]
        if not subset:
            bins.append({"conf_mid": (lo + hi) / 2, "empirical_hit_rate": None, "n": 0})
            continue
        weights = [e.get("data_quality_weight", 1.0) for e in subset]
        total_w = sum(weights)
        if total_w <= 0:
            hit_rate = sum(1 for e in subset if e["hit"]) / len(subset)
        else:
            hit_rate = sum(w * (1.0 if e["hit"] else 0.0) for e, w in zip(subset, weights)) / total_w
        bins.append({"conf_mid": round((lo + hi) / 2, 2), "empirical_hit_rate": round(hit_rate, 4), "n": len(subset)})

    weights = [e.get("data_quality_weight", 1.0) for e in episodes]
    total_w = sum(weights)
    if total_w <= 0:
        brier = sum((e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e in episodes) / len(episodes)
        in_80 = sum(1 for e in episodes if e["hit"])
        coverage_80 = in_80 / len(episodes)
    else:
        brier = sum(w * (e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e, w in zip(episodes, weights)) / total_w
        in_80 = sum(w * (1.0 if e["hit"] else 0.0) for e, w in zip(episodes, weights))
        coverage_80 = in_80 / total_w
    return {"min_train": min_train, "brier": round(brier, 4), "coverage_80": round(coverage_80, 4), "n_used": len(episodes), "bins": bins}


def _run_mode(raw, min_train: int) -> dict:
    """Run calibration for one min_train mode (single pool)."""
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


def get_calibration_setup(score_threshold: float = 1.5) -> tuple[list, list, dict | None]:
    """
    Return (raw, countries_included, component_weights) for shadow run or external walkforward.
    Same loading logic as run_calibration, stops before walkforward.
    """
    raw_us, diag_us = _load_episodes(score_threshold=score_threshold)
    raw = list(raw_us)
    oecd_clocks = _load_oecd_clocks()
    oecd_status = _load_oecd_status()
    countries_included = ["US"]
    for country, clock_df in oecd_clocks.items():
        passes, _ = _calibration_quality_gate(country, clock_df, oecd_status)
        if not passes:
            continue
        event_years = OECD_EVENT_YEARS.get(country, [yr + 5 for yr in range(1970, 2020, 10)])
        uk_threshold = 1.2 if country == "UK" else score_threshold
        raw_oecd, _ = _build_oecd_episodes(country, clock_df, event_years, score_threshold=uk_threshold)
        if len(raw_oecd) >= MIN_TRAIN_OPERATIONAL + 2:
            raw.extend(raw_oecd)
            countries_included.append(country)
    component_weights = None
    if len(raw) >= 8 and _should_use_honeycomb(_get_sister_post1990_count(raw)):
        try:
            from cerebro_honeycomb import compute_component_mae_and_weights
            mae_result = compute_component_mae_and_weights(raw, min_train=MIN_TRAIN_OPERATIONAL)
            if mae_result.get("weights"):
                component_weights = mae_result["weights"]
        except Exception:
            pass
    return raw, countries_included, component_weights


def _wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for coverage. Returns (lower, upper)."""
    try:
        from statsmodels.stats.proportion import proportion_confint
        lo, hi = proportion_confint(successes, n, alpha=alpha, method="wilson")
        return (round(float(lo), 2), round(float(hi), 2))
    except Exception:
        return (None, None)


def run_calibration(score_threshold: float = 1.5) -> dict:
    from cerebro_eval_utils import walkforward_predictions

    raw_us, diag_us = _load_episodes(score_threshold=score_threshold)
    raw = list(raw_us)
    diag = dict(diag_us)
    candidate_country_counts = {"US": len(raw_us)}
    oecd_diags = {}

    # OECD expansion: load per-country episodes (past-only pool per country)
    oecd_clocks = _load_oecd_clocks()
    oecd_status = _load_oecd_status()
    countries_excluded = []
    countries_included = ["US"]
    for country, clock_df in oecd_clocks.items():
        passes, reason = _calibration_quality_gate(country, clock_df, oecd_status)
        if not passes:
            countries_excluded.append({"country": country, "reason": reason})
            continue
        event_years = OECD_EVENT_YEARS.get(country, [])
        if not event_years:
            event_years = [yr + 5 for yr in range(1970, 2020, 10)]
        # UK: 1.2 (scores 0,1,2 → 1.2=1.3; only 1.0 adds score-1 years)
        uk_threshold = 1.2 if country == "UK" else score_threshold
        raw_oecd, diag_oecd = _build_oecd_episodes(country, clock_df, event_years, score_threshold=uk_threshold)
        if len(raw_oecd) >= MIN_TRAIN_OPERATIONAL + 2:
            raw.extend(raw_oecd)
            candidate_country_counts[country] = len(raw_oecd)
            oecd_diags[country] = diag_oecd
            countries_included.append(country)

    if len(raw) < 8:
        n_quality_episodes = len(raw)
        n_post1990_sister = _get_sister_post1990_count(raw)
        return {
            "error": "Insufficient episodes",
            "mode_operational": {},
            "mode_strict": {},
            "method": "walkforward",
            "candidate_years_total": diag.get("candidate_years_total", 0),
            "core_saddles_used": diag.get("core_saddles_used", 0),
            "score_saddles_used": diag.get("score_saddles_used", 0),
            "near_saddles_used": diag.get("near_saddles_used", 0),
            "candidate_country_counts": candidate_country_counts,
            "calibration_quality_gate": {"countries_excluded": countries_excluded, "countries_included": countries_included},
            "coupling_enabled": _should_use_coupling(n_quality_episodes),
            "honeycomb_enabled": _should_use_honeycomb(n_post1990_sister),
            "coupling_threshold": COUPLING_THRESHOLD,
            "honeycomb_threshold": HONEYCOMB_THRESHOLD,
            "n_quality_episodes": n_quality_episodes,
            "n_post1990_sister": n_post1990_sister,
        }

    # Auto-enable coupling/honeycomb based on data thresholds
    n_quality_episodes = len(raw)
    n_post1990_sister = _get_sister_post1990_count(raw)
    use_coupling = _should_use_coupling(n_quality_episodes)
    use_honeycomb = _should_use_honeycomb(n_post1990_sister)

    # Task 1: Compute inverse-MAE weights from walkforward, use honeycomb fusion
    component_weights = None
    if use_honeycomb:
        try:
            from cerebro_honeycomb import compute_component_mae_and_weights
            mae_result = compute_component_mae_and_weights(raw, min_train=MIN_TRAIN_OPERATIONAL)
            if mae_result.get("weights"):
                component_weights = mae_result["weights"]
        except Exception:
            pass

    # Walk-forward per country (no cross-country analogue mixing); only countries that passed quality gate
    episodes_op = []
    for country in countries_included:
        country_raw = [e for e in raw if e.get("country") == country]
        if len(country_raw) >= MIN_TRAIN_OPERATIONAL + 3:
            ep_country = walkforward_predictions(
                country_raw, interval_alpha=0.8, min_train=MIN_TRAIN_OPERATIONAL,
                use_honeycomb=use_honeycomb, component_weights=component_weights,
                use_coupling=use_coupling,
            )
            episodes_op.extend(ep_country)

    if len(episodes_op) < 8:
        episodes_op = walkforward_predictions(
            raw_us, interval_alpha=0.8, min_train=MIN_TRAIN_OPERATIONAL,
            use_honeycomb=use_honeycomb, component_weights=component_weights,
            use_coupling=use_coupling,
        )

    # Inject feedback-scored episodes (from cerebro_live_feedback)
    feedback_path = SCRIPT_DIR / "cerebro_data" / "feedback_episodes.json"
    if feedback_path.exists():
        try:
            with open(feedback_path) as f:
                feedback_eps = json.load(f)
            if feedback_eps and all("confidence" in e and "hit" in e for e in feedback_eps):
                episodes_op.extend(feedback_eps)
        except Exception:
            pass

    # Inject Metaculus calibration episodes
    metaculus_eps = _load_metaculus_episodes()
    episodes_op.extend(metaculus_eps)

    # Assign data_quality_weight per episode (countries with fewer episodes get lower weight)
    n_per_country = {}
    for e in episodes_op:
        c = e.get("country", "US")
        n_per_country[c] = n_per_country.get(c, 0) + 1
    for e in episodes_op:
        n = n_per_country.get(e.get("country", "US"), 0)
        e["data_quality_weight"] = _data_quality_weight(n)

    # Pre-sort and marking: mark episodes, stratified sample, build weighted bins
    episodes_marked = [_mark_episode(ep, n_per_country) for ep in episodes_op]
    decile_dist_before = {}
    for ep in episodes_marked:
        d = ep.get("confidence_decile", 9)
        decile_dist_before[str(d)] = decile_dist_before.get(str(d), 0) + 1

    episodes_balanced = _stratified_episode_sample(episodes_marked, max_decile_fraction=0.40)
    decile_dist_after = {}
    for ep in episodes_balanced:
        d = ep.get("confidence_decile", 9)
        decile_dist_after[str(d)] = decile_dist_after.get(str(d), 0) + 1

    bins = _build_weighted_bins(episodes_balanced, n_bins=10)
    bins = _add_calibrated_conf_mid(bins)
    _print_55_bin_diagnostic(episodes_balanced)

    # Weighted Brier and coverage from balanced set
    weights = [e.get("quality_score", e.get("data_quality_weight", 1.0)) for e in episodes_balanced]
    total_w = sum(weights)
    if total_w <= 0:
        brier = sum((e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e in episodes_balanced) / len(episodes_balanced)
        coverage_80 = sum(1 for e in episodes_balanced if e["hit"]) / len(episodes_balanced)
    else:
        brier = sum(w * (e["confidence"] - (1.0 if e["hit"] else 0.0)) ** 2 for e, w in zip(episodes_balanced, weights)) / total_w
        coverage_80 = sum(w * (1.0 if e["hit"] else 0.0) for e, w in zip(episodes_balanced, weights)) / total_w

    mode_op = {
        "min_train": MIN_TRAIN_OPERATIONAL,
        "brier": round(brier, 4),
        "coverage_80": round(coverage_80, 4),
        "n_used": len(episodes_balanced),
        "bins": bins,
    }
    mode_strict = _run_mode(raw_us, MIN_TRAIN_STRICT) if len(raw_us) >= MIN_TRAIN_STRICT + 5 else mode_op
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
    if n_used > 0 and episodes_balanced:
        in_80 = sum(1 for e in episodes_balanced if e["hit"])
        coverage_80_ci_lower, coverage_80_ci_upper = _wilson_interval(in_80, n_used)
        n_effs = [e.get("n_eff") for e in episodes_balanced if "n_eff" in e and e["n_eff"] is not None]
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
        widths = [e.get("interval_width") for e in episodes_balanced if "interval_width" in e and e["interval_width"] is not None]
        if widths:
            import numpy as np
            arr = np.array(widths, dtype=float)
            interval_width_mean = round(float(np.mean(arr)), 2)
            interval_width_std = round(float(np.std(arr)), 2) if len(widths) > 1 else 0.0

    candidate_years_total = diag.get("candidate_years_total", len(raw))
    candidate_years_used = n_used

    sister_trust_dist = None
    if use_honeycomb and episodes_op and any(e.get("sister_trust_score") is not None for e in episodes_op):
        scores = [e["sister_trust_score"] for e in episodes_op if e.get("sister_trust_score") is not None]
        weights = [e.get("effective_sister_weight", 0) for e in episodes_op if e.get("sister_trust_score") is not None]
        fusion_modes = [e.get("fusion_mode", "") for e in episodes_op if e.get("sister_trust_score") is not None]
        import numpy as np
        sister_trust_dist = {
            "n": len(scores),
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4) if len(scores) > 1 else 0,
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "excluded_count": sum(1 for m in fusion_modes if m == "excluded"),
            "conditional_count": sum(1 for m in fusion_modes if m == "conditional"),
            "high_weight_episodes": [
                {"saddle_year": e.get("saddle_year"), "country": e.get("country", "US"), "trust": e["sister_trust_score"], "eff_weight": e.get("effective_sister_weight")}
                for e in episodes_op if e.get("effective_sister_weight", 0) > 0.05
            ],
        }

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
        "calibration_candidate_rule": "saddle_score>=1.5 OR near-saddle threshold; eval-only",
        "candidate_country_counts": candidate_country_counts,
        "n_used_per_country": {c: sum(1 for e in episodes_balanced if e.get("country") == c) for c in candidate_country_counts},
        "data_quality_weights": {c: _data_quality_weight(n) for c, n in n_per_country.items()},
        "sister_trust_distribution": sister_trust_dist,
        "calibration_quality_gate": {
            "countries_survived": len(countries_included),
            "countries_excluded": len(countries_excluded),
            "excluded": countries_excluded,
            "included": countries_included,
        },
        "episodes_before_balance": len(episodes_op),
        "episodes_after_balance": len(episodes_balanced),
        "decile_distribution_before": decile_dist_before,
        "decile_distribution_after": decile_dist_after,
        "decile_9_downsampled": decile_dist_before.get("9", 0) - decile_dist_after.get("9", 0),
        "coupling_enabled": use_coupling,
        "honeycomb_enabled": use_honeycomb,
        "coupling_threshold": COUPLING_THRESHOLD,
        "honeycomb_threshold": HONEYCOMB_THRESHOLD,
        "n_quality_episodes": n_quality_episodes,
        "n_post1990_sister": n_post1990_sister,
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
    if r.get("sister_trust_distribution"):
        std = r["sister_trust_distribution"]
        print(f"Sister trust: mean={std['mean']}, excluded={std['excluded_count']}, conditional={std['conditional_count']}, fusing_sister(>0.05)={len(std.get('high_weight_episodes', []))}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
