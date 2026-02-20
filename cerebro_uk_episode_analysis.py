#!/usr/bin/env python3
"""
UK EPISODE ANALYSIS — Why 4 candidates, one short of contributing?
===================================================================
Checks: era clustering (1974-2024), score distribution, near-saddle opportunities.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_analysis() -> dict:
    from cerebro_calibration import (
        _load_oecd_clocks,
        _build_oecd_episodes,
        OECD_EVENT_YEARS,
        EVENT_TOLERANCE,
    )

    clocks = _load_oecd_clocks()
    if "UK" not in clocks:
        return {"error": "UK clock not found"}

    # Build with UK hybrid rule (score>=1 AND near_saddle with relaxed |v|<0.55)
    raw, diag = _build_oecd_episodes("UK", clocks["UK"], OECD_EVENT_YEARS["UK"], score_threshold=1.5)
    event_years = OECD_EVENT_YEARS["UK"]
    max_event = max(event_years)

    # Per-year analysis: which years qualify and why
    import pandas as pd
    df = pd.read_csv(SCRIPT_DIR / "cerebro_data" / "oecd" / "UK_clock.csv")
    df = df[df["year"] < max_event]  # exclude yr >= max_event

    candidates = []
    near_saddle_thresh = 0.20
    for _, row in df.iterrows():
        yr = int(row["year"])
        v = row.get("velocity")
        a = row.get("acceleration")
        score = row.get("saddle_score_phase1")
        if pd.isna(v) or pd.isna(a):
            continue
        v, a = float(v), float(a)
        score = float(score) if not pd.isna(score) else 0

        qualifies_score = score >= 1.3
        opposes = (v > 0 and a < 0) or (v < 0 and a > 0)
        near_saddle = abs(v) < near_saddle_thresh and opposes
        qualifies = qualifies_score or near_saddle

        # Event matching
        best_event = None
        best_d = 999
        for ey in event_years:
            if ey > yr and ey - yr <= EVENT_TOLERANCE and ey - yr < best_d:
                best_d = ey - yr
                best_event = ey
        if best_event is None:
            best_event = yr + 5

        if qualifies:
            candidates.append({
                "year": yr,
                "score": score,
                "qualifies_score": qualifies_score,
                "near_saddle": near_saddle,
                "velocity": round(v, 4),
                "acceleration": round(a, 4),
                "event_year": best_event,
                "years_to_event": best_event - yr,
            })

    # Era distribution
    eras = {"1974-1989": [], "1990-1999": [], "2000-2009": [], "2010-2024": []}
    for c in candidates:
        y = c["year"]
        if y < 1990:
            eras["1974-1989"].append(c)
        elif y < 2000:
            eras["1990-1999"].append(c)
        elif y < 2010:
            eras["2000-2009"].append(c)
        else:
            eras["2010-2024"].append(c)

    # Years that almost qualify (score 1, or score 2 but excluded)
    almost = []
    for _, row in df.iterrows():
        yr = int(row["year"])
        if yr >= max_event:
            continue
        v = row.get("velocity")
        a = row.get("acceleration")
        score = row.get("saddle_score_phase1")
        if pd.isna(v) or pd.isna(a):
            continue
        v, a = float(v), float(a)
        score = float(score) if not pd.isna(score) else 0
        opposes = (v > 0 and a < 0) or (v < 0 and a > 0)
        near_saddle = abs(v) < near_saddle_thresh and opposes
        if score == 1 and not near_saddle:
            almost.append({"year": yr, "score": 1, "velocity": round(v, 4), "acceleration": round(a, 4), "near_saddle": near_saddle})
        elif score == 2 and yr >= max_event:
            almost.append({"year": yr, "score": 2, "excluded": "yr >= max_event"})

    return {
        "n_candidates": len(candidates),
        "need_for_contribution": 5,
        "gap": 5 - len(candidates),
        "candidates": candidates,
        "era_distribution": {e: len(v) for e, v in eras.items()},
        "era_detail": {e: [(c["year"], c["score"]) for c in v] for e, v in eras.items()},
        "spread": "clustered" if max(len(v) for v in eras.values()) >= 3 else "spread",
        "event_years": event_years,
        "event_tolerance": EVENT_TOLERANCE,
        "max_event_excludes": f"years >= {max_event}",
        "almost_qualify_count": len(almost),
        "almost_qualify_sample": almost[:5],
        "raw_diag": diag,
    }


def main():
    print("UK Episode Analysis")
    print("=" * 50)
    r = run_analysis()
    if "error" in r:
        print(r["error"])
        return 1
    print(f"Candidates: {r['n_candidates']} (need {r['need_for_contribution']} to contribute)")
    print(f"Era distribution: {r['era_distribution']}")
    print(f"Spread: {r['spread']}")
    for era, items in r["era_detail"].items():
        if items:
            print(f"  {era}: {items}")
    print(f"Almost qualify (score=1, no near-saddle): {r['almost_qualify_count']}")
    if r["almost_qualify_sample"]:
        print(f"  Sample: {r['almost_qualify_sample']}")
    out_path = SCRIPT_DIR / "cerebro_data" / "uk_episode_analysis.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
