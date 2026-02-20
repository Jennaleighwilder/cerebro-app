#!/usr/bin/env python3
"""
DE (Germany) EPISODE ANALYSIS — Why only 2 episodes?
====================================================
Diagnostic: WorldBank data quality (DEU starts 1990) vs physics (few saddle candidates).
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
        NEAR_SADDLE_V_THRESH,
    )

    clocks = _load_oecd_clocks()
    if "DE" not in clocks:
        return {"error": "DE clock not found"}

    raw, diag = _build_oecd_episodes("DE", clocks["DE"], OECD_EVENT_YEARS["DE"], score_threshold=1.5)
    event_years = OECD_EVENT_YEARS["DE"]
    max_event = max(event_years)

    import pandas as pd
    df = pd.read_csv(SCRIPT_DIR / "cerebro_data" / "oecd" / "DE_clock.csv")
    df = df[df["year"] < max_event]

    # WorldBank data coverage
    wb_path = SCRIPT_DIR / "cerebro_data" / "WorldBank_OECD_DEU.csv"
    wb_years = []
    if wb_path.exists():
        wb = pd.read_csv(wb_path)
        if "year" in wb.columns:
            wb_years = sorted(wb["year"].astype(int).tolist())

    # Pre-1990: DEU data gap (reunification)
    pre1990_rows = len(df[df["year"] < 1990])
    pre1990_wb = len([y for y in wb_years if y < 1990])
    data_gap = "WorldBank DEU starts 1990; 1974-1989 use flat defaults → v=0, a=0, score=1 only"

    # Per-year analysis
    candidates = []
    for _, row in df.iterrows():
        yr = int(row["year"])
        v = row.get("velocity")
        a = row.get("acceleration")
        score = row.get("saddle_score_phase1")
        if pd.isna(v) or pd.isna(a):
            continue
        v, a = float(v), float(a)
        score = float(score) if not pd.isna(score) else 0

        qualifies_score = score >= 2.0
        opposes = (v > 0 and a < 0) or (v < 0 and a > 0)
        near_saddle = abs(v) < NEAR_SADDLE_V_THRESH and opposes
        qualifies = qualifies_score or near_saddle

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

    # Root cause
    root_cause = []
    if pre1990_wb == 0:
        root_cause.append("data_quality: WorldBank DEU has no pre-1990 data (reunification); clock 1974-1989 is flat")
    if pre1990_rows > 0 and len([c for c in candidates if c["year"] < 1990]) == 0:
        root_cause.append("physics: pre-1990 flat clock yields score=1 only; no score>=2 or near_saddle (v=0,a=0 → no opposes)")
    if len(candidates) < 5:
        root_cause.append(f"physics: only {len(candidates)} saddle candidates in 1990-2020; DE dynamics may show fewer tension peaks")

    return {
        "n_candidates": len(candidates),
        "n_raw_episodes": len(raw),
        "need_for_contribution": 5,
        "gap": 5 - len(raw),
        "candidates": candidates,
        "era_distribution": {e: len(v) for e, v in eras.items()},
        "era_detail": {e: [(c["year"], c["score"], c.get("near_saddle")) for c in v] for e, v in eras.items()},
        "worldbank_years": (wb_years[:5] + ["..."] + wb_years[-3:]) if len(wb_years) > 8 else wb_years,
        "worldbank_span": f"{min(wb_years)}-{max(wb_years)}" if wb_years else "none",
        "pre1990_clock_rows": pre1990_rows,
        "pre1990_worldbank_rows": pre1990_wb,
        "data_gap_explanation": data_gap,
        "root_cause": root_cause,
        "event_years": event_years,
        "event_tolerance": EVENT_TOLERANCE,
        "raw_diag": diag,
    }


def main():
    print("DE (Germany) Episode Analysis")
    print("=" * 50)
    r = run_analysis()
    if "error" in r:
        print(r["error"])
        return 1
    print(f"Candidates: {r['n_candidates']} (raw episodes: {r['n_raw_episodes']})")
    print(f"Need for contribution: {r['need_for_contribution']}")
    print(f"WorldBank span: {r['worldbank_span']}")
    print(f"Pre-1990: clock rows={r['pre1990_clock_rows']}, WorldBank rows={r['pre1990_worldbank_rows']}")
    print(f"Data gap: {r['data_gap_explanation']}")
    print(f"Era distribution: {r['era_distribution']}")
    for era, items in r["era_detail"].items():
        if items:
            print(f"  {era}: {items}")
    print(f"Root cause: {r['root_cause']}")
    out_path = SCRIPT_DIR / "cerebro_data" / "de_episode_analysis.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
