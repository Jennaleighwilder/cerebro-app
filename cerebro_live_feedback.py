#!/usr/bin/env python3
"""
Real-time feedback loop. When a predicted peak window closes (current year
passes window_end), score it against what actually happened and feed result
back into calibration within 24 hours.
"""

import json
from pathlib import Path
from datetime import datetime, date, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
FEEDBACK_LOG = DATA_DIR / "feedback_log.jsonl"
PENDING_WINDOWS = DATA_DIR / "pending_windows.json"
HARM_CLOCK_CSV = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
FEEDBACK_EPISODES = DATA_DIR / "feedback_episodes.json"


def register_prediction(
    clock_name: str,
    window_start: int,
    window_end: int,
    confidence_pct: float,
    peak_year_predicted: int,
    saddle_year: int | None = None,
    position: float | None = None,
    velocity: float | None = None,
    acceleration: float | None = None,
    ring_B_score: float | None = None,
    country: str = "US",
) -> None:
    """
    Called by cerebro_peak_window when a new forecast is made.
    Stores prediction in pending_windows.json for future scoring.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    if PENDING_WINDOWS.exists():
        try:
            with open(PENDING_WINDOWS) as f:
                pending = json.load(f)
        except Exception:
            pending = []
    entry = {
        "clock": clock_name,
        "window_start": window_start,
        "window_end": window_end,
        "confidence_pct": confidence_pct,
        "peak_year_predicted": peak_year_predicted,
        "saddle_year": saddle_year,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "ring_B_score": ring_B_score,
        "country": country,
        "registered_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Avoid duplicates: same clock + window
    key = (clock_name, window_start, window_end)
    pending = [p for p in pending if (p.get("clock"), p.get("window_start"), p.get("window_end")) != key]
    pending.append(entry)
    with open(PENDING_WINDOWS, "w") as f:
        json.dump(pending, f, indent=2)


def score_closed_windows(current_year: int | None = None) -> int:
    """
    Check all pending windows where window_end < current_year.
    For each closed window, determine if a real event occurred within
    the window by checking cerebro_harm_clock_data.csv for saddle_score >= 2
    in that year range.

    Score as hit (1) or miss (0).
    Append to feedback_log.jsonl.
    Returns count of scored windows.
    """
    if current_year is None:
        current_year = date.today().year
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    if PENDING_WINDOWS.exists():
        try:
            with open(PENDING_WINDOWS) as f:
                pending = json.load(f)
        except Exception:
            pending = []
    if not pending:
        return 0
    # Load saddle years from harm clock
    saddle_years = set()
    if HARM_CLOCK_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(HARM_CLOCK_CSV, index_col=0)
            if "saddle_score" in df.columns:
                for yr, row in df.iterrows():
                    try:
                        sc = float(row.get("saddle_score", 0))
                        if sc >= 2:
                            saddle_years.add(int(yr))
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
    scored = 0
    still_pending = []
    for p in pending:
        we = p.get("window_end")
        if we is None or we >= current_year:
            still_pending.append(p)
            continue
        ws = p.get("window_start", we - 5)
        hit = any(ws <= yr <= we for yr in saddle_years)
        actual_peak_year = None
        if hit:
            candidates = [yr for yr in saddle_years if ws <= yr <= we]
            actual_peak_year = min(candidates) if candidates else None
        log_entry = {
            "clock": p.get("clock"),
            "predicted_window": [ws, we],
            "confidence_pct": p.get("confidence_pct"),
            "hit": hit,
            "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "actual_peak_year": actual_peak_year,
            "saddle_year": p.get("saddle_year"),
            "position": p.get("position"),
            "velocity": p.get("velocity"),
            "acceleration": p.get("acceleration"),
            "ring_B_score": p.get("ring_B_score"),
            "country": p.get("country", "US"),
        }
        with open(FEEDBACK_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        scored += 1
    with open(PENDING_WINDOWS, "w") as f:
        json.dump(still_pending, f, indent=2)
    return scored


def inject_feedback_into_calibration() -> int:
    """
    Read feedback_log.jsonl. Convert scored predictions into calibration
    episodes in the same format as cerebro_calibration.py expects.
    Append to feedback_episodes.json — do not replace.

    Call cerebro_calibration.run_calibration() after injecting.
    Returns count of new episodes added.
    """
    if not FEEDBACK_LOG.exists():
        return 0
    episodes = []
    if FEEDBACK_EPISODES.exists():
        try:
            with open(FEEDBACK_EPISODES) as f:
                episodes = json.load(f)
        except Exception:
            episodes = []
    seen = {(e.get("saddle_year"), tuple(e.get("predicted_window", []))) for e in episodes}
    added = 0
    with open(FEEDBACK_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            key = (entry.get("saddle_year"), tuple(entry.get("predicted_window", [])))
            if key in seen:
                continue
            ws, we = entry.get("predicted_window", [0, 0])
            event_yr = entry.get("actual_peak_year")
            if event_yr is None:
                event_yr = we + 1  # miss: event outside window
            ep = {
                "saddle_year": entry.get("saddle_year"),
                "event_year": event_yr,
                "position": entry.get("position", 0),
                "velocity": entry.get("velocity", 0),
                "acceleration": entry.get("acceleration", 0),
                "ring_B_score": entry.get("ring_B_score"),
                "country": entry.get("country", "US"),
                "confidence": entry.get("confidence_pct", 70) / 100.0,
                "hit": entry.get("hit", False),
                "source": "feedback",
                "predicted_window": [ws, we],
            }
            episodes.append(ep)
            seen.add(key)
            added += 1
    if added > 0:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_EPISODES, "w") as f:
            json.dump(episodes, f, indent=2)
    return added


def run_feedback_cycle() -> int:
    """Main entry point. Score closed windows, inject into calibration, report."""
    scored = score_closed_windows()
    if scored > 0:
        inject_feedback_into_calibration()
    return scored


def main():
    scored = run_feedback_cycle()
    print(f"Feedback cycle: scored {scored} closed windows")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
