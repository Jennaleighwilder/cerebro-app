#!/usr/bin/env python3
"""
CHIMERA FORWARD SIMULATOR — Figure-8 forward arc (10k runs, 15yr horizon)
=========================================================================
State equations with shock mechanism. Systemic instability from |v|+|a|+OOD.
"""

import json
import random
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_forward_simulation.json"

SIM_YEARS = 15
SIM_RUNS = 10000
SIGMA_V = 0.05
SIGMA_A = 0.03
ACC_DECAY = 0.85
SHOCK_PROB_BASE = 0.02
SHOCK_SCALE = 0.10


def _get_initial_state() -> tuple[float, float, float]:
    """Load latest position, velocity, acceleration from harm clock."""
    import pandas as pd
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return 0.0, 0.0, 0.0
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(1)
    if df.empty:
        return 0.0, 0.0, 0.0
    row = df.iloc[-1]
    pos = float(row.get("clock_position_10pt", 0))
    vel = float(row.get("velocity", 0))
    acc = float(row.get("acceleration", 0))
    return pos, vel, acc


def _get_ood_percentile() -> float:
    """Load OOD percentile from distribution_shift.json."""
    p = SCRIPT_DIR / "cerebro_data" / "distribution_shift.json"
    if not p.exists():
        return 0.5
    try:
        with open(p) as f:
            return float(json.load(f).get("percentile", 0.5))
    except Exception:
        return 0.5


def _hazard_p1yr(pos: float, vel: float, acc: float, predict_delta) -> float:
    try:
        delta = predict_delta(pos, vel, acc)
        return min(0.5, 1.0 / max(1.0, delta))
    except Exception:
        return 0.1


def _simulate_path(
    now_year: int,
    predict_delta,
    initial_state: tuple | None = None,
    seed: int | None = None,
    shock_prob_mult: float = 1.0,
) -> int | None:
    """Returns time-to-event or None if censored."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pos, vel, acc = initial_state if initial_state else _get_initial_state()
    ood = _get_ood_percentile()
    systemic_load = abs(vel) + abs(acc) + ood * 0.5

    for t in range(1, SIM_YEARS + 1):
        vel_new = vel + acc + float(np.random.normal(0, SIGMA_V))
        pos_new = pos + vel_new
        acc_new = ACC_DECAY * acc + float(np.random.normal(0, SIGMA_A))

        shock_prob = (SHOCK_PROB_BASE + systemic_load * 0.01) * shock_prob_mult
        if random.random() < shock_prob:
            acc_new += float(np.random.normal(0, SHOCK_SCALE))

        pos, vel, acc = pos_new, vel_new, acc_new
        systemic_load = min(1.0, systemic_load + 0.05)

        p1 = _hazard_p1yr(pos, vel, acc, predict_delta)
        if random.random() < p1:
            return t

    return None


def run_forward_simulation(
    initial_state: tuple | None = None,
    pool: list | None = None,
    now_year: int | None = None,
    n_runs: int | None = None,
) -> dict:
    """Run 10k paths, collect time-to-event distribution."""
    from cerebro_calibration import _load_episodes
    from cerebro_sister_engine import sister_train_predict_fn

    episodes = _load_episodes()[0] if pool is None else []
    if pool is None:
        if len(episodes) < 5:
            return {"error": "Insufficient episodes", "version": 1, "sim_runs": 0}
        now_year = max(e.get("saddle_year", 0) for e in episodes)
        pool = [e for e in episodes if e.get("saddle_year", 0) < now_year]
    if len(pool) < 5:
        return {"error": "Insufficient past episodes", "version": 1, "sim_runs": 0}
    if now_year is None:
        now_year = max(e.get("saddle_year", 0) for e in pool) + 1
    predict_delta, _ = sister_train_predict_fn(pool)
    runs = n_runs or SIM_RUNS

    times = []
    for i in range(runs):
        tt = _simulate_path(now_year, predict_delta, initial_state=initial_state, seed=i)
        times.append(tt)

    censored = sum(1 for t in times if t is None)
    events = [t for t in times if t is not None]
    dist = {}
    for y in range(1, SIM_YEARS + 1):
        dist[str(y)] = round(sum(1 for t in events if t == y) / runs, 4)

    p5 = sum(1 for t in events if t is not None and t <= 5) / runs
    p10 = sum(1 for t in events if t is not None and t <= 10) / runs
    no_event_15 = censored / runs
    mean_tte = np.mean(events) if events else 0
    median_tte = int(np.median(events)) if events else 0

    # Entropy of time distribution
    probs = [dist.get(str(y), 0) for y in range(1, SIM_YEARS + 1)]
    probs.append(no_event_15)
    probs = [p for p in probs if p > 0]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs) if probs else 0

    # Tail heaviness: fraction beyond 10yr
    tail_heavy = sum(1 for t in events if t is not None and t > 10) / runs if events else 0

    return {
        "version": 1,
        "sim_runs": runs,
        "sim_years": SIM_YEARS,
        "event_probability_5yr": round(float(p5), 2),
        "event_probability_10yr": round(float(p10), 2),
        "mean_time_to_event": round(float(mean_tte), 2),
        "median_time_to_event": median_tte,
        "no_event_within_15yr": round(float(no_event_15), 2),
        "distribution": dist,
        "entropy_time_distribution": round(float(entropy), 4),
        "tail_heaviness_index": round(float(tail_heavy), 4),
    }


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_forward_simulation()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera forward sim: P_5yr={r.get('event_probability_5yr')}, entropy={r.get('entropy_time_distribution')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
