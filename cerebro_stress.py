#!/usr/bin/env python3
"""
CEREBRO MONTE CARLO STRESS TEST
Add gaussian noise, run 1000 sims, record peak_year distribution.
"""

import json
import random
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "stress_test.json"
RANDOM_SEED = 42
N_SIMS = 1000
NOISE_STD = 0.1
PEAK_STD_THRESHOLD = 2.0


def run_stress(
    now_year: int,
    position: float,
    velocity: float,
    acceleration: float,
    analogue_episodes: list,
    ring_b_score: Optional[float] = None,
    n_sims: int = N_SIMS,
    noise_std: float = NOISE_STD,
    seed: int = RANDOM_SEED,
) -> dict:
    from cerebro_core import compute_peak_window

    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    peaks = []
    windows = []
    for _ in range(n_sims):
        p = position + np.random.normal(0, noise_std)
        v = velocity + np.random.normal(0, noise_std)
        a = acceleration + np.random.normal(0, noise_std)
        pred = compute_peak_window(now_year, p, v, a, ring_b_score, analogue_episodes, 0.8)
        peaks.append(pred["peak_year"])
        windows.append((pred["window_start"], pred["window_end"]))

    peaks_arr = np.array(peaks)
    peak_mean = float(np.mean(peaks_arr))
    peak_std = float(np.std(peaks_arr))

    # Window robustness: % of sims where window overlaps the median window
    med_ws = int(np.median([w[0] for w in windows]))
    med_we = int(np.median([w[1] for w in windows]))
    overlap = sum(1 for ws, we in windows if not (we < med_ws or ws > med_we)) / n_sims

    return {
        "peak_mean": round(peak_mean, 2),
        "peak_std": round(peak_std, 2),
        "window_robustness": round(overlap * 100, 2),
        "instability_flag": peak_std > PEAK_STD_THRESHOLD,
        "n_sims": n_sims,
    }


def main():
    from cerebro_core import _load_analogue_episodes

    episodes = _load_analogue_episodes()
    if len(episodes) < 5:
        print("Insufficient episodes")
        return 1
    row = episodes[-1]
    r = run_stress(
        row["saddle_year"], row["position"], row["velocity"], row["acceleration"],
        [e for e in episodes if e["saddle_year"] != row["saddle_year"]],
        row.get("ring_B_score"),
    )
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Stress: peak_std={r['peak_std']}, robustness={r['window_robustness']}%")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
