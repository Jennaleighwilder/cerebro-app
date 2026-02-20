#!/usr/bin/env python3
"""
CEREBRO CHANGE-POINT DETECTION — Online regime shift detection
==============================================================
BOCPD or residual-based CUSUM on latent state innovations.
Output: regime_shift_probability, last_change_estimate.

Provenance: method, hyperparams, timestamp.
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "changepoint_status.json"
RANDOM_SEED = 42


def _innovations(means: list[list[float]], observations: list[list[float]]) -> np.ndarray:
    """Compute prediction errors (innovations) from Kalman posterior means vs next observation."""
    if len(means) < 2 or len(observations) < 2:
        return np.array([])
    inno = []
    for i in range(min(len(means) - 1, len(observations) - 1)):
        pred = np.array(means[i]) + np.array([means[i][1], means[i][2], 0])  # p'=p+v, v'=v+a, a'=a
        obs = np.array(observations[i + 1])
        inno.append(obs - pred)
    return np.array(inno) if inno else np.array([]).reshape(0, 3)


def cusum_residual(
    innovations: np.ndarray,
    threshold: float = 2.0,
    drift: float = 0.0,
) -> tuple[float, int, list[float]]:
    """
    CUSUM on residual magnitude. Returns (regime_shift_probability, last_change_idx, cusum_series).
    """
    if len(innovations) < 3:
        return 0.0, -1, []
    mag = np.linalg.norm(innovations, axis=1)
    mean_mag = np.mean(mag)
    std_mag = max(np.std(mag), 1e-8)
    z = (mag - mean_mag - drift) / std_mag
    cusum = np.maximum(0, np.cumsum(z))
    max_cusum = np.max(cusum)
    # Regime shift probability: sigmoid of (max_cusum - threshold)
    prob = 1 / (1 + np.exp(-(max_cusum - threshold)))
    last_change = int(np.argmax(cusum)) if len(cusum) > 0 else -1
    return float(prob), last_change, cusum.tolist()


def run_changepoint(
    posteriors: list[dict],
    observations: Optional[list[dict]] = None,
    threshold: float = 2.0,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Run CUSUM on Kalman innovations from state-space posterior.
    posteriors: list of {year, mean, cov_diag}
    observations: list of {position, velocity, acceleration} or derived from posteriors
    """
    np.random.seed(seed)
    if len(posteriors) < 5:
        return {"error": "Insufficient data", "regime_shift_probability": 0.0}

    means = [p["mean"] for p in posteriors]
    if observations is None:
        observations = [{"position": m[0], "velocity": m[1], "acceleration": m[2]} for m in means]
    obs_arr = [[o["position"], o["velocity"], o["acceleration"]] for o in observations]
    inno = _innovations(means, obs_arr)
    if len(inno) < 3:
        return {"error": "Too few innovations", "regime_shift_probability": 0.0}

    prob, last_idx, cusum_series = cusum_residual(inno, threshold=threshold)
    years = [p["year"] for p in posteriors]
    last_change_year = years[last_idx] if 0 <= last_idx < len(years) else None

    return {
        "regime_shift_probability": round(prob, 4),
        "last_change_estimate": last_change_year,
        "last_change_index": last_idx,
        "n_innovations": len(inno),
        "provenance": {
            "method": "cusum_residual",
            "threshold": threshold,
            "random_seed": seed,
            "timestamp": int(time.time()),
            "version": 1,
        },
    }


def run_from_state_space() -> dict:
    """Load state-space posterior, run changepoint detection."""
    post_path = SCRIPT_DIR / "cerebro_data" / "state_space_posterior.json"
    if not post_path.exists():
        return {"error": "State-space posterior not found", "regime_shift_probability": 0.0}
    with open(post_path) as f:
        data = json.load(f)
    posteriors = data.get("posteriors", [])
    return run_changepoint(posteriors)


def get_changepoint_status() -> Optional[dict]:
    """Load saved changepoint status if exists."""
    if not OUTPUT_PATH.exists():
        return None
    try:
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


if __name__ == "__main__":
    from cerebro_state_space import run_harm_filter
    run_harm_filter()
    r = run_from_state_space()
    if "error" in r:
        print(f"Error: {r['error']}")
    else:
        print(f"Regime shift probability: {r['regime_shift_probability']:.2%}")
        print(f"Last change estimate: year {r.get('last_change_estimate')}")
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  → {OUTPUT_PATH}")
