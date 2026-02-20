#!/usr/bin/env python3
"""
CEREBRO STATE-SPACE ESTIMATION — Bayesian latent state for each clock
======================================================================
Latent state x_t = [p, v, a] (position, velocity, acceleration).
Observations y_t from existing feeds; each source maps to H_k, R_k.
Linear Kalman filter; architecture extensible for particle filter.
Output: posterior mean + covariance per timepoint.

Provenance: deterministic seed, model params, observation mapping.
"""

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "state_space_posterior.json"
RANDOM_SEED = 42

# State: [position, velocity, acceleration]
STATE_DIM = 3
OBS_DIM = 3  # observe all three when available


def _observation_matrix(obs_mask: Tuple[bool, bool, bool]) -> np.ndarray:
    """H: which state components are observed. obs_mask = (p, v, a)."""
    rows = [np.eye(STATE_DIM)[i] for i, m in enumerate(obs_mask) if m]
    if not rows:
        return np.zeros((0, STATE_DIM))
    return np.vstack(rows)


def _transition_matrix(dt: float = 1.0) -> np.ndarray:
    """F: constant-acceleration model. p'=p+v*dt, v'=v+a*dt, a'=a."""
    return np.array([
        [1, dt, 0.5 * dt**2],
        [0, 1, dt],
        [0, 0, 1],
    ], dtype=float)


def _process_noise(dt: float = 1.0, q_scale: float = 0.1) -> np.ndarray:
    """Q: process noise covariance. Higher for acceleration (random walk)."""
    q = np.diag([0.01, 0.05, q_scale])
    return q


def _observation_noise(obs_mask: Tuple[bool, bool, bool], r_scale: float = 0.5) -> np.ndarray:
    """R: observation noise. Diagonal per observed component."""
    n = sum(obs_mask)
    return np.eye(n) * r_scale


def kalman_step(
    x: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    obs_mask: Tuple[bool, bool, bool],
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One Kalman filter step: predict, then update if observation available.
    Returns (x_post, P_post).
    """
    # Predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    if not any(obs_mask):
        return x_pred, P_pred

    H = _observation_matrix(obs_mask)
    y_obs = np.array([y[i] for i, m in enumerate(obs_mask) if m], dtype=float)

    # Update
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_post = x_pred + K @ (y_obs - H @ x_pred)
    I_KH = np.eye(STATE_DIM) - K @ H
    P_post = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
    P_post = 0.5 * (P_post + P_post.T)  # ensure symmetry

    return x_post, P_post


def run_filter(
    years: list[int],
    positions: list[Optional[float]],
    velocities: list[Optional[float]],
    accelerations: list[Optional[float]],
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Run Kalman filter over time series.
    Returns posterior mean + covariance per timepoint.
    """
    np.random.seed(seed)
    n = len(years)
    if n < 2:
        return {"error": "Insufficient data", "posteriors": []}

    F = _transition_matrix(1.0)
    Q = _process_noise(1.0)
    r_scale = 0.5

    if x0 is None:
        p0 = positions[0] if positions[0] is not None else 0.0
        v0 = velocities[0] if velocities[0] is not None else 0.0
        a0 = accelerations[0] if accelerations[0] is not None else 0.0
        x0 = np.array([p0, v0, a0], dtype=float)
    if P0 is None:
        P0 = np.diag([1.0, 0.5, 0.2])

    posteriors = []
    x = x0.copy()
    P = P0.copy()

    for t in range(n):
        y = np.array([
            positions[t] if t < len(positions) else None,
            velocities[t] if t < len(velocities) else None,
            accelerations[t] if t < len(accelerations) else None,
        ])
        obs_mask = (
            y[0] is not None and not (isinstance(y[0], float) and np.isnan(y[0])),
            y[1] is not None and not (isinstance(y[1], float) and np.isnan(y[1])),
            y[2] is not None and not (isinstance(y[2], float) and np.isnan(y[2])),
        )
        y_clean = np.array([
            float(y[0]) if obs_mask[0] else 0.0,
            float(y[1]) if obs_mask[1] else 0.0,
            float(y[2]) if obs_mask[2] else 0.0,
        ])
        R = _observation_noise(obs_mask, r_scale)
        x, P = kalman_step(x, P, y_clean, obs_mask, F, Q, R)
        posteriors.append({
            "year": int(years[t]),
            "mean": [round(float(x[0]), 4), round(float(x[1]), 4), round(float(x[2]), 4)],
            "cov_diag": [round(float(P[0, 0]), 6), round(float(P[1, 1]), 6), round(float(P[2, 2]), 6)],
        })

    return {
        "posteriors": posteriors,
        "provenance": {
            "model": "linear_kalman",
            "state_dim": STATE_DIM,
            "transition": "constant_acceleration",
            "random_seed": seed,
            "timestamp": int(time.time()),
            "version": 1,
        },
    }


def load_harm_clock_series() -> tuple[list[int], list, list, list]:
    """Load harm clock (p, v, a) from CSV. Returns (years, positions, velocities, accelerations)."""
    import pandas as pd
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return [], [], [], []
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["clock_position_10pt"].notna()].tail(80)
    years = df.index.astype(int).tolist()
    positions = df["clock_position_10pt"].tolist()
    velocities = df["velocity"].tolist()
    accelerations = df["acceleration"].tolist()
    return years, positions, velocities, accelerations


def run_harm_filter() -> dict:
    """Run Kalman filter on harm clock data, save posterior."""
    years, pos, vel, acc = load_harm_clock_series()
    result = run_filter(years, pos, vel, acc)
    if "error" not in result:
        OUTPUT_PATH.parent.mkdir(exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
    return result


def get_latest_posterior() -> Optional[dict]:
    """Return latest posterior mean + covariance for harm clock."""
    if not OUTPUT_PATH.exists():
        return None
    try:
        with open(OUTPUT_PATH) as f:
            d = json.load(f)
        posts = d.get("posteriors", [])
        if not posts:
            return None
        last = posts[-1]
        return {
            "year": last["year"],
            "position": last["mean"][0],
            "velocity": last["mean"][1],
            "acceleration": last["mean"][2],
            "position_std": last["cov_diag"][0] ** 0.5,
            "velocity_std": last["cov_diag"][1] ** 0.5,
            "acceleration_std": last["cov_diag"][2] ** 0.5,
        }
    except Exception:
        return None


if __name__ == "__main__":
    r = run_harm_filter()
    if "error" in r:
        print(f"Error: {r['error']}")
    else:
        print(f"Filtered {len(r['posteriors'])} timepoints → {OUTPUT_PATH}")
        lp = get_latest_posterior()
        if lp:
            print(f"Latest: year={lp['year']}, p={lp['position']:.2f}, v={lp['velocity']:.3f}, a={lp['acceleration']:.3f}")
