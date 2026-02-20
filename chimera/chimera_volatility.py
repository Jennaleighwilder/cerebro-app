#!/usr/bin/env python3
"""
CHIMERA Volatility Index — Detect adversarial/noise regime from position/velocity/acceleration.
High volatility → cap confidence. Does not change prediction, only trust.
"""

import numpy as np

VOL_SCALE = 0.8
DEFAULT_K = 10


def compute_volatility_index(
    positions: list[float],
    velocities: list[float],
    accelerations: list[float],
    k: int = DEFAULT_K,
) -> float:
    """
    vol_index = std(Δposition) + std(velocity) + std(acceleration) over last k.
    Normalized to 0–1: vol_norm = min(1.0, vol_index / VOL_SCALE).
    If vol_norm > 0.7 → treat as adversarial/noise → cap confidence.
    """
    if not positions or not velocities or not accelerations:
        return 0.0
    n = min(k, len(positions), len(velocities), len(accelerations))
    if n < 3:
        return 0.0
    pos = np.array(positions[-n:], dtype=float)
    vel = np.array(velocities[-n:], dtype=float)
    acc = np.array(accelerations[-n:], dtype=float)
    pos = pos[~np.isnan(pos)]
    vel = vel[~np.isnan(vel)]
    acc = acc[~np.isnan(acc)]
    if len(pos) < 2:
        delta_pos = np.array([0.0])
    else:
        delta_pos = np.diff(pos)
    std_dp = np.std(delta_pos) if len(delta_pos) else 0.0
    std_v = np.std(vel) if len(vel) else 0.0
    std_a = np.std(acc) if len(acc) else 0.0
    vol_index = float(std_dp) + float(std_v) + float(std_a)
    vol_norm = min(1.0, vol_index / VOL_SCALE)
    return round(vol_norm, 4)
