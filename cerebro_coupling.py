#!/usr/bin/env python3
"""
CEREBRO CROSS-CLOCK COUPLING LAYER
count_saddles_active, joint_descent, systemic_load.
Escalate horizon compression when count_saddles >= 2 AND systemic_load > threshold.
"""

from typing import Optional

SYSTEMIC_LOAD_THRESHOLD = 0.5
COUPLING_FACTOR_HIGH = 0.8  # shorten window to 80%


def count_saddles_active(clocks: list[dict]) -> int:
    """Number of clocks with saddle_score >= 2."""
    return sum(1 for c in clocks if c.get("saddle_score", 0) >= 2)


def joint_descent(clocks: list[dict]) -> int:
    """Number of clocks where velocity < 0 and acceleration < 0."""
    return sum(1 for c in clocks if c.get("velocity", 0) < 0 and c.get("acceleration", 0) < 0)


def systemic_load(clocks: list[dict]) -> float:
    """Sum of normalized intensity across clocks. Intensity from saddle_score/3."""
    return sum(min(1.0, c.get("saddle_score", 0) / 3.0) for c in clocks) / max(1, len(clocks))


def coupling_factor(
    clocks: list[dict],
    threshold: float = SYSTEMIC_LOAD_THRESHOLD,
) -> float:
    """If count_saddles >= 2 AND systemic_load > threshold: apply compression."""
    n_sad = count_saddles_active(clocks)
    load = systemic_load(clocks)
    if n_sad >= 2 and load > threshold:
        return COUPLING_FACTOR_HIGH
    return 1.0


def adjust_window(
    window_start: int,
    window_end: int,
    peak_year: int,
    coupling_factor: float,
) -> tuple[int, int]:
    """adjusted_window = base_window * coupling_factor around peak."""
    if coupling_factor >= 1.0:
        return window_start, window_end
    half = (window_end - window_start) / 2
    new_half = half * coupling_factor
    return (
        int(peak_year - new_half),
        int(peak_year + new_half),
    )


def systemic_instability_index(clocks: list[dict]) -> float:
    """0-1 index. Higher = more systemic instability."""
    n_sad = count_saddles_active(clocks)
    load = systemic_load(clocks)
    joint = joint_descent(clocks)
    return min(1.0, (n_sad / 4.0) * 0.4 + load * 0.4 + (joint / 4.0) * 0.2)
