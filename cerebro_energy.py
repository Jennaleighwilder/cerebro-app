#!/usr/bin/env python3
"""
CEREBRO ENERGY MODEL — Layer, not replacement.
E = 0.5*v^2 + lambda*|position - equilibrium|
"""

from typing import Optional

EQUILIBRIUM = 0.0
LAMBDA = 0.5  # configurable


def energy(velocity: float, position: float, equilibrium: float = EQUILIBRIUM, lam: float = LAMBDA) -> float:
    """E = 0.5 * velocity^2 + lambda * abs(position - equilibrium)"""
    return 0.5 * velocity ** 2 + lam * abs(position - equilibrium)


def energy_derivative(
    v: float, a: float, p: float,
    v_prev: float, a_prev: float, p_prev: float,
    dt: float = 1.0,
) -> float:
    """dE/dt via finite diff."""
    E_now = energy(v, p)
    E_prev = energy(v_prev, p_prev)
    return (E_now - E_prev) / dt if dt > 0 else 0.0


def release_risk(
    is_saddle: bool,
    energy_val: float,
    dE_dt: float,
    high_thresh: float = 0.5,
) -> str:
    """If saddle AND E high AND dE_dt negative: HIGH RELEASE RISK."""
    if is_saddle and energy_val > high_thresh and dE_dt < 0:
        return "HIGH"
    if is_saddle and energy_val > 0.2:
        return "MODERATE"
    return "LOW"


def compute_energy_metrics(
    position: float,
    velocity: float,
    acceleration: float,
    is_saddle: bool,
    position_prev: Optional[float] = None,
    velocity_prev: Optional[float] = None,
    acceleration_prev: Optional[float] = None,
) -> dict:
    """Compute energy_score and release_risk for export."""
    E = energy(velocity, position)
    dE = 0.0
    if position_prev is not None and velocity_prev is not None:
        dE = energy_derivative(
            velocity, acceleration, position,
            velocity_prev, acceleration_prev or 0, position_prev,
        )
    risk = release_risk(is_saddle, E, dE)
    return {
        "energy_score": round(E, 4),
        "release_risk": risk,
    }
