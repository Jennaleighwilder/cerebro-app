#!/usr/bin/env python3
"""
CEREBRO CLOCK COUPLING — Cross-clock interaction matrix
========================================================
Coupling matrix C and cross-features (Class×Harm, Harm×Evil, etc.).
Configurable + documented with provenance.
"""

import json
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "coupling_config.json"

# Clock IDs
CLOCKS = ["harm", "class", "sexual", "evil"]

# Default coupling matrix: C[i][j] = influence of clock j on clock i
# Sparse: Harm is data-driven; Class/Sexual/Evil are placeholders.
# Class×Harm: class permeability affects harm tolerance (redistribution)
# Harm×Evil: harm tolerance affects accountability framing
# Sexual×Class: sexual autonomy and class mobility can co-move
DEFAULT_COUPLING = {
    "harm": {"harm": 1.0, "class": 0.15, "sexual": 0.05, "evil": 0.10},
    "class": {"harm": 0.20, "class": 1.0, "sexual": 0.10, "evil": 0.05},
    "sexual": {"harm": 0.05, "class": 0.15, "sexual": 1.0, "evil": 0.05},
    "evil": {"harm": 0.15, "class": 0.05, "sexual": 0.05, "evil": 1.0},
}


def load_coupling() -> dict:
    """Load coupling config from file or return default."""
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "matrix": DEFAULT_COUPLING,
        "provenance": {
            "source": "default",
            "description": "Sparse coupling: Harm data-driven; Class/Harm, Harm/Evil cross-terms.",
            "version": 1,
        },
    }


def save_coupling(config: dict) -> None:
    """Save coupling config."""
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(config, f, indent=2)


def cross_features(
    clock_states: dict[str, dict],
    coupling: Optional[dict] = None,
) -> dict[str, float]:
    """
    Compute cross-features for a target clock given all clock states.
    clock_states: {clock_id: {position, velocity, acceleration}}
    Returns: {feature_name: value} for use in hazard or state update.
    """
    if coupling is None:
        coupling = load_coupling()
    matrix = coupling.get("matrix", DEFAULT_COUPLING)

    out = {}
    for target in CLOCKS:
        row = matrix.get(target, {target: 1.0})
        for source, weight in row.items():
            if source == target or weight == 0:
                continue
            state = clock_states.get(source, {})
            p = state.get("position", 0)
            v = state.get("velocity", 0)
            a = state.get("acceleration", 0)
            out[f"{target}_from_{source}_p"] = weight * p
            out[f"{target}_from_{source}_v"] = weight * v
            out[f"{target}_from_{source}_a"] = weight * a
    return out


def apply_coupling(
    base_state: dict,
    other_states: dict[str, dict],
    target_clock: str = "harm",
) -> dict:
    """
    Apply coupling to base state. Returns adjusted state.
    base_state: {position, velocity, acceleration} for target
    other_states: {clock_id: {position, velocity, acceleration}}
    """
    coupling = load_coupling()
    matrix = coupling.get("matrix", DEFAULT_COUPLING)
    row = matrix.get(target_clock, {target_clock: 1.0})

    p, v, a = base_state.get("position", 0), base_state.get("velocity", 0), base_state.get("acceleration", 0)
    dp, dv, da = 0.0, 0.0, 0.0
    for source, weight in row.items():
        if source == target_clock:
            continue
        s = other_states.get(source, {})
        sp = s.get("position", 0)
        sv = s.get("velocity", 0)
        sa = s.get("acceleration", 0)
        dp += weight * 0.1 * sp  # weak position coupling
        dv += weight * 0.05 * sv
        da += weight * 0.02 * sa

    return {
        "position": p + dp,
        "velocity": v + dv,
        "acceleration": a + da,
        "coupling_applied": True,
        "provenance": {"config": coupling.get("provenance", {})},
    }


def get_provenance() -> dict:
    """Return coupling provenance for /method and show-math."""
    c = load_coupling()
    return {
        "coupling_matrix": c.get("matrix", DEFAULT_COUPLING),
        "provenance": c.get("provenance", {}),
        "clocks": CLOCKS,
    }


if __name__ == "__main__":
    c = load_coupling()
    print("Coupling matrix:")
    for row, col in c.get("matrix", DEFAULT_COUPLING).items():
        print(f"  {row}: {col}")
    print(f"Provenance: {c.get('provenance', {})}")
