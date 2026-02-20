#!/usr/bin/env python3
"""
CEREBRO NETWORK SPILLOVER MODEL
If Country A enters high energy release + neighboring countries share trade/cultural similarity
→ increase hazard multiplier. Models systems, not nations.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "spillover_metrics.json"

# Adjacency: US neighbors (simplified; expand with full dataset)
ADJACENCY = {
    "USA": ["CAN", "MEX"],
    "CAN": ["USA"],
    "MEX": ["USA", "GTM", "BLZ"],
    "GBR": ["IRL", "FRA"],
    "FRA": ["DEU", "ESP", "ITA", "CHE", "BEL"],
    "DEU": ["FRA", "POL", "CZE", "AUT", "CHE", "NLD", "BEL"],
}

# Trade volume proxy (0–1, placeholder)
TRADE_WEIGHTS = {
    ("USA", "CAN"): 0.9, ("USA", "MEX"): 0.8,
    ("GBR", "USA"): 0.7, ("GBR", "FRA"): 0.6,
    ("FRA", "DEU"): 0.85, ("DEU", "POL"): 0.5,
}

# Linguistic similarity (0–1, placeholder)
LINGUISTIC = {
    ("USA", "GBR"): 0.95, ("USA", "CAN"): 0.9, ("USA", "AUS"): 0.9,
    ("FRA", "BEL"): 0.7, ("DEU", "AUT"): 0.9,
}


def _link_strength(a: str, b: str) -> float:
    """Combined: geographic + trade + linguistic."""
    a, b = a.upper()[:3], b.upper()[:3]
    geo = 1.0 if (a in ADJACENCY.get(b, []) or b in ADJACENCY.get(a, [])) else 0.3
    trade = TRADE_WEIGHTS.get((a, b), TRADE_WEIGHTS.get((b, a), 0.2))
    ling = LINGUISTIC.get((a, b), LINGUISTIC.get((b, a), 0.3))
    return 0.4 * geo + 0.35 * trade + 0.25 * ling


def get_coupling_coefficients() -> dict:
    """
    Cross-dimensional coupling: degree to which movement in one dimension predicts another.
    Dimensions: harm_tolerance, sexual_norms, class_permeability, good_vs_evil.
    Returns 4x4 matrix {dim_i: {dim_j: coeff}} or empty if not computed.
    NOTE: Current spillover models country-level contagion (adjacency/trade/linguistic),
    not cross-dimensional coupling. Returns zeros as placeholder until multi-dimension data exists.
    """
    dims = ["harm_tolerance", "sexual_norms", "class_permeability", "good_vs_evil"]
    return {d: {d2: 0.0 for d2 in dims if d2 != d} for d in dims}


def compute_contagion_risk(
    focal_country: str,
    energy_score: float,
    release_risk: str,
    neighbor_countries: list[str] | None = None,
) -> dict:
    """
    contagion_risk_index: 0–1. Higher when focal has high energy/release
    and neighbors are strongly linked.
    """
    if neighbor_countries is None:
        neighbor_countries = ADJACENCY.get(focal_country.upper()[:3], ["GBR", "MEX", "CAN"])

    # Base risk from focal
    base = 0.5 if release_risk == "HIGH" else (0.3 if release_risk == "MODERATE" else 0.1)
    base += 0.3 * min(1.0, energy_score)

    # Spillover from neighbors
    spill = 0.0
    for nb in neighbor_countries:
        link = _link_strength(focal_country, nb)
        spill += link * 0.15
    spill = min(1.0, spill)

    contagion_risk_index = min(1.0, base + spill)
    return {
        "contagion_risk_index": round(contagion_risk_index, 2),
        "focal_country": focal_country,
        "base_risk": round(base, 2),
        "spillover_contribution": round(spill, 2),
        "sources_used": ["adjacency", "trade_volume", "linguistic_similarity"],
    }


def run_spillover() -> dict:
    from cerebro_energy import compute_energy_metrics
    from cerebro_core import detect_saddle_canonical, _load_analogue_episodes

    episodes = _load_analogue_episodes()
    if not episodes:
        return {"contagion_risk_index": 0.0, "focal_country": "USA"}

    row = episodes[-1]
    is_sad, _ = detect_saddle_canonical(
        row["position"], row["velocity"], row["acceleration"], row.get("ring_B_score"),
    )
    prev = episodes[-2] if len(episodes) >= 2 else None
    em = compute_energy_metrics(
        row["position"], row["velocity"], row["acceleration"], is_sad,
        prev["position"] if prev else None,
        prev["velocity"] if prev else None,
        prev["acceleration"] if prev else None,
    )
    return compute_contagion_risk("USA", em["energy_score"], em["release_risk"])


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_spillover()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Spillover: contagion_risk_index={r.get('contagion_risk_index')}")
    print(f"  → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
