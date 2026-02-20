#!/usr/bin/env python3
"""
CHIMERA ENTROPY — Structural uncertainty metrics
================================================
Entropy of forward event distribution, regime Markov 10yr, analogue weights.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "chimera_entropy.json"


def _entropy(probs: list[float]) -> float:
    """H = -sum p_i log(p_i)."""
    probs = [p for p in probs if p > 0]
    return -sum(p * np.log(p + 1e-10) for p in probs) if probs else 0.0


def run_entropy() -> dict:
    """Compute entropy of forward dist, regime Markov, analogue weights."""
    out = {"version": 1}

    # 1. Forward event distribution (from chimera_forward_simulation or forward_simulation)
    sim_path = SCRIPT_DIR / "cerebro_data" / "chimera_forward_simulation.json"
    if not sim_path.exists():
        sim_path = SCRIPT_DIR / "cerebro_data" / "forward_simulation.json"
    if sim_path.exists():
        try:
            with open(sim_path) as f:
                sim = json.load(f)
            dist = sim.get("distribution", {})
            probs = [float(dist.get(str(y), 0)) for y in range(1, 16)]
            no_event = sim.get("no_event_within_15yr", 0)
            probs.append(float(no_event))
            probs = [p for p in probs if p > 0]
            out["forward_event_entropy"] = round(_entropy(probs), 4)
        except Exception:
            out["forward_event_entropy"] = None
    else:
        out["forward_event_entropy"] = None

    # 2. Regime Markov 10-year distribution
    regime_path = SCRIPT_DIR / "cerebro_data" / "regime_markov.json"
    if regime_path.exists():
        try:
            with open(regime_path) as f:
                rm = json.load(f)
            p10 = rm.get("p_10yr", {})
            if isinstance(p10, dict):
                probs = [float(v) for v in p10.values() if v is not None]
            else:
                probs = []
            out["regime_markov_10yr_entropy"] = round(_entropy(probs), 4) if probs else None
        except Exception:
            out["regime_markov_10yr_entropy"] = None
    else:
        out["regime_markov_10yr_entropy"] = None

    # 3. Analogue weight distribution (from last honeycomb; we'd need to recompute)
    # Approximate: use reconstruction records' n_eff as proxy for "effective" weight concentration
    rec_path = SCRIPT_DIR / "cerebro_data" / "chimera_reconstruction.json"
    if rec_path.exists():
        try:
            with open(rec_path) as f:
                rec = json.load(f)
            records = rec.get("records", [])
            n_effs = [r.get("n_eff", 5) for r in records if r.get("n_eff")]
            if n_effs:
                # Normalize to pseudo-probs
                total = sum(n_effs)
                probs = [n / total for n in n_effs]
                out["analogue_weight_entropy"] = round(_entropy(probs), 4)
            else:
                out["analogue_weight_entropy"] = None
        except Exception:
            out["analogue_weight_entropy"] = None
    else:
        out["analogue_weight_entropy"] = None

    # Interpretation
    fwd = out.get("forward_event_entropy")
    if fwd is not None:
        out["interpretation"] = "high_structural_uncertainty" if fwd > 2.0 else "deterministic_drift"

    return out


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_entropy()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Chimera entropy: forward={r.get('forward_event_entropy')} → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
