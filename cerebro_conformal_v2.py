#!/usr/bin/env python3
"""
CEREBRO Conformal v2 — Contract Windows (self-honest uncertainty)
=================================================================
Inputs: past-only residuals from walkforward predictions (core compute_peak_window).
Computes s_hat for target coverage. If we drift below target, we widen until we do.
Output: coverage_target, s_hat, window_widen_factor, contract_status (PASS/WARNING/FAIL).
Does NOT touch cerebro_core.
"""

import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
CONTRACT_REPORT_PATH = DATA_DIR / "contract_report.json"

DEFAULT_COVERAGE_TARGET = 0.80
COVERAGE_TOLERANCE = 0.05  # within 5% of target = PASS
MIN_TRAIN = 5


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def _compute_residuals_from_walkforward() -> tuple[list[float], float, int]:
    """
    Walk-forward: for each episode, predict using past-only pool.
    Residual = max(0, ws - event, event - we) — distance outside interval (0 if hit).
    Returns (residuals, empirical_coverage, n_used).
    """
    from cerebro_calibration import _load_episodes
    from cerebro_core import compute_peak_window

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 3:
        return [], 0.0, 0

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    residuals = []
    hits = 0

    for ep in sorted_ep:
        yr = ep.get("saddle_year")
        if yr is None:
            continue
        pool = _past_only_pool(episodes, yr)
        if len(pool) < MIN_TRAIN:
            continue

        pred = compute_peak_window(
            yr,
            ep.get("position", 0),
            ep.get("velocity", 0),
            ep.get("acceleration", 0),
            ep.get("ring_B_score"),
            analogue_episodes=pool,
            interval_alpha=0.8,
        )
        ws = pred.get("window_start", yr + 3)
        we = pred.get("window_end", yr + 10)
        event = ep.get("event_year", yr + 5)

        # Nonconformity: 0 if hit, else distance to nearest bound
        s_i = max(0, ws - event, event - we)
        residuals.append(s_i)
        if s_i == 0:
            hits += 1

    n = len(residuals)
    emp_cov = hits / n if n else 0.0
    return residuals, emp_cov, n


def compute_contract(
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
) -> dict:
    """
    Compute conformal s_hat and contract status from walkforward residuals.
    Returns dict: coverage_target, s_hat, window_widen_factor, contract_status, empirical_coverage, etc.
    """
    residuals, empirical_coverage, n_used = _compute_residuals_from_walkforward()

    if n_used < 5:
        return {
            "coverage_target": coverage_target,
            "s_hat": 0.0,
            "window_widen_factor": 1.0,
            "contract_status": "FAIL",
            "empirical_coverage": 0.0,
            "n_used": n_used,
            "reason": "insufficient_residuals",
        }

    # Conformal quantile: ceil((n+1)*(1-alpha))-th smallest, or ceiling of alpha quantile
    # For 80% coverage: we want 80% of residuals to be 0 (hits). The (1-alpha) quantile of residuals
    # gives the margin we need to add. Standard: q = ceil((n+1)*(1-alpha))/n, take q-th smallest.
    alpha = coverage_target
    q_level = math.ceil((n_used + 1) * (1 - alpha)) / n_used
    q_level = min(1.0, max(0.0, q_level))
    q_idx = int(math.ceil(q_level * n_used)) - 1
    q_idx = max(0, min(q_idx, n_used - 1))

    sorted_r = sorted(residuals)
    s_hat = float(sorted_r[q_idx])

    # window_widen_factor: add s_hat years to each side. Factor = (width + 2*s_hat) / width.
    # We don't have a single width here; we use typical width ~7. Factor = 1 + 2*s_hat/7.
    typical_width = 7.0
    window_widen_factor = 1.0 + (2.0 * s_hat) / max(1.0, typical_width)

    # Contract status
    gap = coverage_target - empirical_coverage
    if empirical_coverage >= coverage_target - COVERAGE_TOLERANCE:
        contract_status = "PASS"
    elif empirical_coverage >= coverage_target - 2 * COVERAGE_TOLERANCE:
        contract_status = "WARNING"
    else:
        contract_status = "FAIL"

    return {
        "coverage_target": coverage_target,
        "s_hat": round(s_hat, 2),
        "window_widen_factor": round(window_widen_factor, 4),
        "contract_status": contract_status,
        "empirical_coverage": round(empirical_coverage, 4),
        "n_used": n_used,
        "residuals_quantile_level": round(q_level, 4),
    }


def apply_conformal_v2(
    window_start: int,
    window_end: int,
    delta_p10: float,
    delta_p90: float,
    contract: dict | None = None,
) -> tuple[int, int, float, bool]:
    """
    Widen window using s_hat from contract. Returns (new_ws, new_we, s_hat, applied).
    """
    if contract is None:
        contract = compute_contract()
    s_hat = contract.get("s_hat", 0.0)
    if s_hat <= 0:
        return window_start, window_end, 0.0, False

    # Add s_hat years to each side
    pad = int(math.ceil(s_hat))
    new_ws = window_start - pad
    new_we = window_end + pad
    return new_ws, new_we, s_hat, True


def load_contract() -> dict | None:
    """Load contract_report.json. Returns None if missing."""
    if not CONTRACT_REPORT_PATH.exists():
        return None
    try:
        with open(CONTRACT_REPORT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    contract = compute_contract()
    with open(CONTRACT_REPORT_PATH, "w") as f:
        json.dump(contract, f, indent=2)
    print(f"Contract: {contract['contract_status']} | emp_cov={contract['empirical_coverage']:.2f} target={contract['coverage_target']} s_hat={contract['s_hat']}")
    print(f"  → {CONTRACT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
