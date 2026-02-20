#!/usr/bin/env python3
"""
CHIMERA WEIGHTS ONLINE — SPSA learning of (vel_weight, acc_weight, tau)
======================================================================
Learn parameters via Simultaneous Perturbation Stochastic Approximation.
No gradients, stable online, small parameter space.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
MIN_TRAIN = 5
MIN_N_EFF_MEAN = 7
VW_BOUNDS = (10, 500)
AW_BOUNDS = (200, 20000)
TAU_BOUNDS = (0.5, 4.0)
SPSA_STEP = 0.1
SPSA_DELTA = 0.05


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def _compute_peak_with_tau(
    now_year: int,
    pos: float,
    vel: float,
    acc: float,
    rb: float | None,
    pool: list,
    vw: float,
    aw: float,
    tau: float,
) -> dict:
    """Compute peak using tau-weighted analogues. Does not touch core."""
    from cerebro_core import state_distance, weighted_median, weighted_quantile

    if not pool:
        return {"peak_year": now_year + 5, "window_start": now_year + 3, "window_end": now_year + 10}

    deltas = []
    weights = []
    for ep in pool:
        dt = ep.get("event_year", 0) - ep.get("saddle_year", 0)
        dist = state_distance(
            pos, vel, acc,
            ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
            vel_weight=vw, acc_weight=aw,
        )
        base_w = 1.0 / (1.0 + dist)
        w = base_w ** tau
        deltas.append(float(dt))
        weights.append(w)

    med = weighted_median(deltas, weights)
    p_lo = weighted_quantile(deltas, weights, 0.10)
    p_hi = weighted_quantile(deltas, weights, 0.90)
    return {
        "peak_year": now_year + int(round(med)),
        "window_start": now_year + int(round(p_lo)),
        "window_end": now_year + int(round(p_hi)),
    }


def _evaluate_theta(episodes: list, vw: float, aw: float, tau: float) -> tuple[float, float]:
    """Walk-forward MAE and mean n_eff for given (vw, aw, tau)."""
    mae_list = []
    n_effs = []
    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    for ep in sorted_ep:
        Y = ep.get("saddle_year")
        if Y is None:
            continue
        pool = _past_only_pool(episodes, Y)
        if len(pool) < MIN_TRAIN:
            continue
        pred = _compute_peak_with_tau(
            Y, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
            ep.get("ring_B_score"), pool, vw, aw, tau,
        )
        event_yr = ep.get("event_year", Y + 5)
        mae_list.append(abs(pred["peak_year"] - event_yr))
        n_effs.append(len(pool))
    mae = np.mean(mae_list) if mae_list else 999.0
    n_eff_mean = np.mean(n_effs) if n_effs else 0.0
    return mae, n_eff_mean


def _clip_params(vw: float, aw: float, tau: float) -> tuple[float, float, float]:
    vw = max(VW_BOUNDS[0], min(VW_BOUNDS[1], vw))
    aw = max(AW_BOUNDS[0], min(AW_BOUNDS[1], aw))
    tau = max(TAU_BOUNDS[0], min(TAU_BOUNDS[1], tau))
    return vw, aw, tau


def update_weights(episodes: list | None = None) -> dict:
    """
    SPSA update of (vw, aw, tau).
    Updates only if n_episodes >= min_train AND n_eff_mean >= 7.
    """
    from cerebro_chimera import chimera_store
    load_params = chimera_store.load_params
    atomic_write = chimera_store.atomic_write
    save_params_version = chimera_store.save_params_version
    from cerebro_calibration import _load_episodes

    if episodes is None:
        episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 5:
        return {"error": "Insufficient episodes", "updated": False}

    params = chimera_store.load_params()
    vw = float(params.get("vel_weight", 100))
    aw = float(params.get("acc_weight", 2500))
    tau = float(params.get("tau", 1.0))

    mae_base, n_eff_mean = _evaluate_theta(episodes, vw, aw, tau)
    if n_eff_mean < MIN_N_EFF_MEAN:
        return {"updated": False, "reason": "n_eff_mean < 7", "n_eff_mean": round(n_eff_mean, 2)}

    # SPSA: perturb theta = [log(vw), log(aw), tau]
    theta = np.array([np.log(vw), np.log(aw), tau])
    delta_vec = np.random.choice([-1, 1], size=3)
    theta_plus = theta + SPSA_DELTA * delta_vec
    theta_minus = theta - SPSA_DELTA * delta_vec

    vw_plus = np.exp(theta_plus[0])
    aw_plus = np.exp(theta_plus[1])
    tau_plus = theta_plus[2]
    vw_minus = np.exp(theta_minus[0])
    aw_minus = np.exp(theta_minus[1])
    tau_minus = theta_minus[2]

    mae_plus, _ = _evaluate_theta(episodes, vw_plus, aw_plus, tau_plus)
    mae_minus, _ = _evaluate_theta(episodes, vw_minus, aw_minus, tau_minus)

    # Gradient estimate: (L_plus - L_minus) / (2 * delta)
    g = (mae_plus - mae_minus) / (2 * SPSA_DELTA) * delta_vec
    theta_new = theta - SPSA_STEP * g
    vw_new = np.exp(theta_new[0])
    aw_new = np.exp(theta_new[1])
    tau_new = theta_new[2]
    vw_new, aw_new, tau_new = _clip_params(vw_new, aw_new, tau_new)

    mae_new, _ = _evaluate_theta(episodes, vw_new, aw_new, tau_new)
    rolling_mae = float(mae_new) if mae_new < 999 else params.get("rolling_mae")

    out = {
        "vel_weight": int(round(vw_new)),
        "acc_weight": int(round(aw_new)),
        "tau": round(float(tau_new), 3),
        "n_updates": params.get("n_updates", 0) + 1,
        "updated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "rolling_mae": round(rolling_mae, 3) if rolling_mae is not None else None,
        "mae_before": round(mae_base, 3),
        "mae_after": round(mae_new, 3),
        "version": 1,
    }
    chimera_store.atomic_write(DATA_DIR / "chimera_params.json", out)
    chimera_store.save_params_version(out)
    return out
