#!/usr/bin/env python3
"""
CHIMERA Arena — Controlled parameter experiments without touching frozen core.
Runs candidate configs (vel_weight, acc_weight, tau, interval_alpha), ranks by metrics,
promotes operational override when gates pass. Temporal consistency guard for rollback.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
ARENA_STATE_PATH = DATA_DIR / "chimera_arena_state.json"
LOG_PATH = DATA_DIR / "chimera_log.jsonl"

VW_BOUNDS = (10, 500)
AW_BOUNDS = (200, 20000)
TAU_BOUNDS = (0.5, 4.0)
MIN_TRAIN = 5
PROMOTE_THRESHOLD = 0.02  # Must beat current by 2% on composite
MIN_N_EFF_PROMOTE = 10
MIN_COVERAGE_PROMOTE = 0.75
ROLLBACK_DEGRADE_RUNS = 2  # Rollback if this many consecutive runs degrade


def _past_only_pool(episodes: list, t: int) -> list:
    return [e for e in episodes if e.get("saddle_year", 0) < t]


def _q_from_alpha(alpha: float) -> tuple[float, float]:
    """interval_alpha 0.8 -> (0.1, 0.9); 0.75 -> (0.125, 0.875); 0.85 -> (0.075, 0.925)."""
    q_lo = (1 - alpha) / 2
    q_hi = (1 + alpha) / 2
    return q_lo, q_hi


def _compute_peak_with_config(
    now_year: int,
    pos: float,
    vel: float,
    acc: float,
    rb: float | None,
    pool: list,
    vw: float,
    aw: float,
    tau: float,
    interval_alpha: float,
) -> dict:
    """Compute peak using config. Uses tau-weighted when tau != 1; else core."""
    from cerebro_core import state_distance, weighted_median, weighted_quantile, compute_peak_window

    if not pool:
        return {"peak_year": now_year + 5, "window_start": now_year + 3, "window_end": now_year + 10, "hit": False}

    q_lo, q_hi = _q_from_alpha(interval_alpha)

    if abs(tau - 1.0) < 0.01:
        pred = compute_peak_window(
            now_year, pos, vel, acc, rb, pool,
            interval_alpha=interval_alpha,
            vel_weight=vw,
            acc_weight=aw,
        )
        return pred

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
    p_lo = weighted_quantile(deltas, weights, q_lo)
    p_hi = weighted_quantile(deltas, weights, q_hi)
    return {
        "peak_year": now_year + int(round(med)),
        "window_start": now_year + int(round(p_lo)),
        "window_end": now_year + int(round(p_hi)),
        "analogue_count": len(pool),
    }


def _evaluate_config(episodes: list, config: dict) -> dict:
    """Walk-forward evaluation for one config. Returns brier, coverage_80, mae, mean_n_eff."""
    vw = config.get("vel_weight", 100)
    aw = config.get("acc_weight", 2500)
    tau = config.get("tau", 1.0)
    alpha = config.get("interval_alpha", 0.8)

    sorted_ep = sorted(episodes, key=lambda e: e.get("saddle_year", 0))
    results = []
    for ep in sorted_ep:
        t = ep.get("saddle_year")
        if t is None:
            continue
        pool = _past_only_pool(episodes, t)
        if len(pool) < MIN_TRAIN:
            continue
        pred = _compute_peak_with_config(
            t, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
            ep.get("ring_B_score"), pool, vw, aw, tau, alpha,
        )
        event_yr = ep.get("event_year", t + 5)
        hit = pred["window_start"] <= event_yr <= pred["window_end"]
        err = abs(pred.get("peak_year", t + 5) - event_yr)
        conf = 0.8 if alpha >= 0.75 else 0.5
        results.append({
            "hit": hit,
            "err": err,
            "n_eff": len(pool),
            "confidence": conf,
        })

    if not results:
        return {"brier": 0.5, "coverage_80": 0, "mae": 999, "mean_n_eff": 0, "n": 0}

    n = len(results)
    brier = sum((r["confidence"] - (1.0 if r["hit"] else 0.0)) ** 2 for r in results) / n
    coverage = sum(1 for r in results if r["hit"]) / n
    mae = sum(r["err"] for r in results) / n
    mean_n_eff = sum(r["n_eff"] for r in results) / n
    return {
        "brier": round(brier, 4),
        "coverage_80": round(coverage, 4),
        "mae": round(mae, 3),
        "mean_n_eff": round(mean_n_eff, 2),
        "n": n,
    }


def _composite_score(metrics: dict) -> float:
    """Higher is better. Combines -mae, coverage, -brier."""
    mae = metrics.get("mae", 10)
    cov = metrics.get("coverage_80", 0)
    brier = metrics.get("brier", 0.25)
    return -mae / 10 + cov - brier / 0.25


def _load_episodes():
    """Load calibration episodes."""
    try:
        from cerebro_calibration import _load_episodes as load_ep
        raw, _ = load_ep(score_threshold=2.0)
        return raw
    except Exception:
        return []


def _load_current_params() -> dict:
    """Current operational params (chimera_params or defaults)."""
    p = DATA_DIR / "chimera_params.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {"vel_weight": 100, "acc_weight": 2500, "tau": 1.0, "interval_alpha": 0.8}


def _load_arena_state() -> dict:
    if not ARENA_STATE_PATH.exists():
        return {}
    try:
        with open(ARENA_STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_arena_state(state: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(ARENA_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _promote_config(config: dict, metrics: dict):
    """Write promoted config to chimera_params.json. Use cerebro_chimera store if available."""
    out = {
        "vel_weight": int(config.get("vel_weight", 100)),
        "acc_weight": int(config.get("acc_weight", 2500)),
        "tau": round(float(config.get("tau", 1.0)), 3),
        "interval_alpha": round(float(config.get("interval_alpha", 0.8)), 2),
        "n_updates": 0,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "arena_promotion",
        "arena_metrics": metrics,
    }
    try:
        from cerebro_chimera import chimera_store
        chimera_store.atomic_write(DATA_DIR / "chimera_params.json", out)
        chimera_store.save_params_version(out)
    except Exception:
        with open(DATA_DIR / "chimera_params.json", "w") as f:
            json.dump(out, f, indent=2)


def _append_log(event: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def run_arena(
    candidates: list[dict] | None = None,
    trigger: str = "manual",
    infinity_score: float | None = None,
) -> dict:
    """
    Run parameter arena. Evaluate candidates, rank, promote if gates pass.
    Returns arena result dict.
    """
    episodes = _load_episodes()
    if len(episodes) < 10:
        return {"error": "Insufficient episodes", "n": len(episodes), "trigger": trigger}

    current = _load_current_params()
    current_metrics = _evaluate_config(episodes, current)
    current_metrics["composite"] = _composite_score(current_metrics)

    if candidates is None:
        # Default grid: current + neighbors
        vw = current.get("vel_weight", 100)
        aw = current.get("acc_weight", 2500)
        tau = current.get("tau", 1.0)
        alpha = current.get("interval_alpha", 0.8)
        candidates = [
            current,
            {"vel_weight": max(VW_BOUNDS[0], vw - 50), "acc_weight": aw, "tau": tau, "interval_alpha": alpha},
            {"vel_weight": min(VW_BOUNDS[1], vw + 50), "acc_weight": aw, "tau": tau, "interval_alpha": alpha},
            {"vel_weight": vw, "acc_weight": max(AW_BOUNDS[0], aw - 500), "tau": tau, "interval_alpha": alpha},
            {"vel_weight": vw, "acc_weight": min(AW_BOUNDS[1], aw + 500), "tau": tau, "interval_alpha": alpha},
            {"vel_weight": vw, "acc_weight": aw, "tau": max(TAU_BOUNDS[0], tau - 0.3), "interval_alpha": alpha},
            {"vel_weight": vw, "acc_weight": aw, "tau": min(TAU_BOUNDS[1], tau + 0.3), "interval_alpha": alpha},
            {"vel_weight": vw, "acc_weight": aw, "tau": tau, "interval_alpha": 0.75},
            {"vel_weight": vw, "acc_weight": aw, "tau": tau, "interval_alpha": 0.85},
        ]

    ranked = []
    for c in candidates:
        m = _evaluate_config(episodes, c)
        m["config"] = c
        m["composite"] = _composite_score(m)
        ranked.append(m)

    ranked.sort(key=lambda x: x["composite"], reverse=True)
    best = ranked[0]
    best_config = best["config"]
    best_metrics = {k: v for k, v in best.items() if k != "config"}

    # Integrity cap check
    integrity_cap = "HIGH"
    p = DATA_DIR / "integrity_scores.json"
    if p.exists():
        try:
            with open(p) as f:
                integrity_cap = json.load(f).get("confidence_cap", "HIGH")
        except Exception:
            pass

    promoted = False
    if (
        best["mean_n_eff"] >= MIN_N_EFF_PROMOTE
        and best["coverage_80"] >= MIN_COVERAGE_PROMOTE
        and integrity_cap != "LOW"
        and best["composite"] > current_metrics.get("composite", -999) * (1 + PROMOTE_THRESHOLD)
    ):
        # Store previous best before promoting (for rollback)
        state_pre = _load_arena_state()
        state_pre["previous_best_before_promotion"] = current
        _save_arena_state(state_pre)
        _promote_config(best_config, best_metrics)
        promoted = True
        _append_log({
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": "arena_promotion",
            "config": best_config,
            "metrics": best_metrics,
            "trigger": trigger,
        })

    # Update arena state for temporal consistency
    state = _load_arena_state()
    state["last_run"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state["last_metrics"] = best_metrics
    state["last_config"] = best_config
    if promoted:
        state["promoted_at"] = state["last_run"]
        state["promoted_config"] = best_config
        state["promoted_score"] = best["composite"]
        state["promoted_infinity_score"] = infinity_score  # For rollback guard
        state["runs_since_promotion"] = 0
    else:
        state["runs_since_promotion"] = state.get("runs_since_promotion", 0) + 1
    _save_arena_state(state)

    return {
        "trigger": trigger,
        "n_candidates": len(ranked),
        "best_config": best_config,
        "best_metrics": best_metrics,
        "current_metrics": _evaluate_config(episodes, current),
        "promoted": promoted,
        "ranked": [{"config": r["config"], "composite": r["composite"], "mae": r["mae"], "coverage_80": r["coverage_80"]} for r in ranked[:5]],
    }


def _record_run_score(infinity_score: float, promoted_this_run: bool = False):
    """Record this run's score and increment runs_since_promotion unless we promoted."""
    state = _load_arena_state()
    score_history = state.get("score_history", [])
    score_history.append(infinity_score)
    if len(score_history) > 10:
        score_history = score_history[-10:]
    state["score_history"] = score_history
    if not promoted_this_run and "promoted_at" in state:
        state["runs_since_promotion"] = state.get("runs_since_promotion", 0) + 1
    _save_arena_state(state)


def check_temporal_rollback(infinity_score: float) -> dict | None:
    """
    Temporal consistency guard. If promoted config degraded for 2 consecutive runs, rollback.
    Returns rollback config if rollback needed, else None.
    """
    state = _load_arena_state()
    if "promoted_config" not in state or "promoted_at" not in state:
        return None

    prev_best = state.get("previous_best_before_promotion")
    if prev_best is None:
        return None

    promoted_inf = state.get("promoted_infinity_score")
    if promoted_inf is None:
        promoted_inf = state.get("promoted_score", 0)  # Fallback to composite

    score_history = state.get("score_history", [])
    if len(score_history) >= ROLLBACK_DEGRADE_RUNS:
        recent = score_history[-ROLLBACK_DEGRADE_RUNS:]
        if all(s < promoted_inf - 15 for s in recent):
            return prev_best
    return None


def run_rollback_if_needed(infinity_score: float) -> bool:
    """Check temporal guard and rollback if needed. Returns True if rollback performed."""
    prev = check_temporal_rollback(infinity_score)
    if prev is None:
        return False
    _promote_config(prev, {"rollback": True})
    _append_log({
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": "arena_rollback",
        "reason": "temporal_consistency",
        "restored_config": prev,
    })
    state = _load_arena_state()
    state["promoted_config"] = None
    state["promoted_at"] = None
    state["runs_since_promotion"] = 0
    _save_arena_state(state)
    return True
