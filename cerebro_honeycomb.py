#!/usr/bin/env python3
"""
CEREBRO HONEYCOMB — Ensemble arbitration (Core + Sister + Sim + Shift)
======================================================================
Fuses core analogue peak, sister regression peak, forward simulation, and distribution shift.
"""

import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "honeycomb_latest.json"
MIN_TRAIN = 5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _load_forward_sim() -> dict:
    p = SCRIPT_DIR / "cerebro_data" / "forward_simulation.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_distribution_shift() -> dict:
    p = SCRIPT_DIR / "cerebro_data" / "distribution_shift.json"
    if not p.exists():
        return {"percentile": 0.5, "confidence_modifier": 1.0}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {"percentile": 0.5, "confidence_modifier": 1.0}


def _sim_q25_q75(sim: dict) -> tuple[float, float]:
    """Compute q25, q75 from sim distribution (time-to-event years)."""
    dist = sim.get("distribution", {})
    if not dist:
        med = sim.get("median_time_to_event", 5)
        return max(1, med - 2), min(15, med + 2)
    # dist is { "1": 0.04, "2": 0.08, ... } — fraction at each year
    years = sorted([int(k) for k in dist.keys() if k.isdigit()])
    if not years:
        return 3.0, 8.0
    probs = [float(dist.get(str(y), 0)) for y in years]
    cum = np.cumsum(probs)
    q25_yr = years[0]
    q75_yr = years[-1]
    for i, c in enumerate(cum):
        if c >= 0.25:
            q25_yr = years[i]
            break
    for i, c in enumerate(cum):
        if c >= 0.75:
            q75_yr = years[i]
            break
    return float(q25_yr), float(q75_yr)


def _load_learned_weights() -> tuple[float | None, float | None]:
    """Return (vel_weight, acc_weight) from distance_weights.json or (None, None)."""
    p = SCRIPT_DIR / "cerebro_data" / "distance_weights.json"
    if not p.exists():
        return None, None
    try:
        with open(p) as f:
            d = json.load(f)
        return float(d.get("vel_weight")), float(d.get("acc_weight"))
    except Exception:
        return None, None


def _load_conformal() -> dict | None:
    """Load honeycomb_conformal.json or None."""
    p = SCRIPT_DIR / "cerebro_data" / "honeycomb_conformal.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def compute_honeycomb_fusion(
    now_year: int,
    pos: float,
    vel: float,
    acc: float,
    pool: list,
    rb: float | None,
    sim_summary: dict | None = None,
    shift_dict: dict | None = None,
    apply_learned_weights: bool = True,
    apply_conformal: bool = True,
) -> dict:
    """Fuse core, sister, sim, shift. sim_summary and shift_dict override file loads."""
    from cerebro_peak_window import compute_peak_window
    from cerebro_sister_engine import sister_predict

    if len(pool) < MIN_TRAIN:
        return {"error": "Insufficient past", "peak_year": now_year + 5, "window_start": now_year + 3, "window_end": now_year + 10, "confidence_pct": 50}

    vw, aw = (None, None)
    if apply_learned_weights:
        vw, aw = _load_learned_weights()
    core = compute_peak_window(now_year, pos, vel, acc, rb, pool, interval_alpha=0.8, vel_weight=vw, acc_weight=aw)
    sis = sister_predict(now_year, pos, vel, acc, pool)

    core_peak = core["peak_year"]
    core_ws = core["window_start"]
    core_we = core["window_end"]
    core_conf = core.get("confidence_pct", 70)

    sis_peak = sis["peak_year"]
    sis_ws = sis["window_start"]
    sis_we = sis["window_end"]
    sis_conf = sis.get("confidence_pct", 70)

    sim = sim_summary if sim_summary is not None else _load_forward_sim()
    if sim.get("error"):
        sim_peak = now_year + 5
        sim_ws = now_year + 3
        sim_we = now_year + 8
        sim_conf = 50
    else:
        med = sim.get("median_time_to_event", 5)
        q25, q75 = _sim_q25_q75(sim)
        sim_peak = now_year + int(round(med))
        sim_ws = now_year + int(round(q25))
        sim_we = now_year + int(round(q75))
        sim_conf = 70

    shift = shift_dict if shift_dict is not None else _load_distribution_shift()
    shift_mod = shift.get("confidence_modifier", 1.0)

    w_core = _clamp(core_conf / 100.0, 0.4, 0.9)
    w_sis = _clamp(sis_conf / 100.0, 0.2, 0.8)
    w_sim = 0.3
    total = w_core + w_sis + w_sim
    w_core /= total
    w_sis /= total
    w_sim /= total

    peak_float = w_core * core_peak + w_sis * sis_peak + w_sim * sim_peak
    peak_year = int(round(peak_float))

    ws_float = w_core * core_ws + w_sis * sis_ws + w_sim * sim_ws
    we_float = w_core * core_we + w_sis * sis_we + w_sim * sim_we
    window_start = int(round(ws_float))
    window_end = int(round(we_float))

    if window_start >= window_end:
        window_start = peak_year - 2
        window_end = peak_year + 2
    if window_start > peak_year:
        window_start = peak_year - 1
    if window_end < peak_year:
        window_end = peak_year + 1
    width = window_end - window_start
    if width < 3:
        half = (3 - width) // 2
        window_start = max(now_year, window_start - half - (3 - width - 2 * half))
        window_end = window_start + 3
    if width > 15:
        window_start = peak_year - 7
        window_end = peak_year + 8

    base_conf = w_core * core_conf + w_sis * sis_conf + w_sim * sim_conf
    base_conf *= shift_mod
    disp = float(np.std([core_peak, sis_peak, sim_peak]))
    base_conf -= min(25, int(disp * 6))
    confidence_pct = int(_clamp(base_conf, 40, 95))

    out = {
        "peak_year": peak_year,
        "window_start": window_start,
        "window_end": window_end,
        "confidence_pct": confidence_pct,
        "method": "honeycomb_ensemble",
        "components": {
            "core": {"peak_year": core_peak, "window_start": core_ws, "window_end": core_we, "confidence_pct": core_conf},
            "sister": {"peak_year": sis_peak, "window_start": sis_ws, "window_end": sis_we, "confidence_pct": sis_conf},
            "simulation": {"peak_year": sim_peak, "window_start": sim_ws, "window_end": sim_we},
            "shift": {"percentile": shift.get("percentile", 0.5), "confidence_modifier": shift_mod},
        },
        "disagreement_std_years": round(disp, 2),
    }

    if apply_conformal:
        cal = _load_conformal()
        if cal and cal.get("s_hat") is not None:
            s_hat = float(cal["s_hat"])
            out["window_start"] = int(window_start - s_hat)
            out["window_end"] = int(window_end + s_hat)
            out["window_label"] = "80% calibrated (honeycomb)"
            out["conformal_applied"] = True
            out["conformal_s_hat"] = round(s_hat, 4)
            out["target_coverage"] = cal.get("target_coverage", 0.8)
            if out["window_start"] > peak_year:
                out["window_start"] = peak_year - 1
            if out["window_end"] < peak_year:
                out["window_end"] = peak_year + 1
        else:
            out.setdefault("conformal_applied", False)
    else:
        out.setdefault("conformal_applied", False)

    return out


def run_honeycomb() -> dict:
    """Fuse core, sister, sim, shift into honeycomb verdict."""
    from cerebro_calibration import _load_episodes
    from cerebro_eval_utils import past_only_pool

    episodes, _ = _load_episodes(score_threshold=2.0)
    if len(episodes) < MIN_TRAIN + 2:
        return {"error": "Insufficient episodes", "peak_year": 0, "window_start": 0, "window_end": 0, "confidence_pct": 50}

    latest = max(episodes, key=lambda e: e.get("saddle_year", 0))
    pool = past_only_pool(episodes, latest["saddle_year"])
    if len(pool) < MIN_TRAIN:
        return {"error": "Insufficient past", "peak_year": 0, "window_start": 0, "window_end": 0, "confidence_pct": 50}

    return compute_honeycomb_fusion(
        latest["saddle_year"],
        latest.get("position", 0),
        latest.get("velocity", 0),
        latest.get("acceleration", 0),
        pool,
        latest.get("ring_B_score"),
    )


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_honeycomb()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"Honeycomb: peak={r.get('peak_year')}, conf={r.get('confidence_pct')}% → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
