#!/usr/bin/env python3
"""
CEREBRO HONEYCOMB — Ensemble arbitration (Core + Sister + Sim + Shift)
======================================================================
Fuses core analogue peak, sister regression peak, forward simulation, and distribution shift.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "cerebro_data" / "honeycomb_latest.json"
WEIGHTS_PATH = SCRIPT_DIR / "cerebro_data" / "honeycomb_weights.json"
MIN_TRAIN = 5
BACKTEST_SIM_RUNS = 50


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
    """Return (vel_weight, acc_weight). Prefer chimera_params over distance_weights when learned."""
    # CHIMERA adaptive: prefer chimera_params when it has updates
    cp = SCRIPT_DIR / "cerebro_data" / "chimera_params.json"
    if cp.exists():
        try:
            with open(cp) as f:
                d = json.load(f)
            if d.get("n_updates", 0) > 0:
                vw, aw = d.get("vel_weight"), d.get("acc_weight")
                if vw is not None and aw is not None:
                    return float(vw), float(aw)
        except Exception:
            pass
    # Fallback: distance_weights.json
    p = SCRIPT_DIR / "cerebro_data" / "distance_weights.json"
    if not p.exists():
        return None, None
    try:
        with open(p) as f:
            d = json.load(f)
        return float(d.get("vel_weight")), float(d.get("acc_weight"))
    except Exception:
        return None, None


def compute_sister_trust(episode_context: dict) -> float:
    """
    Trust score 0.0–1.0 for sister engine. Sister is better post-1990, unreliable in high volatility.
    episode_context: {now_year, pool, pos?, vel?, acc?}
    """
    now_year = episode_context.get("now_year", 2000)
    pool = episode_context.get("pool", [])

    # era_factor: 1.0 post-1990, 0.6 pre-1990
    era_factor = 1.0 if now_year >= 1990 else 0.6

    # volatility_factor: 0.3 if vol_norm > 0.6, else 1.0
    vol_norm = 0.0
    try:
        from chimera.chimera_volatility import compute_volatility_index
        sorted_pool = sorted(pool, key=lambda e: e.get("saddle_year", 0))
        pos_list = [float(e.get("position", 0)) for e in sorted_pool]
        vel_list = [float(e.get("velocity", 0)) for e in sorted_pool]
        acc_list = [float(e.get("acceleration", 0)) for e in sorted_pool]
        if episode_context.get("pos") is not None:
            pos_list.append(float(episode_context["pos"]))
            vel_list.append(float(episode_context.get("vel", 0)))
            acc_list.append(float(episode_context.get("acc", 0)))
        vol_norm = compute_volatility_index(pos_list, vel_list, acc_list)
    except Exception:
        vol_norm = 0.0
    volatility_factor = 0.3 if vol_norm > 0.6 else 1.0

    # n_post1990_factor: 1.0 if n_post1990 >= 10, else 0.5
    post1990 = [e for e in pool if e.get("event_year", 0) >= 1990]
    n_post1990_factor = 1.0 if len(post1990) >= 10 else 0.5

    trust_score = era_factor * volatility_factor * n_post1990_factor
    return round(min(1.0, max(0.0, trust_score)), 4)


def _chimera_safety_gate() -> bool:
    """True if any gate triggers: failure>0.8, OOD severe, coverage<0.6, stability<0.5."""
    try:
        for name, check in [
            ("chimera_failure.json", lambda j: j.get("severity", 0) > 0.8),
            ("distribution_shift.json", lambda j: j.get("ood_level") == "SEVERE"),
            ("chimera_reconstruction.json", lambda j: j.get("coverage_80_mean") is not None and j.get("coverage_80_mean") < 0.6),
            ("chimera_stress_matrix.json", lambda j: j.get("mean_stability") is not None and j.get("mean_stability") < 0.5),
        ]:
            p = SCRIPT_DIR / "cerebro_data" / name
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                if check(d):
                    return True
    except Exception:
        pass
    return False


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


def compute_component_mae_and_weights(
    episodes: list,
    min_train: int = 5,
) -> dict:
    """
    Walk-forward MAE per component (core, sister, simulation). Returns inverse-MAE weights.
    Uses per-country pooling when episodes have 'country' key. Logs MAE for diagnostics.
    """
    from cerebro_eval_utils import past_only_pool
    from cerebro_peak_window import compute_peak_window
    from cerebro_sister_engine import sister_predict
    from cerebro_forward_simulation import run_forward_simulation

    by_country = {}
    for ep in episodes:
        c = ep.get("country", "US")
        by_country.setdefault(c, []).append(ep)

    core_mae_list, sister_mae_list, sim_mae_list = [], [], []

    for country, country_eps in by_country.items():
        if len(country_eps) < min_train + 2:
            continue
        sorted_ep = sorted(country_eps, key=lambda e: e.get("saddle_year", 0))
        for ep in sorted_ep:
            t = ep.get("saddle_year")
            if t is None:
                continue
            pool = past_only_pool(country_eps, t)
            if len(pool) < min_train:
                continue
            event_yr = ep.get("event_year", t + 5)
            try:
                core = compute_peak_window(
                    t, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0),
                    ep.get("ring_B_score"), pool, interval_alpha=0.8,
                )
                sis = sister_predict(t, ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0), pool)
                initial = (ep.get("position", 0), ep.get("velocity", 0), ep.get("acceleration", 0))
                sim = run_forward_simulation(
                    initial_state=initial, pool=pool, now_year=t, n_runs=BACKTEST_SIM_RUNS,
                )
            except Exception:
                continue

            core_mae_list.append(abs(core["peak_year"] - event_yr))
            sister_mae_list.append(abs(sis["peak_year"] - event_yr))
            if sim.get("error"):
                sim_peak = t + 5
            else:
                sim_peak = t + int(round(sim.get("median_time_to_event", 5)))
            sim_mae_list.append(abs(sim_peak - event_yr))

    n = len(core_mae_list)
    if n < 5:
        return {"core_mae": None, "sister_mae": None, "sim_mae": None, "weights": None, "n_episodes": n}

    def mean(x):
        return sum(x) / len(x) if x else 1e-6

    core_mae = mean(core_mae_list)
    sister_mae = mean(sister_mae_list)
    sim_mae = mean(sim_mae_list)

    # Inverse-MAE: weight_i = 1/MAE_i, normalized
    inv_core = 1.0 / max(core_mae, 0.1)
    inv_sis = 1.0 / max(sister_mae, 0.1)
    inv_sim = 1.0 / max(sim_mae, 0.1)
    total = inv_core + inv_sis + inv_sim
    w_core = inv_core / total
    w_sis = inv_sis / total
    w_sim = inv_sim / total

    result = {
        "core_mae": round(core_mae, 4),
        "sister_mae": round(sister_mae, 4),
        "sim_mae": round(sim_mae, 4),
        "weights": {"core": round(w_core, 4), "sister": round(w_sis, 4), "simulation": round(w_sim, 4)},
        "n_episodes": n,
        "worst_component": max(
            [("core", core_mae), ("sister", sister_mae), ("simulation", sim_mae)],
            key=lambda x: x[1],
        )[0],
    }
    print(f"[Honeycomb] Component MAE (n={n}): core={core_mae:.2f}, sister={sister_mae:.2f}, sim={sim_mae:.2f} → worst={result['worst_component']}")
    print(f"[Honeycomb] Inverse-MAE weights: core={w_core:.3f}, sister={w_sis:.3f}, sim={w_sim:.3f}")

    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    return result


def _load_current_clock_velocities(now_year: int, harm_vel: float, harm_pos: float, harm_acc: float) -> dict:
    """Load current velocities for harm, sexual, class, evil for coupling correction."""
    state = {
        "harm": {"velocity": harm_vel, "position": harm_pos, "acceleration": harm_acc},
        "sexual": {"velocity": 0.0, "position": 0.0, "acceleration": 0.0},
        "class": {"velocity": 0.0, "position": 0.0, "acceleration": 0.0},
        "evil": {"velocity": 0.0, "position": 0.0, "acceleration": 0.0},
    }
    for name, path in [
        ("sexual", SCRIPT_DIR / "cerebro_sexual_clock_data.csv"),
        ("class", SCRIPT_DIR / "cerebro_class_clock_data.csv"),
    ]:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if "year" in df.columns and len(df) >= 5:
                    df = df[df["year"] <= now_year].tail(1)
                    if not df.empty:
                        row = df.iloc[-1]
                        pos_col = "clock_position_10pt" if "clock_position_10pt" in df.columns else "position"
                        vel_col = "velocity"
                        acc_col = "acceleration"
                        state[name]["position"] = float(row.get(pos_col, 0))
                        state[name]["velocity"] = float(row.get(vel_col, 0)) if pd.notna(row.get(vel_col)) else 0.0
                        state[name]["acceleration"] = float(row.get(acc_col, 0)) if pd.notna(row.get(acc_col)) else 0.0
            except Exception:
                pass
    return state


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
    component_weights: dict | None = None,
    use_spillover_coupling: bool = False,
    use_coupling: bool = False,
    raw_window_width: int | float | None = None,
) -> dict:
    """Fuse core, sister, sim, shift. sim_summary and shift_dict override file loads.
    component_weights: optional {core, sister, simulation} for inverse-MAE weighting.
    use_spillover_coupling: if True, apply coupling correction from spillover (dampening 0.15).
    NOTE: Spillover currently returns zeros; coupling is no-op until multi-dimension data exists."""
    from cerebro_peak_window import compute_peak_window
    from cerebro_sister_engine import sister_predict

    if len(pool) < MIN_TRAIN:
        # Apply wide-interval boost even on early return
        conf = 50
        raw_w = raw_window_width if raw_window_width is not None else 0
        if raw_w >= 5.0 and (rb is None or rb >= 0):
            conf = 68
        return {
            "error": "Insufficient past",
            "peak_year": now_year + 5,
            "window_start": now_year + 3,
            "window_end": now_year + 10,
            "confidence_pct": conf
        }

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

    # Conditional sister fusion: trust-based weight
    episode_context = {"now_year": now_year, "pool": pool, "pos": pos, "vel": vel, "acc": acc}
    trust_score = compute_sister_trust(episode_context)
    base_sister_weight = 0.15
    effective_sister_weight = base_sister_weight * trust_score if trust_score >= 0.4 else 0.0
    fusion_mode = "excluded" if trust_score < 0.4 else "conditional"

    # Sister bias correction: applied in sister_predict before return (adds +1 when post-1990, n_post1990>=10)
    # sis_peak is already corrected; use directly for fusion.

    if component_weights and "core" in component_weights and "sister" in component_weights and "simulation" in component_weights:
        base_core = float(component_weights["core"])
        base_sim = float(component_weights["simulation"])
        # Scale core+sim to sum to 0.85 so base_sister=0.15 gives total 1.0
        s = base_core + base_sim
        if s > 1e-6:
            base_core = base_core * 0.85 / s
            base_sim = base_sim * 0.85 / s
        else:
            base_core, base_sim = 0.5, 0.35
    else:
        base_core, base_sim = 0.5, 0.35  # sum 0.85, base_sister 0.15

    # Redistribute removed sister weight to core and sim proportionally (by inverse-MAE)
    removed = base_sister_weight - effective_sister_weight
    if removed > 1e-6 and (base_core + base_sim) > 1e-6:
        core_frac = base_core / (base_core + base_sim)
        w_core = base_core + removed * core_frac
        w_sim = base_sim + removed * (1 - core_frac)
    else:
        w_core = base_core + removed
        w_sim = base_sim
    w_sis = effective_sister_weight
    total = w_core + w_sis + w_sim
    if total > 1e-6:
        w_core /= total
        w_sis /= total
        w_sim /= total

    peak_float = w_core * core_peak + w_sis * sis_peak + w_sim * sim_peak
    coupling_corrections = {}
    # Coupled state vector: apply clock coupling correction when use_coupling
    if use_coupling:
        try:
            from cerebro_coupling_matrix import load_coupling_matrix, compute_coupling_correction
            coupling = load_coupling_matrix()
            if coupling:
                current_state = _load_current_clock_velocities(now_year, vel, pos, acc)
                coupling_corrections = compute_coupling_correction(current_state, coupling, dampening=0.10)
                harm_corr = coupling_corrections.get("harm", 0.0)
                peak_float += harm_corr
        except Exception:
            pass
    # Task 3: Coupling correction (dampening 0.15) when use_spillover_coupling and coeffs available
    if use_spillover_coupling:
        try:
            from cerebro_spillover import get_coupling_coefficients
            coeffs = get_coupling_coefficients()
            if coeffs:
                damp = 0.15
                # Velocities: time-to-peak deltas (core_peak-now, etc). Single-dimension: vel as proxy.
                v_core = (core_peak - now_year) / 5.0
                v_sis = (sis_peak - now_year) / 5.0
                v_sim = (sim_peak - now_year) / 5.0
                # Coupling from other dims (simplified: core/sis/sim as 3 of 4 dims)
                adj = damp * (coeffs.get("harm_tolerance", {}).get("sexual_norms", 0) * v_sis +
                              coeffs.get("harm_tolerance", {}).get("class_permeability", 0) * 0 +
                              coeffs.get("harm_tolerance", {}).get("good_vs_evil", 0) * 0)
                peak_float += adj
        except Exception:
            pass
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

    core_interval_width = (core_we - core_ws) if (core_ws is not None and core_we is not None) else 0
    fused_interval_width = (window_end - window_start) if (window_start is not None and window_end is not None) else 0

    base_conf = w_core * core_conf + w_sis * sis_conf + w_sim * sim_conf
    base_conf *= shift_mod
    disp = float(np.std([core_peak, sis_peak, sim_peak]))
    disagreement_penalty = min(25, int(disp * 6))
    fused_window_width = fused_interval_width if fused_interval_width is not None else 0
    check_width = max(
        raw_window_width if raw_window_width is not None else 0,
        fused_window_width if fused_window_width is not None else 0,
    )
    if check_width < 5.0:
        base_conf -= disagreement_penalty
    confidence_pct = int(_clamp(base_conf, 40, 95))

    # Hard safety gate (CHIMERA): clamp to 55 if structural instability
    if _chimera_safety_gate():
        confidence_pct = min(55, confidence_pct)

    # Wide-interval boost: AFTER fusion — use fused window width (7), not core (4)
    if check_width >= 5.0 and confidence_pct < 72 and (rb is None or rb >= 0):
        confidence_pct = min(78, confidence_pct + 18)

    out = {
        "peak_year": peak_year,
        "window_start": window_start,
        "window_end": window_end,
        "confidence_pct": confidence_pct,
        "core_interval_width": core_interval_width,
        "fused_interval_width": fused_interval_width,
        "method": "honeycomb_ensemble",
        "sister_trust_score": round(trust_score, 4),
        "effective_sister_weight": round(effective_sister_weight, 4),
        "fusion_mode": fusion_mode,
        "components": {
            "core": {"peak_year": core_peak, "window_start": core_ws, "window_end": core_we, "confidence_pct": core_conf},
            "sister": {"peak_year": sis_peak, "window_start": sis_ws, "window_end": sis_we, "confidence_pct": sis_conf},
            "simulation": {"peak_year": sim_peak, "window_start": sim_ws, "window_end": sim_we},
            "shift": {"percentile": shift.get("percentile", 0.5), "confidence_modifier": shift_mod},
        },
        "disagreement_std_years": round(disp, 2),
    }
    if coupling_corrections:
        out["coupling_corrections"] = coupling_corrections
        out["use_coupling"] = True

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
        use_coupling=False,  # Infrastructure ready; enable when more per-country data or stratified coupling
    )


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    r = run_honeycomb()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(r, f, indent=2)
    # Register prediction for live feedback loop
    if "error" not in r:
        try:
            from cerebro_live_feedback import register_prediction
            from cerebro_calibration import _load_episodes
            from cerebro_eval_utils import past_only_pool
            episodes, _ = _load_episodes(score_threshold=2.0)
            if episodes:
                latest = max(episodes, key=lambda e: e.get("saddle_year", 0))
                register_prediction(
                    "harm",
                    r.get("window_start", 0),
                    r.get("window_end", 0),
                    r.get("confidence_pct", 70),
                    r.get("peak_year", 0),
                    saddle_year=latest.get("saddle_year"),
                    position=latest.get("position"),
                    velocity=latest.get("velocity"),
                    acceleration=latest.get("acceleration"),
                    ring_B_score=latest.get("ring_B_score"),
                    country=latest.get("country", "US"),
                )
        except Exception:
            pass
    print(f"Honeycomb: peak={r.get('peak_year')}, conf={r.get('confidence_pct')}% → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
