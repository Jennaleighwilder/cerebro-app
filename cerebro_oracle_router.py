#!/usr/bin/env python3
"""
CEREBRO ORACLE ROUTER — Route queries to live Cerebro engine
============================================================
Maps natural-language queries to engine calls and formats responses.
"""

import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
PUBLIC_DIR = SCRIPT_DIR / "public"


def _classify_intent(q: str) -> str | None:
    """Server-side matchQuery: classify query intent."""
    l = q.lower()
    if "exact rule" in l or "exact mathematical" in l or "mathematical rule" in l or "formula" in l or "equation" in l:
        return "METHOD"
    if ("how many" in l and ("saddle" in l or "event" in l or "historical" in l)) or "backtest" in l or "average forecast error" in l or "forecast error" in l or "mae" in l:
        return "VALIDATION"
    if "resemble" in l or "historical era" in l or "era does" in l:
        return "era"
    if "when does" in l or "when will" in l or "peak" in l or "heading into" in l or "era" in l or "explode" in l or "redistrib" in l:
        return "FORECAST"
    if "unequal" in l or ("top 10" in l and "countr" in l) or ("gini" in l and "rank" in l):
        return "country_unequal"
    if "country risk" in l or "political violence" in l or ("risk" in l and "2030" in l) or "class-driven" in l:
        return "country_risk"
    if ("apogee" in l or "historical peak" in l) and ("compare" in l or ("sexual" in l and "harm" in l)):
        return "apogee_compare"
    if ("apogee" in l or "all four" in l or "show" in l) and ("apogee" in l or "four" in l):
        return "apogee_multi"
    if ("cross" in l and "zero" in l) or "when will class" in l:
        return "apogee_crossing"
    if "loaded" in l or "how loaded" in l or "system right now" in l or "pressure" in l:
        return "loaded"
    if "confidence" in l or "accurate" in l or "forecast" in l and "confidence" in l:
        return "confidence"
    if "cultural velocity" in l or "defund" in l or "back the blue" in l or "trends" in l:
        return "trends"
    if "punitive" in l or "tough" in l or "crime" in l:
        return "punitive"
    if "wealth" in l or "redistrib" in l or "class" in l:
        return "wealth"
    if "sexual" in l or "backlash" in l:
        return "sexual"
    return "loaded"


def _load_cerebro_data() -> dict:
    """Load cerebro_data.json (or cerebro_offline.json)."""
    for p in [PUBLIC_DIR / "cerebro_data.json", PUBLIC_DIR / "cerebro_offline.json", DATA_DIR / "cerebro_data.json"]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _load_latest_harm_state() -> tuple[int, float, float, float, float | None]:
    """Load latest year, position, velocity, acceleration, ring_B from harm clock CSV."""
    csv_path = SCRIPT_DIR / "cerebro_harm_clock_data.csv"
    if not csv_path.exists():
        return 2022, 0.77, 0.11, 0.07, None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0)
        df = df[df["clock_position_10pt"].notna()].tail(1)
        if df.empty:
            return 2022, 0.77, 0.11, 0.07, None
        row = df.iloc[-1]
        yr = int(df.index[-1])
        pos = float(row["clock_position_10pt"])
        vel = float(row["velocity"]) if pd.notna(row.get("velocity")) else 0.11
        acc = float(row["acceleration"]) if pd.notna(row.get("acceleration")) else 0.07
        rb = row.get("ring_B_score")
        rb = float(rb) if rb is not None and pd.notna(rb) else None
        return yr, pos, vel, acc, rb
    except Exception:
        return 2022, 0.77, 0.11, 0.07, None


def _compute_peak_window_live() -> dict | None:
    """Call compute_peak_window with latest state. Returns peak_window dict or None."""
    try:
        from cerebro_peak_window import compute_peak_window
        now_year, pos, vel, acc, rb = _load_latest_harm_state()
        pw = compute_peak_window(now_year, pos, vel, acc, rb, apply_conformal=False)
        return pw
    except Exception:
        return None


def _format_peak_window_response(pw: dict, clock_name: str = "Harm Tolerance") -> str:
    """Format peak window as natural language."""
    if not pw:
        return f"The {clock_name} clock could not compute a peak window. Check data availability."
    ws = pw.get("window_start", "—")
    we = pw.get("window_end", "—")
    conf = pw.get("confidence_pct", 70)
    ana = pw.get("analogue_count", 0)
    best = pw.get("best_analogue_year") or pw.get("analogues", [[None, None]])[0][0] if pw.get("analogues") else None
    sim = pw.get("analogue_similarity") or (pw.get("analogues", [[None, None]])[0][1] if pw.get("analogues") else None)
    parts = [f"The {clock_name} clock forecasts peak pressure between {ws} and {we}, with {conf}% confidence."]
    if ana:
        parts.append(f"Based on {ana} historical analogue episodes.")
    if best:
        sim_str = f" at {sim}% similarity" if sim is not None else ""
        parts.append(f"Nearest historical analogue: {best}{sim_str}.")
    return " ".join(parts)


def _handle_forecast(query: str) -> tuple[str, dict, float]:
    """Handle peak/timing queries — call compute_peak_window live."""
    pw = _compute_peak_window_live()
    if pw:
        answer = _format_peak_window_response(pw, "Class Permeability" if "wealth" in query.lower() or "redistrib" in query.lower() else "Harm Tolerance")
        data = {"peak_year": pw.get("peak_year"), "window_start": pw.get("window_start"), "window_end": pw.get("window_end"), "confidence_pct": pw.get("confidence_pct"), "analogue_count": pw.get("analogue_count")}
        return answer, data, pw.get("confidence_pct", 70) / 100.0
    d = _load_cerebro_data()
    md = d.get("method_data", {})
    pw = md.get("peak_window", {})
    if pw:
        answer = _format_peak_window_response(pw)
        return answer, pw, pw.get("confidence_pct", 70) / 100.0
    return "Peak window data not available. Run cerebro_calibration.py and cerebro_export_ui_data.py.", {}, 0.5


def _handle_loaded(query: str) -> tuple[str, dict, float]:
    """Handle system state queries — read from cerebro_data, optionally recompute if stale."""
    d = _load_cerebro_data()
    h = d.get("harm_clock", {})
    sys = d.get("system", {})
    pos = h.get("position", 0.77)
    vel = h.get("velocity", 0.11)
    acc = h.get("acceleration", 0.07)
    rb = h.get("ring_B_pct")
    conf = sys.get("confidence_pct", 85) or h.get("confidence", 85)
    load = sys.get("system_load_pct", 73)
    answer = f"Harm clock at {pos:+.2f}/10 (velocity {vel:+.2f}, acceleration {acc:+.2f}). "
    answer += f"System load {load}%. "
    if rb:
        answer += f"Ring B loaded ({rb}%). "
    answer += f"Confidence {conf}%."
    return answer, {"position": pos, "velocity": vel, "system_load_pct": load, "confidence_pct": conf}, conf / 100.0


def _handle_era(query: str) -> tuple[str, dict, float]:
    """Handle historical analogue queries."""
    d = _load_cerebro_data()
    h = d.get("harm_clock", {})
    analogues = h.get("analogues", [[1934, 87], [1997, 72], [1968, 58]])
    if analogues:
        best_yr, best_sim = analogues[0]
        answer = f"Best historical match: {best_yr} at {best_sim}% similarity. "
        if len(analogues) > 1:
            others = ", ".join(f"{yr} ({sim}%)" for yr, sim in analogues[1:3])
            answer += f"Other close analogues: {others}."
        return answer, {"analogues": analogues}, 0.85
    return "Analogue data not available.", {}, 0.5


def _handle_confidence(query: str) -> tuple[str, dict, float]:
    """Handle confidence/validation queries — call compute_infinity_score live."""
    try:
        from chimera.chimera_infinity_score import compute_infinity_score
        s = compute_infinity_score()
        diag = s.get("diagnostics", {})
        brier = diag.get("brier", 0.2)
        ece = diag.get("ece", 0.07)
        n_used = diag.get("n_used", 0)
        inf = s.get("infinity_score", 0)
        answer = f"Current forecast confidence: Brier {brier:.3f}, ECE {ece:.3f}. "
        answer += f"Calibration uses {n_used} episodes. Infinity Score {inf:.1f}."
        return answer, {"brier": brier, "ece": ece, "n_used": n_used, "infinity_score": inf}, 0.85
    except Exception:
        d = _load_cerebro_data()
        cal = d.get("calibration_curve", d.get("mode_operational", {}))
        brier = cal.get("brier", cal.get("brier_score", 0.2))
        n = cal.get("n_used", 0)
        return f"Brier {brier:.3f}, {n} episodes used in calibration.", {"brier": brier, "n_used": n}, 0.7


def _handle_country_risk(query: str) -> tuple[str, dict, float]:
    """Handle country risk queries."""
    d = _load_cerebro_data()
    cr = d.get("country_risk", {})
    lst = cr.get("top_10_risk", cr.get("top_10_unequal", []))
    if not lst:
        return "Country risk data not yet loaded. Run cerebro_risk_engine.py.", {}, 0.4
    names = {"TUR": "Turkey", "MDG": "Madagascar", "STP": "São Tomé", "DEU": "Germany", "USA": "United States"}
    lines = []
    for i, r in enumerate(lst[:5], 1):
        iso = r.get("iso", "?")
        n = names.get(iso, iso)
        risk = r.get("risk_score", 0)
        prob = r.get("probability_2030", "?")
        lines.append(f"{i}. {n} ({iso}): risk {risk:.0f}, P(2030)={prob}")
    answer = "Top country risks (class-driven political violence before 2030):\n" + "\n".join(lines)
    return answer, {"top_risk": lst[:5]}, 0.7


def _handle_method(query: str) -> tuple[str, dict, float]:
    """Handle method/rule queries."""
    try:
        from cerebro_peak_window import get_method_equations
        eq = get_method_equations()
        prov = eq.get("provenance", {})
        answer = f"Saddle rule: {eq.get('saddle_rule', '')} "
        answer += f"Peak window: {eq.get('peak_window_rule', '')} "
        answer += f"V_THRESH={prov.get('v_thresh', 0.15)}, distance weights learned."
        return answer, eq, 0.9
    except Exception:
        return "Method data not available.", {}, 0.5


def _handle_validation(query: str) -> tuple[str, dict, float]:
    """Handle validation/backtest queries."""
    bt_path = DATA_DIR / "backtest_metrics.json"
    cal_path = DATA_DIR / "calibration_curve.json"
    data = {}
    try:
        if bt_path.exists():
            with open(bt_path) as f:
                data = json.load(f)
    except Exception:
        pass
    try:
        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            mo = cal.get("mode_operational", cal)
            data["n_used"] = mo.get("n_used", 0)
            data["brier"] = mo.get("brier", mo.get("brier_score"))
            data["coverage_80"] = mo.get("coverage_80")
    except Exception:
        pass
    n = data.get("n_saddles_tested", data.get("n_used", 0))
    mae = data.get("mae_years", data.get("mae_walkforward"))
    brier = data.get("brier", 0.2)
    cov = data.get("coverage_80")
    answer = f"N episodes tested: {n}. "
    if mae is not None:
        answer += f"MAE: {mae:.2f} years. "
    answer += f"Brier: {brier:.3f}. "
    if cov is not None:
        answer += f"80% coverage: {cov:.1%}."
    return answer, data, 0.85


def _handle_trends(query: str) -> tuple[str, dict, float]:
    """Handle L1/cultural velocity queries."""
    d = _load_cerebro_data()
    cv = d.get("cultural_velocity")
    cvC = d.get("class_velocity")
    if cv:
        v = cv.get("cultural_velocity_smooth", 0)
        dir_ = "toward reform" if v > 0 else "toward punitive"
        answer = f"L1 Harm cultural velocity: {v:+.2f} ({dir_}). "
        if cvC:
            vc = cvC.get("velocity_smooth", 0)
            answer += f"L1 Class: {vc:+.2f}."
        return answer, {"cultural_velocity": cv, "class_velocity": cvC}, 0.75
    return "L1 Google Trends layer not yet populated. Run cerebro_trends_loader.py.", {}, 0.4


def route_query(query: str) -> dict:
    """Route query to handler and return {answer, data, confidence, intent}."""
    intent = _classify_intent(query)
    handlers = {
        "FORECAST": _handle_forecast,
        "wealth": _handle_forecast,
        "loaded": _handle_loaded,
        "era": _handle_era,
        "confidence": _handle_confidence,
        "country_risk": _handle_country_risk,
        "country_unequal": _handle_country_risk,
        "METHOD": _handle_method,
        "VALIDATION": _handle_validation,
        "trends": _handle_trends,
        "punitive": lambda q: _handle_forecast(q) if "peak" in q.lower() else _handle_loaded(q),
        "sexual": _handle_loaded,
    }
    handler = handlers.get(intent, _handle_loaded)
    try:
        result = handler(query)
        if isinstance(result, tuple) and len(result) >= 2:
            answer, data, conf = result[0], result[1], result[2] if len(result) > 2 else 0.7
        else:
            answer, data, conf = str(result), {}, 0.7
    except Exception as e:
        answer = f"Oracle error: {e}"
        data = {}
        conf = 0.0
    return {
        "answer": answer,
        "data": data,
        "confidence": conf,
        "intent": intent,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
