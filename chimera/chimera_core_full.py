#!/usr/bin/env python3
"""
CHIMERA Core — Template-based capability deployer (local-only, safe templates).
Emulates Chimera Core from PDF: gap detection, whitelist templates, validation, audit.
"""

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
LOG_PATH = DATA_DIR / "chimera_log.jsonl"

TEMPLATE_WHITELIST = frozenset({
    "text_summary",
    "basic_stats",
    "detect_outliers",
    "fit_linear_trend",
    "compute_hazard_curve",
    "score_model_run",
})


def _text_summary(input_text: str, context: Optional[dict] = None) -> dict:
    """Summarize text (first 200 chars)."""
    return {"summary": (input_text or "")[:200], "len": len(input_text or "")}


def _basic_stats(input_text: str, context: Optional[dict] = None) -> dict:
    """Basic stats on numeric context or input."""
    if context and "values" in context:
        vals = [float(x) for x in context["values"] if isinstance(x, (int, float))]
        if vals:
            return {"n": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
    return {"n": 0, "mean": 0, "min": 0, "max": 0}


def _detect_outliers(input_text: str, context: Optional[dict] = None) -> dict:
    """Simple outlier detection (values > 2 std from mean)."""
    if context and "values" in context:
        vals = [float(x) for x in context["values"] if isinstance(x, (int, float))]
        if len(vals) >= 3:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = var ** 0.5 if var > 0 else 0
            outliers = [x for x in vals if std > 0 and abs(x - mean) > 2 * std]
            return {"n_outliers": len(outliers), "outliers": outliers[:10]}
    return {"n_outliers": 0, "outliers": []}


def _fit_linear_trend(input_text: str, context: Optional[dict] = None) -> dict:
    """Fit linear trend to (x,y) pairs in context."""
    if context and "x" in context and "y" in context:
        x = context["x"]
        y = context["y"]
        if len(x) == len(y) and len(x) >= 2:
            n = len(x)
            sx, sy = sum(x), sum(y)
            sxx = sum(xi * xi for xi in x)
            sxy = sum(xi * yi for xi, yi in zip(x, y))
            denom = n * sxx - sx * sx
            if denom != 0:
                slope = (n * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n
                return {"slope": slope, "intercept": intercept, "n": n}
    return {"slope": 0, "intercept": 0, "n": 0}


def _compute_hazard_curve(input_text: str, context: Optional[dict] = None) -> dict:
    """Placeholder: return hazard curve metadata from context or file."""
    p = DATA_DIR / "hazard_curve.json"
    if p.exists():
        try:
            with open(p) as f:
                d = json.load(f)
            return {"source": "file", "keys": list(d.keys())[:5]}
        except Exception:
            pass
    return {"source": "none", "keys": []}


def _score_model_run(input_text: str, context: Optional[dict] = None) -> dict:
    """Placeholder: hook for infinity score."""
    p = DATA_DIR / "infinity_score.json"
    if p.exists():
        try:
            with open(p) as f:
                d = json.load(f)
            return {"infinity_score": d.get("infinity_score"), "G": d.get("G")}
        except Exception:
            pass
    return {"infinity_score": None, "G": None}


_CAPABILITIES: dict[str, Callable] = {
    "text_summary": _text_summary,
    "basic_stats": _basic_stats,
    "detect_outliers": _detect_outliers,
    "fit_linear_trend": _fit_linear_trend,
    "compute_hazard_curve": _compute_hazard_curve,
    "score_model_run": _score_model_run,
}


def _append_audit(timestamp: str, task: str, status: str, module_path: str, result: Any, error: Optional[str] = None):
    """Append audit entry to chimera_log.jsonl."""
    DATA_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": timestamp,
        "task": task,
        "status": status,
        "module_path": module_path,
        "result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result),
        "error": error,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def perform(task_name: str, input_text: str = "", context: Optional[dict] = None) -> dict:
    """
    Execute capability. If not in whitelist, gap_detected; otherwise run.
    No LLM calls. No shell. No code execution outside templates.
    """
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if task_name not in TEMPLATE_WHITELIST:
        _append_audit(ts, task_name, "gap_detected", "", {}, f"task '{task_name}' not in whitelist")
        return {"status": "gap_detected", "task": task_name, "error": "not in whitelist"}

    fn = _CAPABILITIES.get(task_name)
    if fn is None:
        _append_audit(ts, task_name, "gap_detected", task_name, {}, "no handler")
        return {"status": "gap_detected", "task": task_name, "error": "no handler"}

    try:
        result = fn(input_text, context)
        _append_audit(ts, task_name, "ok", task_name, result, None)
        return {"status": "ok", "task": task_name, "result": result}
    except Exception as e:
        _append_audit(ts, task_name, "error", task_name, {}, str(e))
        return {"status": "error", "task": task_name, "error": str(e)}
