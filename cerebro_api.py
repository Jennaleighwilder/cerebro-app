#!/usr/bin/env python3
"""
CEREBRO API — Lightweight server for live Oracle queries
========================================================
Exposes POST /oracle and GET /health. Serves static frontend from public/.
"""

import json
import os
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("Install Flask: pip install flask")
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = SCRIPT_DIR / "public"

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="")


def _get_infinity_score() -> float:
    """Read infinity_score from same source as compute_infinity_score(): cerebro_data/infinity_score.json."""
    path = SCRIPT_DIR / "cerebro_data" / "infinity_score.json"
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return float(data.get("infinity_score", 0))
        except Exception:
            pass
    try:
        from chimera.chimera_infinity_score import compute_infinity_score
        s = compute_infinity_score()
        return float(s.get("infinity_score", 0))
    except Exception:
        return 0.0


@app.route("/health")
def health():
    """Health check with Infinity Score from same source as compute_infinity_score()."""
    inf = _get_infinity_score()
    return jsonify({"status": "ok", "infinity_score": inf})


@app.route("/oracle", methods=["POST"])
def oracle():
    """Handle Oracle query. Body: {"query": "..."}."""
    try:
        body = request.get_json() or {}
        query = body.get("query", "").strip()
        if not query:
            return jsonify({"answer": "Please provide a query.", "data": {}, "confidence": 0, "timestamp": ""}), 400
        from cerebro_oracle_router import route_query
        result = route_query(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "answer": f"Oracle error: {e}",
            "data": {},
            "confidence": 0,
            "timestamp": "",
        }), 500


@app.route("/feedback/run", methods=["POST"])
def run_feedback():
    """Run feedback cycle: score closed windows, inject into calibration."""
    try:
        from cerebro_live_feedback import run_feedback_cycle
        from datetime import datetime
        scored = run_feedback_cycle()
        return jsonify({
            "scored_windows": scored,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "scored_windows": 0}), 500


@app.route("/")
def index():
    """Serve index.html."""
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static files from public/."""
    return send_from_directory(PUBLIC_DIR, path)


def main():
    port = int(os.environ.get("CEREBRO_API_PORT", 5000))
    print(f"CEREBRO API: http://localhost:{port}")
    print("  GET  /health  — health check + Infinity Score")
    print("  POST /oracle  — live query")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
