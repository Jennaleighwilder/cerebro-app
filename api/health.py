"""GET /api/health — lightweight health for static + serverless split."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self) -> None:
        inf = 0.0
        try:
            p = ROOT / "cerebro_data" / "infinity_score.json"
            if p.exists():
                with open(p) as f:
                    inf = float(json.load(f).get("infinity_score", 0))
        except Exception:
            pass
        body = json.dumps({"status": "ok", "infinity_score": inf}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
