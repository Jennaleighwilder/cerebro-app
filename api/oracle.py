"""
Vercel Python serverless: POST /api/oracle → cerebro_oracle_router.route_query
Rewrite /oracle → /api/oracle in vercel.json for the static frontend.
"""

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

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
        except (json.JSONDecodeError, ValueError):
            self._send_json(
                400,
                {
                    "answer": "Invalid JSON body.",
                    "data": {},
                    "confidence": 0,
                    "timestamp": "",
                    "intent": None,
                },
            )
            return

        query = (body.get("query") or "").strip()
        if not query:
            self._send_json(
                400,
                {
                    "answer": "Please provide a query.",
                    "data": {},
                    "confidence": 0,
                    "timestamp": "",
                    "intent": None,
                },
            )
            return

        try:
            from cerebro_oracle_router import route_query

            result = route_query(query)
            self._send_json(200, result)
        except Exception as e:
            self._send_json(
                500,
                {
                    "answer": f"Oracle error: {e}",
                    "data": {},
                    "confidence": 0,
                    "timestamp": "",
                    "intent": None,
                },
            )
