#!/usr/bin/env python3
"""
CHIMERA Bridge — Single entrypoint for pipeline.
Calls chimera.run_figure8(). Swallows errors, never crashes pipeline.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cerebro_data"
EXPORT_PATH = DATA_DIR / "chimera_export.json"
INFINITY_PATH = DATA_DIR / "infinity_score.json"


def main():
    try:
        from chimera import chimera_figure8
        result = chimera_figure8.run_figure8()
        print("CHIMERA figure-8 OK")
        return 0
    except Exception as e:
        # Minimal error export — never crash pipeline
        DATA_DIR.mkdir(exist_ok=True)
        err_export = {"error": str(e), "status": "failed"}
        try:
            with open(EXPORT_PATH, "w") as f:
                json.dump(err_export, f, indent=2)
        except Exception:
            pass
        try:
            with open(INFINITY_PATH, "w") as f:
                json.dump({"infinity_score": None, "error": str(e)}, f, indent=2)
        except Exception:
            pass
        print(f"CHIMERA figure-8 error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
