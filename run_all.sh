#!/bin/bash
# CEREBRO — Run full pipeline: gather data + phase1 ingest
set -e
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || { python3 -m venv .venv && source .venv/bin/activate && pip install -q -r requirements.txt; }
echo ">>> Running data gather..."
python3 cerebro_data_gather.py
echo ""
echo ">>> Running GSS loader (if .dta/.sav/.sas7bdat present)..."
python3 cerebro_gss_loader.py || true
echo ""
echo ">>> Running GLOPOP-S aggregator (extract zip, aggregate .dat.gz by country)..."
python3 cerebro_glopop_aggregator.py || true
echo ""
echo ">>> Running deep data loader (GLOPOP-S, ISSP, GBCD, WDI, etc.)..."
python3 cerebro_deep_data_loader.py || true
echo ""
echo ">>> Running country risk engine (WDI Gini + deep data)..."
python3 cerebro_risk_engine.py || true
echo ""
echo ">>> Running Phase 2 pipeline (PRIMARY→BACKUP chain)..."
python3 cerebro_pipeline.py || python3 cerebro_trends_loader.py || true
echo ""
echo ">>> Running phase1 harm clock ingest..."
python3 cerebro_phase1_ingest.py
echo ""
echo ">>> Running backtest..."
python3 cerebro_backtest.py || true
echo ""
echo ">>> State-space + hazard + changepoint (Phase 2 engine)..."
python3 cerebro_state_space.py || true
python3 cerebro_hazard.py || true
python3 cerebro_changepoint.py 2>/dev/null || true
python3 cerebro_validation_v2.py || true
python3 cerebro_model_metrics.py || true
echo ""
echo ">>> Running tests..."
python3 test_data_gather.py
python3 tests/test_oracle.py || true
python3 tests/test_calibration.py || true
echo ""
echo ">>> Fitting distance weights (optional)..."
python3 cerebro_fit_distance_weights.py || true
echo ""
echo ">>> Exporting UI data..."
python3 cerebro_export_ui_data.py
echo ""
echo "DONE. Output: cerebro_harm_clock_data.csv, cerebro_harm_clock_phase1.xlsx, public/index.html, public/cerebro_data.json"
