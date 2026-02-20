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
echo ">>> Walk-forward + calibration + integrity (v2 dominance)..."
python3 cerebro_walkforward.py || true
python3 cerebro_calibration.py || true
python3 cerebro_integrity.py || true
python3 cerebro_stress.py || true
python3 cerebro_ablation.py || true
python3 cerebro_live_monitor.py || true
echo ""
echo ">>> Four-track elite (cross-national, baselines, stability, hazard, regime, spillover, rolling-origin)..."
python3 cerebro_crossnational.py || true
python3 cerebro_baselines.py || true
python3 cerebro_parameter_stability.py || true
python3 cerebro_hazard_curve.py || true
python3 cerebro_regime.py || true
python3 cerebro_spillover.py || true
python3 cerebro_rolling_origin.py || true
echo ""
echo ">>> Running tests..."
python3 test_data_gather.py || true
python3 tests/test_core_frozen.py -v || true
python3 tests/test_dominance.py -v || true
python3 -m pytest tests/ -q 2>/dev/null || true
echo ""
echo ">>> Exporting UI data..."
python3 cerebro_export_ui_data.py
echo ""
echo "DONE. Output: cerebro_harm_clock_data.csv, cerebro_harm_clock_phase1.xlsx, public/index.html, public/cerebro_data.json"
