# CEREBRO DATA COLLECTION — Step-by-Step Checklist

Run `./run_all.sh` or `python cerebro_data_gather.py` then `python cerebro_phase1_ingest.py` after each step.

---

## ✅ COMPLETED (automated)

| Step | Source | Status | File |
|------|--------|--------|------|
| 1 | FRED (unemployment, LFPR, income, Gini, CPI, etc.) | ✓ | cerebro_data/FRED_combined.csv |
| 2 | World Bank homicide | ✓ | cerebro_data/WorldBank_homicide_US.csv |
| 3 | UCDP Armed Conflict + Battle Deaths | ✓ | cerebro_data/UCDP_*.xlsx |
| 4 | **Pew Trust in Government** | ✓ | cerebro_data/PEW_trust_government.csv |
| 5 | **GSS attitudes** (SAS/Stata/SPSS) | ✓ | cerebro_data/GSS_RingB_annual.csv |
| 6 | Phase1 Harm Clock (full Ring B) | ✓ | cerebro_harm_clock_data.csv, cerebro_harm_clock_phase1.xlsx |

**Ring B is now populated** with GSS (COURTS, CAPPUN, TRUST, FEAR, POLHITOK) + Pew trust. Saddle signals sharpened.

---

## 🔲 NEXT: GSS Batch (single step, ~25 min)

**This one download covers ~70% of remaining Ring B needs.**

1. Go to https://gssdataexplorer.norc.org/
2. Register (email + password) — see cerebro_data_collection_guide.md for exact steps
3. Sign in → My GSS → Extracts → Create Extract
4. Add variables: COURTS, POLHITOK, CAPPUN, GRASS, TRUST, FEAR, PREMARSX, HOMOSEX, ABANY, ATTEND, RELITEN, PARSOL, EQWLTH, CLASS, YEAR
5. Select all years (1972–2022), format CSV
6. Submit — file arrives by email in ~10 min
7. Save the CSV to: `cerebro_data/GSS_BATCH_all_variables.csv`
8. Run: `python cerebro_phase1_ingest.py` (after we add GSS loader)

---

## 🔲 AFTER GSS: Remaining manual sources

| Step | Source | Time | Action |
|------|--------|------|--------|
| 7 | Gallup crime | 5 min | Copy table from news.gallup.com/poll/1603/crime.aspx → cerebro_data/GALLUP_crime.csv |
| 8 | CDC WONDER overdose 2023 | 15 min | wonder.cdc.gov → drug poisoning → export → cerebro_data/CDC_overdose_2023.csv |
| 9 | ACLED USA | 15 min | Register at acleddata.com → export USA 2000–present → cerebro_data/ACLED_usa.csv |

---

## Quick run

```bash
cd "/Users/jenniferwest/Downloads/files (88)"
./run_all.sh
```
