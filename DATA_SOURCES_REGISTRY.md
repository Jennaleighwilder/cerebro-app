# CEREBRO DATA SOURCES REGISTRY

Master list of all datasets with Ring assignments, confidence scoring, and ingestion status.

---

## RING ASSIGNMENTS

| Ring | Purpose | Data Type |
|------|---------|-----------|
| **Ring A** | Structural (behavioral, institutional) | FRED, BJS, FBI, CDC, World Bank |
| **Ring B** | Normative (attitudes, beliefs) | GSS, Pew, ISSP |
| **Ring C** | External pressure (geopolitical, environmental) | UCDP, ACLED, Freedom House, UNHCR |
| **L1** | Leading indicators (search, attention) | Google Trends |
| **L2** | Narrative intelligence (text, migration) | GBCD Corpus |

---

## TIER 1 — GLOBAL POPULATION & INDIVIDUAL-LEVEL

### GLOPOP-S — 7.3B Individuals
| Field | Value |
|-------|-------|
| **Ring** | Ring A (Class micro-validation) |
| **Confidence** | 92% (academic-grade, LIS+DHS origin) |
| **Coverage** | 78 countries (81.4% global pop) + 99 synthetic |
| **Source** | Springer Nature / Scientific Data |
| **URL** | https://link.springer.com/article/10.1038/s41597-024-03864-2 |
| **Format** | Per region/country, open source |
| **Clocks** | Class (income/wealth), Harm (household risk), Sexual (household type), Evil (vulnerability) |
| **Status** | 🔲 Pending — requires download + aggregation |

### World Development Indicators — 265 Countries, 1960–2024
| Field | Value |
|-------|-------|
| **Ring** | Ring A (structural backfill) |
| **Confidence** | 90% (World Bank official) |
| **Coverage** | 265 countries, 64 years |
| **Source** | Kaggle / World Bank |
| **URL** | https://www.kaggle.com/datasets/umitka/world-development-indicators/ |
| **Format** | CSV 228 MB |
| **Clocks** | Class (GDP, Gini), Evil (governance), Harm (health), all clocks |
| **Status** | 🔲 Pending — Kaggle API or manual download |

### Global Development Analysis — 2000–2020
| Field | Value |
|-------|-------|
| **Ring** | Ring A + Ring C |
| **Confidence** | 85% |
| **Coverage** | 190+ countries |
| **Source** | Kaggle / Google BigQuery |
| **URL** | https://www.kaggle.com/datasets/michaelmatta0/global-development-indicators-2000-2020 |
| **Format** | CSV 3.13 MB |
| **Clocks** | Digital transformation, governance quality, climate |
| **Status** | 🔲 Pending |

---

## TIER 2 — ATTITUDINAL & CULTURAL

### ISSP Cumulations — 1985–2023, 40+ Countries
| Field | Value |
|-------|-------|
| **Ring** | Ring B (global attitudes) |
| **Confidence** | 88% (GESIS, global GSS) |
| **Coverage** | 40+ countries, 7 modules |
| **Source** | GESIS — Leibniz Institute |
| **URL** | https://www.gesis.org/en/issp/data-and-documentation/data-cumulations |
| **Format** | SPSS, Stata, CSV |
| **Modules** | Environment, Health, National Identity, Religion, Role of Government, Social Inequality, Work |
| **Clocks** | All four — redistribution, secularization, trust, xenophobia |
| **Status** | 🔲 Pending — GESIS registration required |

### GSS (US) — Ring B
| Field | Value |
|-------|-------|
| **Ring** | Ring B |
| **Confidence** | 94% |
| **Status** | ✅ Active |

### Pew Trust — Ring B
| Field | Value |
|-------|-------|
| **Ring** | Ring B |
| **Confidence** | 90% |
| **Status** | ✅ Active |

---

## TIER 3 — NARRATIVE & L2 INTELLIGENCE

### GBCD Corpus — 2.9B Tokens, 223 Countries
| Field | Value |
|-------|-------|
| **Ring** | L2 (narrative intelligence) |
| **Confidence** | 82% (real-time, Nature/Scientific Data) |
| **Coverage** | 223 countries, 2000–2024 |
| **Source** | Nature / GitHub |
| **URL** | https://github.com/Computational-social-science/GBCD |
| **Format** | Structured + narrative text |
| **Clocks** | Class (migration/talent flow), Evil (geopolitical tension) |
| **Status** | 🔲 Pending — clone + parse |

---

## TIER 4 — STRUCTURAL PRESSURE

### NASA Socioeconomic Data
| Field | Value |
|-------|-------|
| **Ring** | Ring C (structural pressure) |
| **Confidence** | 85% (US gov) |
| **Coverage** | 1980–2100 (projections) |
| **Source** | NASA Earthdata |
| **URL** | https://www.earthdata.nasa.gov/topics/human-dimensions/socioeconomics/data-access-tools |
| **Datasets** | GDP projections, IPCC baseline, poverty mapping, INFORM risk |
| **Clocks** | Class (poverty), Evil (risk), all (climate vulnerability) |
| **Status** | 🔲 Pending |

---

## TIER 5 — EXISTING (EXPAND)

### UCDP — Expand to GED Georeferenced
| Field | Value |
|-------|-------|
| **Ring** | Ring C |
| **Confidence** | 92% |
| **Current** | Armed conflict annual count |
| **Expand** | UCDP GED — 1946–present, lat/lon, fatalities, conflict type |
| **URL** | https://ucdp.uu.se/downloads/index.html#ged_global |
| **Status** | ⚠ Partial — expand to GED |

### ACLED — Full History
| Field | Value |
|-------|-------|
| **Ring** | Ring C |
| **Confidence** | 88% |
| **Current** | Placeholder (143 events) |
| **Expand** | 1997–present, battles, riots, protests, actor data, 15-min updates |
| **URL** | https://acleddata.com/data-export-tool/ |
| **Status** | ⚠ Placeholder — requires API/export |

### Freedom House — Disaggregated
| Field | Value |
|-------|-------|
| **Ring** | Ring C (Evil clock) |
| **Confidence** | 88% |
| **Current** | Not integrated |
| **Expand** | Electoral Democracy, Civil Liberties components, country reports |
| **URL** | https://freedomhouse.org/report/freedom-world |
| **Status** | 🔲 Pending |

### UNHCR — Microdata
| Field | Value |
|-------|-------|
| **Ring** | Ring C |
| **Confidence** | 85% |
| **Current** | Not integrated |
| **Expand** | Microdata Library — demographics, displacement reasons, intention surveys |
| **URL** | https://microdata.unhcr.org |
| **Status** | 🔲 Pending |

---

## ACTIVE SOURCES (LIVE)

| Source | Ring | Confidence | File |
|--------|------|------------|------|
| FRED | A | 95% | FRED_combined.csv |
| World Bank | A | 90% | WorldBank_homicide_US.csv |
| UCDP ACD | C | 92% | UCDP_conflict_annual.csv |
| Pew Trust | B | 90% | PEW_trust_government.csv |
| CDC STI | A | 88% | cerebro_gathered_raw |
| GSS | B | 94% | GSS_RingB_annual.csv |
| Google Trends | L1 | 85% | GoogleTrends_*.csv |

---

## PRIORITY ORDER FOR INGESTION

1. **GLOPOP-S** — Class clock micro-validation
2. **ISSP Cumulations** — Global Ring B
3. **GBCD Corpus** — L2 narrative layer
4. **NASA Socioeconomic** — Structural pressure
5. **World Development Indicators** — Full backfill
6. **UCDP GED** — Georeferenced conflict
7. **ACLED Full** — Protest/riot history
8. **UNHCR Microdata** — Displacement
9. **Freedom House Disaggregated** — Evil clock
