# GLOPOP-S — 7.3 Billion Individuals

**Reference:** Ton et al. (2024). A global dataset of 7 billion individuals with socio-economic characteristics. *Scientific Data* 11:1096. https://doi.org/10.1038/s41597-024-03864-2

## Data

- **1,999,227,130 households**
- **7,335,881,094 individuals**
- **Year:** 2015
- **78 countries** with LIS/DHS microdata (81.4% global pop)
- **99 countries** synthetic (15.7% global pop)

## Download

1. **Harvard Dataverse:** https://doi.org/10.7910/DVN/KJC3RH  (direct data)
2. **GitHub (read scripts):** https://github.com/VU-IVM/GLOPOP-S/

Data is stored in **binary format**. Use `read_synthpop_data.py` or `read_synthpop_data.R` from the GitHub repo to load.

## Folder Structure

- Folders named by **ISO country codes**
- Within each: **GDL regions** (admin unit 1)
- For large countries (China, India): download per region to save memory

## Attributes (Table 2 from paper)

| Attribute | Level | Values |
|------------|-------|--------|
| Income | Individual | 1–5 (poorest→richest 20%), −1 unavailable |
| Wealth | Individual | 1–5 (poorest→richest 20%), −1 unavailable |
| Settlement type | Household | 0 urban, 1 rural |
| Age | Individual | 1: 0–4, 2: 5–14, 3: 15–24, 4: 25–34, 5: 35–44, 6: 45–54, 7: 55–64, 8: 65+ |
| Gender | Individual | 1 male, 0 female |
| Education | Individual | 1–5 (less than primary → higher) |
| Household type | Household | 1 single, 2 couple, 3 couple+children, etc. |
| Household size | Household | 1–6 (1, 2, 3–4, 5–6, 7–10, 10+) |

## For CEREBRO

Place extracted data in `cerebro_data/GLOPOP-S/` (or per-country CSVs). The loader expects:
- Income or wealth column (quintiles 1–5)
- Country/region identifier

**Note:** Income/wealth is **not cross-country comparable** — poorest 20% in country X ≠ poorest 20% in country Y.
