# Spatiotemporal Datacenter Coordination Research

Research project studying cross-region grid stress complementarity to justify datacenter load migration (CAISO, PJM, BPA). Goal: show temporal offsets between regional stress peaks enable a MIP-optimized coordinator to reduce grid impact by 35–40%.

## Environment

```bash
source venv/bin/activate          # always activate before running scripts
pip install -r requirements.txt   # if adding packages, pin versions
```

Python 3.x, all scripts run from project root.

## Pipeline (run in order)

```bash
# CAISO (complete)
python src/data_collection/fetch_caiso.py    # fetch raw CSVs
python src/data_collection/clean_caiso.py   # validate + merge → parquet
python src/analysis/stress_analysis.py      # annotate stress hours
python src/plotting/visualize_stress.py     # generate figures/caiso/

# PJM (next — blocked on API key)
python src/data_collection/fetch_pjm.py
python src/data_collection/clean_pjm.py
python src/analysis/stress_analysis.py --region pjm

# BPA (next)
python src/data_collection/fetch_bpa.py     # via EIA API or direct Excel
```

## Data Layout

```
data/raw/{caiso,pjm,bpa}/          # monthly CSVs
data/processed/{caiso,pjm,bpa}/    # parquet files
figures/{caiso,pjm,bpa}/           # output plots
```

Key processed files: `caiso_merged_hourly.parquet`, `caiso_stress_annotated.parquet`, `caiso_stress_summary.json`

## gridstatus API Rules (critical)

- **Use enum, NOT string for LMP:** `from gridstatus.base import Markets` → `Markets.DAY_AHEAD_HOURLY`
- `"DAM"` string does NOT work — silent failure
- CAISO load: works for all years (2020–2024), no key needed
- CAISO LMP: **only 2023+ works**; 2022 and earlier return OASIS error 1015 — this is a server-side limit, not a bug
- PJM: requires API key from accountmanager.pjm.com; test LMP historical range early (may hit same limit)
- EIA API: has hourly demand only (`electricity/rto/region-data`) — **no hourly LMP/wholesale prices**
- BPA: use EIA API for demand, or direct Excel from transmission.bpa.gov; LMP likely unavailable (non-ISO market)

## Stress Analysis Design

- **Primary signal:** load > seasonal P90 (per calendar month, not global — prevents summer domination)
- **Blip filter:** events < 2 consecutive hours removed (too short for real DR response)
- **LMP:** validation only — DR is demand-triggered, not price-triggered
- **LMP threshold:** mean + 2σ (global, not seasonal — spikes are structural)
- CAISO results: 4,244 stress hours (9.7%), 957 events, peak at 00:00–04:00 UTC (6–8 PM Pacific, duck curve)

## API Keys (config.py)

```python
PJM_API_KEY = None   # TODO: accountmanager.pjm.com — submitted, pending activation
EIA_API_KEY = None   # TODO: registered and active — add key
```

Do not commit API keys. Use environment variables or a local `.env` file.

## Current Status & Next Steps

**Done:** CAISO full pipeline (fetch → clean → stress → plots)

**Next (in order):**
1. Activate PJM API key → write `fetch_pjm.py` (clone CAISO pattern, monthly chunks, resumable)
2. Write `fetch_bpa.py` (EIA demand; no LMP expected)
3. Run stress analysis for PJM + BPA, generate heatmaps
4. **Complementarity analysis (make-or-break):** ρ < 0.3, σ (simultaneous stress) rare, fleet availability A > 95%

## Key Pitfalls

- Grid stress ≠ carbon intensity ≠ high prices (17.1% LMP/load overlap in CAISO — expected)
- Migration shifts load geographically, does NOT reduce total power
- No-snap-back constraint applies to return migration after DR event
- Flex tier is a business decision, not a workload property
- Emerald is signal-driven, not forecast-driven
- Don't claim >50% curtailment reduction — target 35–40%
- PJM peak is ~18:00–22:00 UTC vs CAISO 00:00–04:00 UTC (~6h offset) — this is the complementarity thesis
