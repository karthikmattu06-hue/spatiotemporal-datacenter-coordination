# Data Collection

## Overview

This directory contains scripts to download hourly grid stress data from three U.S. grid operators for the period 2020–2024. All data is freely available from public sources with no paid access required.

## Data Sources

### PJM (Mid-Atlantic)
- **Portal:** [Data Miner 2](https://dataminer2.pjm.com)
- **API:** `https://api.pjm.com/api/v1/` (free account required — register at [accountmanager.pjm.com](https://accountmanager.pjm.com))
- **Data collected:**
  - Hourly metered load by zone (`hrl_load_metered`)
  - Day-ahead hourly LMPs (`da_hrl_lmps`)
  - Operating reserves
- **Format:** JSON/CSV via REST API
- **Note:** Non-member API rate limit is 6 calls/minute

### CAISO (California)
- **Portal:** [OASIS](https://oasis.caiso.com)
- **API:** `http://oasis.caiso.com/oasisapi/SingleZip` (no registration needed)
- **Data collected:**
  - System load and net demand (`SLD_FCST`)
  - Day-ahead LMPs (`PRC_LMP`)
  - Ancillary services (`AS_REQ`, `AS_RESULTS`)
- **Format:** XML/CSV (zipped), 31-day query limit per request
- **Note:** Append `resultformat=6` for CSV output

### BPA (Pacific Northwest)
- **Portal:** [transmission.bpa.gov](https://transmission.bpa.gov/business/operations/wind/)
- **Data collected:**
  - 5-minute load, wind, hydro, thermal generation (annual Excel files)
  - Reserves deployed
- **Format:** Excel (.xls for 2020–2021, .xlsx for 2022+)
- **Alternative:** [EIA API v2](https://api.eia.gov/v2/) for hourly demand (free key at [eia.gov/opendata](https://www.eia.gov/opendata/))

## Usage

```bash
# PJM (requires PJM API subscription key in environment)
export PJM_API_KEY="your-key-here"
python fetch_pjm.py

# CAISO (no credentials needed)
python fetch_caiso.py

# BPA (no credentials needed)
python fetch_bpa.py
```

Downloaded data is saved to `data/raw/{pjm,caiso,bpa}/` and is .gitignored due to file size.

## Python Shortcut: gridstatus

The `gridstatus` library provides a unified interface for PJM and CAISO, and reaches BPA via EIA:

```python
import gridstatus

pjm = gridstatus.PJM()
df = pjm.get_load(start="Jan 1, 2020", end="Dec 31, 2024")

caiso = gridstatus.CAISO()
df = caiso.get_load(start="Jan 1, 2020", end="Dec 31, 2024")
```
