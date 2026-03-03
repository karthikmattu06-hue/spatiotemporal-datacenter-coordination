"""
Fetch PJM hourly load data via PJM API v1 (hrl_load_metered endpoint).

Note: gridstatus PJM.get_load() only supports the last 30 days (inst_load endpoint).
For historical data (2020-2024) we call the PJM API directly.

Fetches zone='RTO' — the PJM-wide aggregated total load.
First available: 1993-01-01. Data lag: ~90 days for company-verified values.

Load: 2020-2024.
LMP: Skipped. gridstatus PJM LMP fetches all pricing nodes (~195 pages/month),
     causing 429 rate limits. Since LMP is validation-only and BPA has no LMP,
     we use load-only for complementarity analysis across all three regions.

API key: Set PJM_API_KEY in config.py or as environment variable PJM_API_KEY.

Saves raw CSVs to data/raw/pjm/ in monthly chunks for resumability.

Usage:
    python -m src.data_collection.fetch_pjm
    python -m src.data_collection.fetch_pjm --start 2022-01 --end 2022-06
    python -m src.data_collection.fetch_pjm --test   # fetch one month to verify key
"""

import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    from .config import RAW_DIR, START_DATE, END_DATE, LOG_LEVEL, PJM_API_KEY
except ImportError:
    from config import RAW_DIR, START_DATE, END_DATE, LOG_LEVEL, PJM_API_KEY

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = RAW_DIR / "pjm"

PJM_BASE_URL = "https://api.pjm.com/api/v1/hrl_load_metered"
PJM_MAX_ROWS = 50_000


def get_api_key() -> str:
    """Resolve PJM API key from config or environment variable."""
    key = PJM_API_KEY or os.environ.get("PJM_API_KEY")
    if not key:
        raise RuntimeError(
            "PJM API key not found. Set PJM_API_KEY in config.py or as an "
            "environment variable. Register at: accountmanager.pjm.com"
        )
    return key


def month_ranges(start: datetime, end: datetime):
    """Generate (month_start, month_end) tuples covering the date range."""
    current = start.replace(day=1)
    while current <= end:
        if current.month == 12:
            month_end = current.replace(year=current.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = current.replace(month=current.month + 1, day=1) - timedelta(days=1)
        month_end = min(month_end, end)
        yield current, month_end
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)


def fetch_pjm_page(api_key: str, ept_range: str, offset: int = 0) -> dict:
    """Fetch one page of PJM hourly load data for zone=RTO, with retry on 429."""
    params = {
        "startRow": offset + 1,
        "rowCount": PJM_MAX_ROWS,
        "datetime_beginning_ept": ept_range,
        "zone": "RTO",
        "order": "Asc",
        "sort": "datetime_beginning_utc",
    }
    backoff = 10
    for attempt in range(5):
        resp = requests.get(
            PJM_BASE_URL,
            params=params,
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=60,
        )
        if resp.status_code == 429:
            logger.warning(f"  429 rate limit — sleeping {backoff}s (attempt {attempt+1}/5)")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # raise after all retries exhausted


def fetch_month(api_key: str, m_start: datetime, m_end: datetime) -> pd.DataFrame | None:
    """Fetch all RTO load records for a month, handling pagination."""
    # PJM API uses EPT (Eastern Prevailing Time) for datetime_beginning_ept filter
    # Format: MM/DD/YYYY HH:MMtoMM/DD/YYYY HH:MM
    start_str = m_start.strftime("%m/%d/%Y 00:00")
    end_str = (m_end + timedelta(days=1)).strftime("%m/%d/%Y 00:00")
    ept_range = f"{start_str}to{end_str}"

    all_records = []
    offset = 0

    while True:
        try:
            data = fetch_pjm_page(api_key, ept_range, offset)
        except requests.HTTPError as e:
            logger.error(f"  HTTP error: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"  Request error: {e}")
            return None

        records = data.get("items", [])
        if not records:
            break

        all_records.extend(records)
        total = int(data.get("totalRows", 0))

        if len(all_records) >= total or len(records) < PJM_MAX_ROWS:
            break

        offset += PJM_MAX_ROWS
        time.sleep(0.5)

    if not all_records:
        return None

    return pd.DataFrame(all_records)


def fetch_load(api_key: str, start: datetime, end: datetime, output_dir: Path) -> int:
    """Fetch PJM RTO hourly load, saving monthly CSVs. Returns months fetched."""
    success_count = 0
    for m_start, m_end in month_ranges(start, end):
        tag = m_start.strftime("%Y-%m")
        out_path = output_dir / f"load_{tag}.csv"

        if out_path.exists():
            logger.info(f"[LOAD] {tag} already exists, skipping.")
            success_count += 1
            continue

        logger.info(f"[LOAD] Fetching {tag} ...")
        df = fetch_month(api_key, m_start, m_end)

        if df is not None and not df.empty:
            df.to_csv(out_path, index=False)
            logger.info(f"[LOAD] {tag}: {len(df)} rows → {out_path.name}")
            success_count += 1
        else:
            logger.warning(f"[LOAD] {tag}: empty response")

        time.sleep(2)  # PJM API rate limit: be conservative between months

    return success_count


def parse_month_arg(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m")


def end_of_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1) - timedelta(days=1)
    return dt.replace(month=dt.month + 1, day=1) - timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Fetch PJM historical load via PJM API v1")
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start month (YYYY-MM). Default: 2020-01"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End month (YYYY-MM). Default: 2024-12"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Fetch one month (2024-01) to verify API key and data availability"
    )
    args = parser.parse_args()

    api_key = get_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        logger.info("Running PJM API test (2024-01)...")
        test_start = datetime(2024, 1, 1)
        test_end = datetime(2024, 1, 31)
        n = fetch_load(api_key, test_start, test_end, OUTPUT_DIR)
        if n > 0:
            logger.info("PJM API test PASSED — historical load data available.")
        else:
            logger.error("PJM API test FAILED — check API key and network.")
        return

    fetch_start = parse_month_arg(args.start) if args.start else START_DATE
    fetch_end = end_of_month(parse_month_arg(args.end)) if args.end else END_DATE

    logger.info(f"PJM load fetch (zone=RTO): {fetch_start.date()} → {fetch_end.date()}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    n = fetch_load(api_key, fetch_start, fetch_end, OUTPUT_DIR)
    logger.info(f"Load fetch complete: {n} months")
    logger.info("Done.")


if __name__ == "__main__":
    main()
