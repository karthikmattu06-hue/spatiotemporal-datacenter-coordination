"""
Fetch BPA hourly demand data via EIA API v2.

BPA (Bonneville Power Administration) is a non-ISO market — no LMP data available.
Demand is fetched from EIA-930 balancing authority data (respondent: BPAT).

API key: Set EIA_API_KEY in config.py or as environment variable EIA_API_KEY.

Saves raw CSVs to data/raw/bpa/ in monthly chunks for resumability.

Usage:
    python -m src.data_collection.fetch_bpa
    python -m src.data_collection.fetch_bpa --start 2022-01 --end 2022-12
    python -m src.data_collection.fetch_bpa --test   # fetch one month to verify key
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
    from .config import RAW_DIR, START_DATE, END_DATE, LOG_LEVEL, EIA_API_KEY
except ImportError:
    from config import RAW_DIR, START_DATE, END_DATE, LOG_LEVEL, EIA_API_KEY

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = RAW_DIR / "bpa"

EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
BPA_RESPONDENT = "BPAT"
EIA_MAX_RECORDS = 5000


def get_api_key() -> str:
    """Resolve EIA API key from config or environment variable."""
    key = EIA_API_KEY or os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError(
            "EIA API key not found. Set EIA_API_KEY in config.py or as an "
            "environment variable. Register at: www.eia.gov/opendata/"
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


def fetch_eia_page(api_key: str, start_str: str, end_str: str, offset: int = 0) -> dict:
    """Fetch one page of EIA hourly demand data for BPAT, with retry on errors."""
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[type][0]": "D",
        "facets[respondent][0]": BPA_RESPONDENT,
        "start": start_str,
        "end": end_str,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": EIA_MAX_RECORDS,
        "offset": offset,
    }
    backoff = 10
    for attempt in range(5):
        try:
            resp = requests.get(EIA_BASE_URL, params=params, timeout=90)
            if resp.status_code in (429, 502, 503, 504):
                logger.warning(f"  HTTP {resp.status_code} — sleeping {backoff}s (attempt {attempt+1}/5)")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            logger.warning(f"  Timeout — sleeping {backoff}s (attempt {attempt+1}/5)")
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)
    resp.raise_for_status()


def fetch_month(api_key: str, m_start: datetime, m_end: datetime) -> pd.DataFrame | None:
    """Fetch all demand records for a month, handling pagination."""
    # EIA period format: YYYY-MM-DDTHH (UTC)
    start_str = m_start.strftime("%Y-%m-%dT%H")
    # Fetch through the last hour of the last day
    end_str = (m_end + timedelta(days=1)).strftime("%Y-%m-%dT%H")

    all_records = []
    offset = 0

    while True:
        try:
            data = fetch_eia_page(api_key, start_str, end_str, offset)
        except requests.HTTPError as e:
            logger.error(f"  HTTP error: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"  Request error: {e}")
            return None

        response = data.get("response", {})
        records = response.get("data", [])

        if not records:
            break

        all_records.extend(records)
        total = int(response.get("total", 0))

        if len(all_records) >= total or len(records) < EIA_MAX_RECORDS:
            break

        offset += EIA_MAX_RECORDS
        time.sleep(0.5)  # be polite between pages

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    return df


def fetch_demand(api_key: str, start: datetime, end: datetime, output_dir: Path) -> int:
    """Fetch BPA hourly demand, saving monthly CSVs. Returns months fetched."""
    success_count = 0
    for m_start, m_end in month_ranges(start, end):
        tag = m_start.strftime("%Y-%m")
        out_path = output_dir / f"load_{tag}.csv"

        if out_path.exists():
            logger.info(f"[DEMAND] {tag} already exists, skipping.")
            success_count += 1
            continue

        logger.info(f"[DEMAND] Fetching {tag} ...")
        df = fetch_month(api_key, m_start, m_end)

        if df is not None and not df.empty:
            df.to_csv(out_path, index=False)
            logger.info(f"[DEMAND] {tag}: {len(df)} rows → {out_path.name}")
            success_count += 1
        else:
            logger.warning(f"[DEMAND] {tag}: empty response")

        time.sleep(1)

    return success_count


def parse_month_arg(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m")


def end_of_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1) - timedelta(days=1)
    return dt.replace(month=dt.month + 1, day=1) - timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Fetch BPA hourly demand via EIA API v2")
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
        help="Fetch one month (2023-01) to verify API key and data availability"
    )
    args = parser.parse_args()

    api_key = get_api_key()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        logger.info("Running EIA API test (2023-01)...")
        test_start = datetime(2023, 1, 1)
        test_end = datetime(2023, 1, 31)
        n = fetch_demand(api_key, test_start, test_end, OUTPUT_DIR)
        if n > 0:
            logger.info("EIA API test PASSED — BPA demand data available.")
        else:
            logger.error("EIA API test FAILED — check API key and network.")
        return

    fetch_start = parse_month_arg(args.start) if args.start else START_DATE
    fetch_end = end_of_month(parse_month_arg(args.end)) if args.end else END_DATE

    logger.info(f"BPA demand fetch: {fetch_start.date()} → {fetch_end.date()}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    n = fetch_demand(api_key, fetch_start, fetch_end, OUTPUT_DIR)
    logger.info(f"Fetch complete: {n} months")
    logger.info("Done.")


if __name__ == "__main__":
    main()
