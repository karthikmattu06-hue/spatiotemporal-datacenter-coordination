"""
Fetch CAISO hourly load and LMP data using gridstatus.

Load: 2020-2024 (full range, OASIS reliable)
LMP:  2023-2024 (OASIS returns error 1015 for older data)

Saves raw CSVs to data/raw/caiso/ in monthly chunks for resumability.

Usage:
    python -m src.data_collection.fetch_caiso
    python -m src.data_collection.fetch_caiso --load-only
    python -m src.data_collection.fetch_caiso --lmp-only --start 2023-01 --end 2023-06
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from gridstatus.base import Markets

try:
    from gridstatus import CAISO
except ImportError:
    raise ImportError(
        "gridstatus is required. Install with: pip install gridstatus"
    )

try:
    from .config import (
        RAW_DIR, START_DATE, END_DATE, LMP_START_DATE, LMP_END_DATE, LOG_LEVEL,
    )
except ImportError:
    from config import (
        RAW_DIR, START_DATE, END_DATE, LMP_START_DATE, LMP_END_DATE, LOG_LEVEL,
    )

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = RAW_DIR / "caiso"
iso = CAISO()


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


def fetch_load(start: datetime, end: datetime, output_dir: Path) -> int:
    """Fetch hourly load data, saving monthly CSVs. Returns months fetched."""
    success_count = 0
    for m_start, m_end in month_ranges(start, end):
        tag = m_start.strftime("%Y-%m")
        out_path = output_dir / f"load_{tag}.csv"

        if out_path.exists():
            logger.info(f"[LOAD] {tag} already exists, skipping.")
            success_count += 1
            continue

        logger.info(f"[LOAD] Fetching {tag} ...")
        try:
            df = iso.get_load(
                start=m_start.strftime("%Y-%m-%d"),
                end=(m_end + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if df is not None and not df.empty:
                df.to_csv(out_path, index=True)
                logger.info(f"[LOAD] {tag}: {len(df)} rows → {out_path.name}")
                success_count += 1
            else:
                logger.warning(f"[LOAD] {tag}: empty response")
        except Exception as e:
            logger.error(f"[LOAD] {tag} failed: {e}")

        time.sleep(1)

    return success_count


def fetch_lmp(start: datetime, end: datetime, output_dir: Path) -> int:
    """Fetch hourly DAM LMP data, saving monthly CSVs. Returns months fetched."""
    success_count = 0
    for m_start, m_end in month_ranges(start, end):
        tag = m_start.strftime("%Y-%m")
        out_path = output_dir / f"lmp_{tag}.csv"

        if out_path.exists():
            logger.info(f"[LMP] {tag} already exists, skipping.")
            success_count += 1
            continue

        logger.info(f"[LMP] Fetching {tag} ...")
        try:
            df = iso.get_lmp(
                start=m_start.strftime("%Y-%m-%d"),
                end=(m_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                market=Markets.DAY_AHEAD_HOURLY,
            )
            if df is not None and not df.empty:
                df.to_csv(out_path, index=True)
                logger.info(f"[LMP] {tag}: {len(df)} rows → {out_path.name}")
                success_count += 1
            else:
                logger.warning(f"[LMP] {tag}: empty response")
        except Exception as e:
            logger.error(f"[LMP] {tag} failed: {e}")

        time.sleep(1)

    return success_count


def parse_month_arg(s: str) -> datetime:
    """Parse YYYY-MM string to datetime, defaulting to first of month."""
    return datetime.strptime(s, "%Y-%m")


def end_of_month(dt: datetime) -> datetime:
    """Return last day of the month for a given datetime."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1) - timedelta(days=1)
    return dt.replace(month=dt.month + 1, day=1) - timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Fetch CAISO load and LMP data")
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start month (YYYY-MM). Default: 2020-01 for load, 2023-01 for LMP"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End month (YYYY-MM). Default: 2024-12"
    )
    parser.add_argument(
        "--load-only", action="store_true", help="Fetch only load data"
    )
    parser.add_argument(
        "--lmp-only", action="store_true", help="Fetch only LMP data"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Fetch load ────────────────────────────────────────────────────
    if not args.lmp_only:
        load_start = parse_month_arg(args.start) if args.start else START_DATE
        load_end = end_of_month(parse_month_arg(args.end)) if args.end else END_DATE
        logger.info(f"CAISO load fetch: {load_start.date()} → {load_end.date()}")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        n = fetch_load(load_start, load_end, OUTPUT_DIR)
        logger.info(f"Load fetch complete: {n} months")

    # ── Fetch LMP ─────────────────────────────────────────────────────
    if not args.load_only:
        # Default to the restricted LMP range unless user overrides
        lmp_start = parse_month_arg(args.start) if args.start else LMP_START_DATE
        lmp_end = end_of_month(parse_month_arg(args.end)) if args.end else LMP_END_DATE

        if lmp_start < LMP_START_DATE and not args.start:
            logger.warning(
                f"LMP data only available from {LMP_START_DATE.date()} via OASIS. "
                f"Using {LMP_START_DATE.date()} as start."
            )
            lmp_start = LMP_START_DATE

        logger.info(f"CAISO LMP fetch: {lmp_start.date()} → {lmp_end.date()}")
        n = fetch_lmp(lmp_start, lmp_end, OUTPUT_DIR)
        logger.info(f"LMP fetch complete: {n} months")

    logger.info("Done.")


if __name__ == "__main__":
    main()
