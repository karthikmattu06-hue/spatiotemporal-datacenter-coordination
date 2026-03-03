"""
Clean and validate PJM raw data.

Reads monthly CSVs from data/raw/pjm/ (fetched via PJM API hrl_load_metered,
zone=RTO), normalizes timestamps to UTC, and outputs cleaned parquet files
to data/processed/pjm/.

PJM API CSV columns: datetime_beginning_utc, datetime_beginning_ept,
                     nerc_region, mkt_region, zone, load_area, mw, is_verified

PJM RTO load range: ~60–165 GW. No LMP (gridstatus PJM LMP fetches all
pricing nodes causing rate limits; LMP is validation-only anyway).

Usage:
    python -m src.data_collection.clean_pjm
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import RAW_DIR, PROCESSED_DIR, LOG_LEVEL
except ImportError:
    from config import RAW_DIR, PROCESSED_DIR, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_PJM = RAW_DIR / "pjm"
OUT_DIR = PROCESSED_DIR / "pjm"


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all monthly load CSVs."""
    files = sorted(raw_dir.glob("load_*.csv"))
    if not files:
        raise FileNotFoundError(f"No load_*.csv files found in {raw_dir}")

    logger.info(f"Loading {len(files)} PJM files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined PJM: {len(combined)} rows from {len(dfs)} files")
    return combined


def clean_load(df: pd.DataFrame) -> pd.DataFrame:
    """Clean PJM API load data → hourly UTC DataFrame with 'load_mw' column.

    PJM API columns (from fetch_pjm.py, zone=RTO filter):
        datetime_beginning_utc : UTC timestamp (no timezone suffix)
        mw                     : RTO-wide hourly load in MW
    """
    if "datetime_beginning_utc" not in df.columns:
        raise ValueError(
            f"Expected 'datetime_beginning_utc' column from PJM API. Got: {list(df.columns)}"
        )

    # PJM UTC timestamps come without timezone info — force UTC
    df["timestamp"] = pd.to_datetime(df["datetime_beginning_utc"], utc=True)

    if "mw" not in df.columns:
        raise ValueError(f"Expected 'mw' column. Got: {list(df.columns)}")

    out = df[["timestamp", "mw"]].copy()
    out.rename(columns={"mw": "load_mw"}, inplace=True)
    out["load_mw"] = pd.to_numeric(out["load_mw"], errors="coerce")

    out = out.set_index("timestamp").sort_index()
    out = out[~out.index.duplicated(keep="first")]

    # Resample to enforce uniform hourly index
    out = out.resample("h").mean()

    n_missing = out["load_mw"].isna().sum()
    if n_missing > 0:
        pct = 100 * n_missing / len(out)
        logger.warning(f"Load: {n_missing} missing hours ({pct:.2f}%)")
        out["load_mw"] = out["load_mw"].interpolate(
            method="linear", limit=6, limit_direction="both"
        )
        n_still = out["load_mw"].isna().sum()
        if n_still > 0:
            logger.warning(f"Load: {n_still} hours still missing after interpolation")

    return out


def validate(load_df: pd.DataFrame):
    """Run validation checks and print summary statistics."""
    logger.info("=" * 60)
    logger.info("VALIDATION REPORT — PJM")
    logger.info("=" * 60)
    logger.info(f"  Date range:    {load_df.index.min()} → {load_df.index.max()}")
    logger.info(f"  Total hours:   {len(load_df)}")

    expected = pd.date_range(start=load_df.index.min(), end=load_df.index.max(), freq="h")
    missing_hours = len(expected) - len(load_df)
    logger.info(f"  Missing hours (gaps): {missing_hours}")
    logger.info(f"  NaN values:    {load_df['load_mw'].isna().sum()}")
    logger.info(f"  Min: {load_df['load_mw'].min():.2f} MW")
    logger.info(f"  Max: {load_df['load_mw'].max():.2f} MW")
    logger.info(f"  Mean: {load_df['load_mw'].mean():.2f} MW")
    logger.info(f"  Std: {load_df['load_mw'].std():.2f} MW")

    if load_df["load_mw"].min() < 0:
        logger.warning("  ⚠ Negative load detected!")
    # PJM RTO peaks ~165 GW; flag >200 GW as likely data error
    if load_df["load_mw"].max() > 200_000:
        logger.warning("  ⚠ Load exceeds 200 GW — likely data error")

    logger.info("=" * 60)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cleaning PJM load data (PJM API, zone=RTO)...")
    raw = load_raw_csvs(RAW_PJM)
    load_clean = clean_load(raw)

    load_path = OUT_DIR / "pjm_load_hourly.parquet"
    load_clean.to_parquet(load_path)
    logger.info(f"Saved: {load_path}")

    validate(load_clean)

    # PJM is load-only (no LMP due to rate limits; LMP is validation-only)
    merged_path = OUT_DIR / "pjm_merged_hourly.parquet"
    load_clean.to_parquet(merged_path)
    logger.info(f"Saved merged (load-only): {merged_path}")


if __name__ == "__main__":
    main()
