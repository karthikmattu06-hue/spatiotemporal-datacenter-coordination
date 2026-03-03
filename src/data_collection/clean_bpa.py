"""
Clean and validate BPA raw demand data from EIA API v2.

Reads monthly CSVs from data/raw/bpa/ and outputs cleaned parquet to
data/processed/bpa/. BPA is load-only (no LMP — non-ISO market).

EIA period format: YYYY-MM-DDTHH (UTC). Value is hourly MWh (= average MW
over that hour), so it's treated directly as load_mw.

BPA system load range: ~3–12 GW (Pacific Northwest hydro-heavy grid).

Usage:
    python -m src.data_collection.clean_bpa
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

RAW_BPA = RAW_DIR / "bpa"
OUT_DIR = PROCESSED_DIR / "bpa"


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all monthly load CSVs."""
    files = sorted(raw_dir.glob("load_*.csv"))
    if not files:
        raise FileNotFoundError(f"No load_*.csv files found in {raw_dir}")

    logger.info(f"Loading {len(files)} BPA files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined BPA: {len(combined)} rows from {len(dfs)} files")
    return combined


def clean_load(df: pd.DataFrame) -> pd.DataFrame:
    """Clean EIA demand data → hourly UTC DataFrame with 'load_mw' column.

    EIA API columns (from fetch_bpa.py):
        period     : YYYY-MM-DDTHH (UTC)
        respondent : BPAT
        type       : D
        value      : demand in MWh for that hour (= avg MW)
        value-units: megawatthours
    """
    if "period" not in df.columns:
        raise ValueError(
            f"Expected 'period' column from EIA API. Got: {list(df.columns)}"
        )

    # EIA period format: "2020-01-01T01" — UTC hour
    df["timestamp"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)

    if "value" not in df.columns:
        raise ValueError(f"Expected 'value' column. Got: {list(df.columns)}")

    out = df[["timestamp", "value"]].copy()
    out.rename(columns={"value": "load_mw"}, inplace=True)
    out["load_mw"] = pd.to_numeric(out["load_mw"], errors="coerce")

    out = out.set_index("timestamp").sort_index()

    # Drop duplicates (can occur at DST transitions in EIA data)
    out = out[~out.index.duplicated(keep="first")]

    # Resample to enforce uniform hourly index (fills gaps with NaN)
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
    logger.info("VALIDATION REPORT — BPA")
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
    # BPA max is ~15 GW; flag >20 GW as likely data error
    if load_df["load_mw"].max() > 20_000:
        logger.warning("  ⚠ Load exceeds 20 GW — likely data error")

    logger.info("=" * 60)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cleaning BPA demand data (EIA API v2)...")
    raw = load_raw_csvs(RAW_BPA)
    load_clean = clean_load(raw)

    load_path = OUT_DIR / "bpa_load_hourly.parquet"
    load_clean.to_parquet(load_path)
    logger.info(f"Saved: {load_path}")

    validate(load_clean)

    # BPA has no LMP — merged file is load only
    merged_path = OUT_DIR / "bpa_merged_hourly.parquet"
    load_clean.to_parquet(merged_path)
    logger.info(f"Saved merged (load-only): {merged_path}")


if __name__ == "__main__":
    main()
