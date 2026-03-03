"""
Clean and validate CAISO raw data.

Reads monthly CSVs from data/raw/caiso/, handles missing values,
normalizes timestamps to UTC, and outputs cleaned parquet files
to data/processed/caiso/.

Handles mismatched date ranges: load (2020-2024) vs LMP (2023-2024).

Usage:
    python -m src.data_collection.clean_caiso
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from .config import RAW_DIR, PROCESSED_DIR, LOG_LEVEL
except ImportError:
    from config import RAW_DIR, PROCESSED_DIR, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_CAISO = RAW_DIR / "caiso"
OUT_DIR = PROCESSED_DIR / "caiso"


def load_raw_csvs(prefix: str, raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all monthly CSVs matching a prefix."""
    files = sorted(raw_dir.glob(f"{prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.csv files found in {raw_dir}")

    logger.info(f"Loading {len(files)} {prefix} files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=True)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {prefix}: {len(combined)} rows from {len(dfs)} files")
    return combined


def find_column(df: pd.DataFrame, candidates: list[str], fallback_pattern: str = None) -> str:
    """Find a column by trying candidate names, then optional pattern match."""
    for c in candidates:
        if c in df.columns:
            return c
    if fallback_pattern:
        matches = [c for c in df.columns if fallback_pattern.lower() in c.lower()]
        if matches:
            logger.info(f"Using '{matches[0]}' (matched pattern '{fallback_pattern}')")
            return matches[0]
    raise ValueError(
        f"Cannot find column. Tried: {candidates}. "
        f"Available: {list(df.columns)}"
    )


def clean_load(df: pd.DataFrame) -> pd.DataFrame:
    """Clean load data → hourly UTC DataFrame with 'load_mw' column."""
    # Find time column
    time_col = find_column(df, ["Time", "time", "Interval Start", "interval_start"])

    df["timestamp"] = pd.to_datetime(df[time_col], utc=True)

    # Find load column
    load_col = find_column(df, ["Load", "load", "Load (MW)", "System Load"], fallback_pattern="load")

    out = df[["timestamp", load_col]].copy()
    out.rename(columns={load_col: "load_mw"}, inplace=True)
    out["load_mw"] = pd.to_numeric(out["load_mw"], errors="coerce")

    # Resample to hourly (gridstatus may return 5-min intervals)
    out = out.set_index("timestamp").sort_index()
    out = out.resample("h").mean()
    out = out[~out.index.duplicated(keep="first")]

    # Interpolate gaps ≤ 6 hours
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


def clean_lmp(df: pd.DataFrame) -> pd.DataFrame:
    """Clean LMP data → hourly UTC DataFrame with 'lmp_usd_mwh' column."""
    time_col = find_column(df, ["Time", "time", "Interval Start", "interval_start"])

    df["timestamp"] = pd.to_datetime(df[time_col], utc=True)

    lmp_col = find_column(df, ["LMP", "lmp", "LMP ($/MWh)", "Price"], fallback_pattern="lmp")

    out = df[["timestamp", lmp_col]].copy()
    out.rename(columns={lmp_col: "lmp_usd_mwh"}, inplace=True)
    out["lmp_usd_mwh"] = pd.to_numeric(out["lmp_usd_mwh"], errors="coerce")

    # Average across locations if multiple exist, then resample to hourly
    out = out.set_index("timestamp").sort_index()
    out = out.resample("h").mean()
    out = out[~out.index.duplicated(keep="first")]

    # Flag extremes but don't clip (LMPs can legitimately spike or go negative)
    extreme_high = (out["lmp_usd_mwh"] > 1000).sum()
    extreme_low = (out["lmp_usd_mwh"] < -50).sum()
    if extreme_high > 0:
        logger.info(f"LMP: {extreme_high} hours > $1000/MWh (kept as-is)")
    if extreme_low > 0:
        logger.info(f"LMP: {extreme_low} hours < -$50/MWh (kept as-is)")

    n_missing = out["lmp_usd_mwh"].isna().sum()
    if n_missing > 0:
        pct = 100 * n_missing / len(out)
        logger.warning(f"LMP: {n_missing} missing hours ({pct:.2f}%)")
        out["lmp_usd_mwh"] = out["lmp_usd_mwh"].interpolate(
            method="linear", limit=6, limit_direction="both"
        )

    return out


def validate(load_df: pd.DataFrame, lmp_df: pd.DataFrame = None):
    """Run validation checks and print summary statistics."""
    logger.info("=" * 60)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 60)

    datasets = [("Load", load_df, "load_mw")]
    if lmp_df is not None:
        datasets.append(("LMP", lmp_df, "lmp_usd_mwh"))

    for name, df, col in datasets:
        logger.info(f"\n--- {name} ---")
        logger.info(f"  Date range: {df.index.min()} → {df.index.max()}")
        logger.info(f"  Total hours: {len(df)}")

        expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        missing_hours = len(expected) - len(df)
        logger.info(f"  Missing hours (gaps): {missing_hours}")
        logger.info(f"  NaN values: {df[col].isna().sum()}")
        logger.info(f"  Min: {df[col].min():.2f}")
        logger.info(f"  Max: {df[col].max():.2f}")
        logger.info(f"  Mean: {df[col].mean():.2f}")
        logger.info(f"  Std: {df[col].std():.2f}")

        if name == "Load":
            if df[col].min() < 0:
                logger.warning(f"  ⚠ Negative load detected!")
            if df[col].max() > 60000:
                logger.warning(f"  ⚠ Load exceeds 60 GW — likely data error")
        elif name == "LMP":
            neg_pct = 100 * (df[col] < 0).sum() / len(df)
            logger.info(f"  Negative LMP hours: {(df[col] < 0).sum()} ({neg_pct:.1f}%)")

    if lmp_df is not None:
        overlap_start = max(load_df.index.min(), lmp_df.index.min())
        overlap_end = min(load_df.index.max(), lmp_df.index.max())
        logger.info(f"\n--- Coverage ---")
        logger.info(f"  Load:    {load_df.index.min().date()} → {load_df.index.max().date()}")
        logger.info(f"  LMP:     {lmp_df.index.min().date()} → {lmp_df.index.max().date()}")
        logger.info(f"  Overlap: {overlap_start.date()} → {overlap_end.date()}")

    logger.info("=" * 60)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Clean load (2020-2024) ────────────────────────────────────────
    logger.info("Cleaning CAISO load data...")
    load_raw = load_raw_csvs("load", RAW_CAISO)
    load_clean = clean_load(load_raw)
    load_path = OUT_DIR / "caiso_load_hourly.parquet"
    load_clean.to_parquet(load_path)
    logger.info(f"Saved: {load_path}")

    # ── Clean LMP (2023-2024, if available) ───────────────────────────
    lmp_clean = None
    lmp_files = sorted(RAW_CAISO.glob("lmp_*.csv"))
    if lmp_files:
        logger.info("Cleaning CAISO LMP data...")
        lmp_raw = load_raw_csvs("lmp", RAW_CAISO)
        lmp_clean = clean_lmp(lmp_raw)
        lmp_path = OUT_DIR / "caiso_lmp_hourly.parquet"
        lmp_clean.to_parquet(lmp_path)
        logger.info(f"Saved: {lmp_path}")
    else:
        logger.info("No LMP files found — skipping. Run fetch_caiso.py with --lmp-only first.")

    # ── Validate ──────────────────────────────────────────────────────
    validate(load_clean, lmp_clean)

    # ── Merged file ───────────────────────────────────────────────────
    # Left join: load is primary (2020-2024), LMP fills in where available (2023-2024).
    # Hours without LMP data will have NaN in lmp_usd_mwh — this is expected.
    if lmp_clean is not None:
        merged = load_clean.join(lmp_clean, how="left")
        lmp_coverage = merged["lmp_usd_mwh"].notna().sum()
        logger.info(
            f"Merged: {len(merged)} hours total, "
            f"{lmp_coverage} with LMP data ({100*lmp_coverage/len(merged):.1f}%)"
        )
    else:
        merged = load_clean.copy()
        logger.info("Merged: load-only (no LMP data available)")

    merged_path = OUT_DIR / "caiso_merged_hourly.parquet"
    merged.to_parquet(merged_path)
    logger.info(f"Saved merged: {merged_path}")


if __name__ == "__main__":
    main()
