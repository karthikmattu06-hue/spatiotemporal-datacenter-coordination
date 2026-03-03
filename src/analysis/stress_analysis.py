"""
Grid stress identification and summary statistics.

Region-agnostic: works on any cleaned parquet file with
'load_mw' and optionally 'lmp_usd_mwh' columns indexed by UTC timestamp.

Load is the primary stress signal (2020-2024).
LMP is used for validation only where available (2023-2024 for CAISO).

Usage:
    python -m src.analysis.stress_analysis
    python -m src.analysis.stress_analysis --region caiso
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from src.data_collection.config import (
        PROCESSED_DIR, LOG_LEVEL,
        LOAD_STRESS_PERCENTILE, LMP_SPIKE_SIGMA, MIN_CONSECUTIVE_STRESS_HOURS,
    )
except ImportError:
    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    LOG_LEVEL = "INFO"
    LOAD_STRESS_PERCENTILE = 90
    LMP_SPIKE_SIGMA = 2.0
    MIN_CONSECUTIVE_STRESS_HOURS = 2

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def identify_stress_hours(
    df: pd.DataFrame,
    load_percentile: float = LOAD_STRESS_PERCENTILE,
    lmp_sigma: float = LMP_SPIKE_SIGMA,
    min_consecutive: int = MIN_CONSECUTIVE_STRESS_HOURS,
) -> pd.DataFrame:
    """Flag hours as grid-stressed based on load and optionally LMP.

    Primary signal: Load > seasonal Nth percentile (computed per calendar month).
    Validation signal: LMP > mean + K*sigma (where LMP data exists).

    The 'stress_any' column is driven by load only — LMP is a separate
    validation column ('stress_lmp') that can be compared downstream.

    Parameters
    ----------
    df : DataFrame with 'load_mw' and optionally 'lmp_usd_mwh', UTC datetime index.
    load_percentile : Percentile threshold for load-based stress (0-100).
    lmp_sigma : Std devs above mean for LMP spike detection.
    min_consecutive : Minimum consecutive hours to count as a stress event.

    Returns
    -------
    DataFrame with added columns:
        'stress_load' (bool), 'stress_lmp' (bool or NaN),
        'stress_any' (bool, load-driven), 'stress_event_id' (int)
    """
    out = df.copy()

    # ── Load-based stress (primary, seasonal threshold) ───────────────
    if "load_mw" not in out.columns:
        raise ValueError("load_mw column required for stress identification")

    out["month"] = out.index.month
    monthly_thresholds = out.groupby("month")["load_mw"].quantile(
        load_percentile / 100
    )
    out["load_threshold"] = out["month"].map(monthly_thresholds)
    out["stress_load"] = out["load_mw"] > out["load_threshold"]
    out.drop(columns=["month", "load_threshold"], inplace=True)
    logger.info(
        f"Load stress (P{load_percentile}, seasonal): "
        f"{out['stress_load'].sum()} hours "
        f"({100 * out['stress_load'].mean():.1f}%)"
    )

    # ── LMP-based stress (validation only, where data exists) ─────────
    has_lmp = "lmp_usd_mwh" in out.columns
    if has_lmp:
        lmp_valid = out["lmp_usd_mwh"].notna()
        n_lmp = lmp_valid.sum()
        if n_lmp > 0:
            lmp_data = out.loc[lmp_valid, "lmp_usd_mwh"]
            lmp_mean = lmp_data.mean()
            lmp_std = lmp_data.std()
            lmp_threshold = lmp_mean + lmp_sigma * lmp_std

            # Only flag stress where LMP data exists; NaN elsewhere
            out["stress_lmp"] = pd.Series(np.nan, index=out.index, dtype="boolean")
            out.loc[lmp_valid, "stress_lmp"] = out.loc[lmp_valid, "lmp_usd_mwh"] > lmp_threshold

            lmp_stress_count = out["stress_lmp"].sum()
            logger.info(
                f"LMP stress (>{lmp_threshold:.2f} $/MWh): "
                f"{lmp_stress_count} hours out of {n_lmp} with data "
                f"({100 * lmp_stress_count / n_lmp:.1f}%)"
            )

            # Validation: overlap between load stress and LMP stress
            overlap_mask = lmp_valid & out["stress_load"]
            if overlap_mask.any():
                both = (out.loc[overlap_mask, "stress_load"] & out.loc[overlap_mask, "stress_lmp"]).sum()
                load_only = overlap_mask.sum()
                logger.info(
                    f"LMP validation: {both}/{load_only} load-stress hours "
                    f"also show LMP stress ({100 * both / load_only:.1f}% agreement)"
                )
        else:
            out["stress_lmp"] = pd.Series(np.nan, index=out.index, dtype="boolean")
            logger.info("LMP: no valid data points — skipping LMP stress")
    else:
        out["stress_lmp"] = pd.Series(np.nan, index=out.index, dtype="boolean")
        logger.info("No lmp_usd_mwh column — LMP validation skipped")

    # ── Combined stress (load-driven only) ────────────────────────────
    out["stress_any"] = out["stress_load"]

    # ── Filter short blips ────────────────────────────────────────────
    out["_group"] = (out["stress_any"] != out["stress_any"].shift()).cumsum()
    stress_groups = out[out["stress_any"]].groupby("_group").size()
    short_groups = stress_groups[stress_groups < min_consecutive].index
    out.loc[out["_group"].isin(short_groups), "stress_any"] = False

    # Assign event IDs
    out["_group"] = (out["stress_any"] != out["stress_any"].shift()).cumsum()
    event_counter = 0
    event_ids = pd.Series(0, index=out.index, dtype=int)
    for gid, group in out[out["stress_any"]].groupby("_group"):
        event_counter += 1
        event_ids.loc[group.index] = event_counter
    out["stress_event_id"] = event_ids
    out.drop(columns=["_group"], inplace=True)

    n_events = out["stress_event_id"].max()
    logger.info(
        f"After filtering (<{min_consecutive}h blips removed): "
        f"{out['stress_any'].sum()} stress hours across {n_events} events"
    )

    return out


def stress_summary(df: pd.DataFrame, region: str = "unknown") -> dict:
    """Compute summary statistics for stress analysis."""
    if "stress_any" not in df.columns:
        raise ValueError("Run identify_stress_hours() first")

    total_hours = len(df)
    stress_hours = int(df["stress_any"].sum())
    n_events = int(df["stress_event_id"].max())

    event_durations = (
        df[df["stress_event_id"] > 0]
        .groupby("stress_event_id")
        .size()
    )

    df_stress = df[df["stress_any"]]
    seasonal = df_stress.groupby(df_stress.index.month).size()
    diurnal = df_stress.groupby(df_stress.index.hour).size()

    # LMP validation stats (where available)
    lmp_agreement = None
    if "stress_lmp" in df.columns:
        lmp_valid = df["stress_lmp"].notna()
        if lmp_valid.any():
            overlap = lmp_valid & df["stress_load"]
            if overlap.any():
                both = int((df.loc[overlap, "stress_load"] & df.loc[overlap, "stress_lmp"]).sum())
                total_overlap = int(overlap.sum())
                lmp_agreement = round(100 * both / total_overlap, 1) if total_overlap > 0 else None

    summary = {
        "region": region,
        "total_hours": total_hours,
        "stress_hours": stress_hours,
        "stress_pct": round(100 * stress_hours / total_hours, 2),
        "n_events": n_events,
        "event_duration_mean": round(float(event_durations.mean()), 1) if len(event_durations) > 0 else 0,
        "event_duration_median": round(float(event_durations.median()), 1) if len(event_durations) > 0 else 0,
        "event_duration_max": int(event_durations.max()) if len(event_durations) > 0 else 0,
        "peak_stress_month": int(seasonal.idxmax()) if len(seasonal) > 0 else None,
        "peak_stress_hour": int(diurnal.idxmax()) if len(diurnal) > 0 else None,
        "lmp_agreement_pct": lmp_agreement,
        "seasonal_distribution": {int(k): int(v) for k, v in seasonal.to_dict().items()},
        "diurnal_distribution": {int(k): int(v) for k, v in diurnal.to_dict().items()},
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"STRESS SUMMARY: {region.upper()}")
    logger.info(f"{'='*50}")
    logger.info(f"  Total hours:       {summary['total_hours']:,}")
    logger.info(f"  Stress hours:      {summary['stress_hours']:,} ({summary['stress_pct']:.1f}%)")
    logger.info(f"  Stress events:     {summary['n_events']}")
    logger.info(f"  Mean duration:     {summary['event_duration_mean']:.1f}h")
    logger.info(f"  Max duration:      {summary['event_duration_max']}h")
    logger.info(f"  Peak month:        {summary['peak_stress_month']}")
    logger.info(f"  Peak hour (UTC):   {summary['peak_stress_hour']}")
    if lmp_agreement is not None:
        logger.info(f"  LMP agreement:     {summary['lmp_agreement_pct']}%")
    else:
        logger.info(f"  LMP agreement:     N/A (no overlap data)")
    logger.info(f"{'='*50}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run grid stress analysis")
    parser.add_argument(
        "--region", type=str, default="caiso",
        help="Region name (matches subdirectory in data/processed/)"
    )
    args = parser.parse_args()

    region = args.region.lower()
    merged_path = PROCESSED_DIR / region / f"{region}_merged_hourly.parquet"

    if not merged_path.exists():
        logger.error(f"Merged data not found: {merged_path}")
        logger.error(f"Run clean_{region}.py first.")
        return

    logger.info(f"Loading {merged_path}...")
    df = pd.read_parquet(merged_path)

    df_stress = identify_stress_hours(df)

    out_path = PROCESSED_DIR / region / f"{region}_stress_annotated.parquet"
    df_stress.to_parquet(out_path)
    logger.info(f"Saved: {out_path}")

    summary = stress_summary(df_stress, region=region)

    summary_path = PROCESSED_DIR / region / f"{region}_stress_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
