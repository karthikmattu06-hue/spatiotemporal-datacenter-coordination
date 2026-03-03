"""
Visualize grid stress patterns for a single region.

Generates:
1. Load duration curve (how often load exceeds each level)
2. Stress hour heatmap (hour-of-day × month)
3. August 2020 heat wave time series (CAISO sanity check)
4. Monthly stress hour bar chart

Usage:
    python -m src.plotting.visualize_stress
    python -m src.plotting.visualize_stress --region caiso --no-aug2020
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from src.data_collection.config import PROCESSED_DIR, FIGURES_DIR, LOG_LEVEL
except ImportError:
    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
    LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})


def plot_load_duration_curve(df: pd.DataFrame, region: str, out_dir: Path):
    """Load duration curve: sorted load vs. percentage of hours exceeded."""
    if "load_mw" not in df.columns:
        logger.warning("No load_mw column — skipping load duration curve")
        return

    load = df["load_mw"].dropna().sort_values(ascending=False).values
    pct = np.linspace(0, 100, len(load))

    fig, ax = plt.subplots()
    ax.plot(pct, load / 1000, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Percentage of Hours Exceeded (%)")
    ax.set_ylabel("Load (GW)")
    ax.set_title(f"{region.upper()} — Load Duration Curve (2020–2024)")
    ax.grid(True, alpha=0.3)

    # Mark stress threshold (P90)
    p90 = np.percentile(load, 90)
    ax.axhline(y=p90 / 1000, color="#dc2626", linestyle="--", alpha=0.7,
               label=f"P90 threshold: {p90/1000:.1f} GW")
    ax.legend()

    path = out_dir / f"{region}_load_duration_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_stress_heatmap(df: pd.DataFrame, region: str, out_dir: Path):
    """Heatmap of stress hour frequency: hour-of-day (y) × month (x)."""
    if "stress_any" not in df.columns:
        logger.warning("No stress_any column — skipping heatmap")
        return

    df_local = df.copy()
    df_local["hour"] = df_local.index.hour
    df_local["month"] = df_local.index.month

    # Count stress hours per (month, hour) cell
    stress_counts = (
        df_local[df_local["stress_any"]]
        .groupby(["month", "hour"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure full 12×24 grid
    full_index = pd.RangeIndex(1, 13, name="month")
    full_cols = pd.RangeIndex(0, 24, name="hour")
    stress_counts = stress_counts.reindex(index=full_index, columns=full_cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        stress_counts.T.values,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        origin="lower",
    )
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(0, 24, 3))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of Day (UTC)")
    ax.set_title(f"{region.upper()} — Grid Stress Frequency by Month & Hour (2020–2024)")
    plt.colorbar(im, ax=ax, label="Stress Hours (count over 5 years)")

    path = out_dir / f"{region}_stress_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_aug2020_event(df: pd.DataFrame, region: str, out_dir: Path):
    """Time series of the August 2020 CAISO heat wave event.

    The rolling blackouts occurred Aug 14-19, 2020.
    This serves as a sanity check that our data captures known stress events.
    """
    start = pd.Timestamp("2020-08-10", tz="UTC")
    end = pd.Timestamp("2020-08-25", tz="UTC")

    window = df.loc[start:end].copy()
    if window.empty:
        logger.warning("No data for Aug 2020 — skipping event plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Load plot
    if "load_mw" in window.columns:
        ax = axes[0]
        ax.plot(window.index, window["load_mw"] / 1000, color="#2563eb", linewidth=1)
        if "stress_any" in window.columns:
            stress_mask = window["stress_any"]
            ax.fill_between(
                window.index, 0, window["load_mw"] / 1000,
                where=stress_mask, alpha=0.3, color="#dc2626", label="Stress hours"
            )
        ax.set_ylabel("Load (GW)")
        ax.set_title(f"{region.upper()} — August 2020 Heat Wave Event")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark the rolling blackout dates
        for date in ["2020-08-14", "2020-08-15"]:
            ax.axvline(
                pd.Timestamp(date, tz="UTC"), color="#9333ea",
                linestyle=":", alpha=0.5
            )
        ax.text(
            pd.Timestamp("2020-08-14T12:00", tz="UTC"),
            ax.get_ylim()[1] * 0.95,
            "Rolling\nblackouts",
            ha="center", va="top", fontsize=9, color="#9333ea",
        )

    # LMP plot
    if "lmp_usd_mwh" in window.columns:
        ax = axes[1]
        ax.plot(window.index, window["lmp_usd_mwh"], color="#059669", linewidth=1)
        ax.set_ylabel("LMP ($/MWh)")
        ax.set_xlabel("Date (UTC)")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.tight_layout()
    path = out_dir / f"{region}_aug2020_event.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_monthly_stress_bars(df: pd.DataFrame, region: str, out_dir: Path):
    """Bar chart: total stress hours per calendar month (averaged over years)."""
    if "stress_any" not in df.columns:
        return

    df_stress = df[df["stress_any"]].copy()
    df_stress["year"] = df_stress.index.year
    df_stress["month"] = df_stress.index.month

    # Average stress hours per month across years
    monthly = df_stress.groupby(["year", "month"]).size().reset_index(name="hours")
    monthly_avg = monthly.groupby("month")["hours"].agg(["mean", "std"]).reset_index()

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots()
    bars = ax.bar(
        range(1, 13),
        monthly_avg["mean"],
        yerr=monthly_avg["std"],
        capsize=3,
        color="#f59e0b",
        edgecolor="#d97706",
        alpha=0.85,
    )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_ylabel("Average Stress Hours per Month")
    ax.set_title(f"{region.upper()} — Monthly Grid Stress Distribution (2020–2024)")
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / f"{region}_monthly_stress.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize grid stress patterns")
    parser.add_argument("--region", type=str, default="caiso")
    parser.add_argument("--no-aug2020", action="store_true",
                        help="Skip August 2020 event plot")
    args = parser.parse_args()

    region = args.region.lower()
    stress_path = PROCESSED_DIR / region / f"{region}_stress_annotated.parquet"

    if not stress_path.exists():
        logger.error(f"Stress-annotated data not found: {stress_path}")
        logger.error("Run stress_analysis.py first.")
        return

    logger.info(f"Loading {stress_path}...")
    df = pd.read_parquet(stress_path)

    out_dir = FIGURES_DIR / region
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_load_duration_curve(df, region, out_dir)
    plot_stress_heatmap(df, region, out_dir)
    plot_monthly_stress_bars(df, region, out_dir)

    if not args.no_aug2020 and region == "caiso":
        plot_aug2020_event(df, region, out_dir)

    logger.info("All plots generated.")


if __name__ == "__main__":
    main()
