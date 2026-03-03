"""
Three-region complementarity analysis: CAISO, PJM, BPA.

Computes:
  ρ  — Pairwise Pearson correlation of stress_any between regions.
       Target: < 0.3 (low correlation = complementary stress timing).

  σ  — Simultaneous stress frequency: fraction of hours where ≥2 regions
       are stressed at the same time.
       Target: rare (supports the thesis that migration is usually available).

  A  — Fleet availability: for each stress hour in region X, the fraction
       of those hours where ≥1 other region is NOT stressed and can absorb
       migrated load.
       Target: > 95%.

Generates:
  - figures/complementarity/stress_heatmap_overlay.png  (3-panel heatmap)
  - figures/complementarity/pairwise_stress_overlap.png (hourly overlap bars)
  - data/processed/complementarity/complementarity_summary.json

Requires: stress_annotated.parquet for all three regions.
Run stress_analysis.py --region {caiso,pjm,bpa} first.

Usage:
    python -m src.analysis.complementarity_analysis
    python -m src.analysis.complementarity_analysis --regions caiso pjm bpa
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_region(region: str) -> pd.DataFrame | None:
    """Load stress-annotated parquet for a region. Returns None if not found."""
    path = PROCESSED_DIR / region / f"{region}_stress_annotated.parquet"
    if not path.exists():
        logger.warning(f"Not found: {path} — run stress_analysis.py --region {region} first")
        return None
    logger.info(f"Loading {path}...")
    df = pd.read_parquet(path)[["load_mw", "stress_any"]]
    df.columns = [f"{region}_load_mw", f"{region}_stress"]
    return df


def align_regions(region_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align region DataFrames on a common UTC hourly index (intersection)."""
    frames = list(region_dfs.values())
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="inner")

    # Drop any rows where all stress columns are NaN
    stress_cols = [c for c in merged.columns if c.endswith("_stress")]
    merged = merged.dropna(subset=stress_cols)

    logger.info(
        f"Aligned dataset: {len(merged)} hours "
        f"({merged.index.min().date()} → {merged.index.max().date()})"
    )
    return merged


# ── Metric computation ────────────────────────────────────────────────────────

def compute_rho(df: pd.DataFrame, regions: list[str]) -> dict:
    """Pearson correlation of stress_any between each pair of regions.

    Binary stress indicators (0/1) — Pearson r on binary vars is equivalent
    to phi coefficient, appropriate for this use case.
    """
    stress_cols = {r: f"{r}_stress" for r in regions}
    results = {}

    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            pair = f"{r1}_{r2}"
            s1 = df[stress_cols[r1]].astype(float)
            s2 = df[stress_cols[r2]].astype(float)
            rho = float(s1.corr(s2))
            results[pair] = round(rho, 4)
            status = "✓ GOOD" if rho < 0.3 else "✗ HIGH"
            logger.info(f"  ρ({r1.upper()}, {r2.upper()}) = {rho:.4f}  {status} (target < 0.3)")

    return results


def compute_sigma(df: pd.DataFrame, regions: list[str]) -> dict:
    """Fraction of hours where ≥2 (and ≥3) regions are simultaneously stressed."""
    stress_cols = [f"{r}_stress" for r in regions]
    stress_count = df[stress_cols].astype(float).sum(axis=1)

    total = len(df)
    two_plus = int((stress_count >= 2).sum())
    three_plus = int((stress_count >= 3).sum()) if len(regions) >= 3 else 0

    sigma_2 = round(100 * two_plus / total, 2)
    sigma_3 = round(100 * three_plus / total, 2)

    logger.info(f"  σ(≥2 regions stressed): {two_plus} hours ({sigma_2}%)")
    if len(regions) >= 3:
        logger.info(f"  σ(≥3 regions stressed): {three_plus} hours ({sigma_3}%)")

    return {
        "hours_2_plus": two_plus,
        "pct_2_plus": sigma_2,
        "hours_3_plus": three_plus,
        "pct_3_plus": sigma_3,
        "total_hours": total,
    }


def compute_availability(df: pd.DataFrame, regions: list[str]) -> dict:
    """For each region, fraction of its stress hours where ≥1 other region is not stressed.

    Availability A_r = P(∃ r' ≠ r : NOT stressed(r') | stressed(r))
    This represents: when region r needs to offload jobs, how often is at least
    one other region available to absorb them?
    """
    results = {}
    stress_cols = {r: f"{r}_stress" for r in regions}

    for region in regions:
        stressed_mask = df[stress_cols[region]].astype(bool)
        stressed_df = df[stressed_mask]

        if len(stressed_df) == 0:
            logger.warning(f"  {region.upper()}: no stress hours found")
            results[region] = None
            continue

        other_regions = [r for r in regions if r != region]
        other_stress = stressed_df[[stress_cols[r] for r in other_regions]].astype(float)

        # Available = at least one other region is NOT stressed
        any_other_not_stressed = (other_stress < 1).any(axis=1)
        available_hours = int(any_other_not_stressed.sum())
        total_stress_hours = len(stressed_df)
        availability_pct = round(100 * available_hours / total_stress_hours, 2)

        status = "✓ GOOD" if availability_pct >= 95 else "✗ LOW"
        logger.info(
            f"  A({region.upper()}): {available_hours}/{total_stress_hours} stress hours "
            f"have ≥1 available region ({availability_pct}%)  {status}"
        )
        results[region] = {
            "stress_hours": total_stress_hours,
            "available_hours": available_hours,
            "availability_pct": availability_pct,
        }

    return results


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_heatmap_overlay(df: pd.DataFrame, regions: list[str], out_dir: Path):
    """Side-by-side stress heatmaps for all regions on the same color scale."""
    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    # Compute per-region heatmaps and find global max for shared scale
    all_counts = []
    for region in regions:
        col = f"{region}_stress"
        local = df[[col]].copy()
        local["hour"] = local.index.hour
        local["month"] = local.index.month
        counts = (
            local[local[col].astype(bool)]
            .groupby(["month", "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(1, 13), columns=range(0, 24), fill_value=0)
        )
        all_counts.append(counts)

    vmax = max(c.values.max() for c in all_counts)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for ax, region, counts in zip(axes, regions, all_counts):
        im = ax.imshow(
            counts.T.values,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(0, 24, 3))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
        ax.set_title(f"{region.upper()}")
        ax.set_xlabel("Month")
        if ax == axes[0]:
            ax.set_ylabel("Hour of Day (UTC)")

    plt.colorbar(im, ax=axes[-1], label="Stress Hours (count)")
    fig.suptitle("Grid Stress Frequency by Region, Month & Hour (UTC)", y=1.02)
    fig.tight_layout()

    path = out_dir / "stress_heatmap_overlay.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_diurnal_overlap(df: pd.DataFrame, regions: list[str], out_dir: Path):
    """Average hourly stress rate per region, overlaid on one plot.

    Shows the temporal offset between CAISO (~00-04 UTC) and PJM (~18-22 UTC).
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {"caiso": "#2563eb", "pjm": "#dc2626", "bpa": "#059669"}
    default_colors = ["#2563eb", "#dc2626", "#059669", "#9333ea"]

    for i, region in enumerate(regions):
        col = f"{region}_stress"
        hourly = df.groupby(df.index.hour)[col].mean() * 100
        color = colors.get(region, default_colors[i % len(default_colors)])
        ax.plot(range(24), hourly, marker="o", markersize=4, linewidth=2,
                color=color, label=region.upper())

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Stress Frequency (%)")
    ax.set_title("Regional Stress by Hour of Day — Temporal Offset (UTC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate known peak windows
    ax.axvspan(0, 4, alpha=0.05, color="#2563eb", label="_CAISO peak")
    ax.axvspan(18, 23, alpha=0.05, color="#dc2626", label="_PJM peak")

    path = out_dir / "pairwise_stress_overlap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_simultaneous_stress(df: pd.DataFrame, regions: list[str], out_dir: Path):
    """Monthly breakdown of simultaneous multi-region stress hours."""
    stress_cols = [f"{r}_stress" for r in regions]
    stress_count = df[stress_cols].astype(float).sum(axis=1)

    df_plot = pd.DataFrame({
        "month": df.index.month,
        "n_stressed": stress_count,
    })

    monthly = df_plot.groupby("month")["n_stressed"].apply(
        lambda x: {
            "one": int((x == 1).sum()),
            "two_plus": int((x >= 2).sum()),
        }
    )

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    one_region = [monthly[m]["one"] if m in monthly else 0 for m in range(1, 13)]
    two_plus = [monthly[m]["two_plus"] if m in monthly else 0 for m in range(1, 13)]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(1, 13)
    ax.bar(x, one_region, label="1 region stressed", color="#93c5fd", alpha=0.85)
    ax.bar(x, two_plus, bottom=one_region, label="≥2 regions stressed", color="#dc2626", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(months)
    ax.set_ylabel("Total Stress Hours")
    ax.set_title("Monthly Stress Hours by Number of Simultaneously Stressed Regions")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / "simultaneous_stress_monthly.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Summary and main ──────────────────────────────────────────────────────────

def build_summary(
    regions: list[str],
    rho: dict,
    sigma: dict,
    availability: dict,
    df: pd.DataFrame,
) -> dict:
    """Build the full complementarity summary dict."""
    return {
        "regions": regions,
        "date_range": {
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "total_hours": len(df),
        },
        "rho": rho,
        "sigma": sigma,
        "availability": availability,
        "thesis_check": {
            "rho_all_below_0.3": all(v < 0.3 for v in rho.values()),
            "sigma_2plus_pct": sigma["pct_2_plus"],
            "availability_all_above_95": all(
                v is not None and v["availability_pct"] >= 95
                for v in availability.values()
            ),
        },
    }


def print_thesis_verdict(summary: dict):
    checks = summary["thesis_check"]
    logger.info("\n" + "=" * 60)
    logger.info("COMPLEMENTARITY THESIS CHECK")
    logger.info("=" * 60)

    rho_ok = checks["rho_all_below_0.3"]
    sigma_pct = checks["sigma_2plus_pct"]
    avail_ok = checks["availability_all_above_95"]

    logger.info(f"  ρ < 0.3 (all pairs):      {'✓ PASS' if rho_ok else '✗ FAIL'}")
    logger.info(f"  σ (≥2 regions, %):         {sigma_pct}%  {'(low ✓)' if sigma_pct < 20 else '(high — investigate)'}")
    logger.info(f"  A > 95% (all regions):     {'✓ PASS' if avail_ok else '✗ FAIL'}")

    all_pass = rho_ok and avail_ok and sigma_pct < 20
    logger.info(f"\n  Overall: {'✓ THESIS SUPPORTED' if all_pass else '✗ THESIS WEAKENED — review numbers'}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Three-region complementarity analysis")
    parser.add_argument(
        "--regions", nargs="+", default=["caiso", "pjm", "bpa"],
        help="Regions to analyze (default: caiso pjm bpa)"
    )
    args = parser.parse_args()

    regions = [r.lower() for r in args.regions]
    logger.info(f"Analyzing regions: {', '.join(r.upper() for r in regions)}")

    # ── Load data ─────────────────────────────────────────────────────
    region_dfs = {}
    for region in regions:
        df = load_region(region)
        if df is not None:
            region_dfs[region] = df

    if len(region_dfs) < 2:
        logger.error("Need stress-annotated data for at least 2 regions. Run stress_analysis.py first.")
        return

    available_regions = list(region_dfs.keys())
    if len(available_regions) < len(regions):
        missing = set(regions) - set(available_regions)
        logger.warning(f"Proceeding without: {', '.join(missing)}")

    # ── Align ─────────────────────────────────────────────────────────
    df = align_regions(region_dfs)

    # ── Compute metrics ───────────────────────────────────────────────
    logger.info("\n--- Pairwise stress correlation (ρ) ---")
    rho = compute_rho(df, available_regions)

    logger.info("\n--- Simultaneous stress frequency (σ) ---")
    sigma = compute_sigma(df, available_regions)

    logger.info("\n--- Fleet availability (A) ---")
    availability = compute_availability(df, available_regions)

    # ── Save summary ──────────────────────────────────────────────────
    summary = build_summary(available_regions, rho, sigma, availability, df)
    print_thesis_verdict(summary)

    out_dir = PROCESSED_DIR / "complementarity"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "complementarity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    fig_dir = FIGURES_DIR / "complementarity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmap_overlay(df, available_regions, fig_dir)
    plot_diurnal_overlap(df, available_regions, fig_dir)
    plot_simultaneous_stress(df, available_regions, fig_dir)

    logger.info("Complementarity analysis complete.")


if __name__ == "__main__":
    main()
