"""
Simulation results visualizer.

Reads all summary JSONs from data/processed/simulation/ and generates:
  Fig 1 — curtailment_comparison.png  : curtailment% by strategy × region
  Fig 2 — qos_comparison.png          : QoS cost by strategy × region
  Fig 3 — oracle_gap.png              : MIP vs Oracle gap by region
  Fig 4 — ensemble_sensitivity.png    : MIP metrics across ensembles 1–4
  Fig 5 — action_breakdown.png        : MIP action mix (migrate/DVFS/pause/nothing)

Usage:
    python -m src.plotting.visualize_simulation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    from src.data_collection.config import PROCESSED_DIR, FIGURES_DIR, LOG_LEVEL
except ImportError:
    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    FIGURES_DIR   = Path(__file__).resolve().parents[2] / "figures"
    LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SIM_DIR = PROCESSED_DIR / "simulation"
OUT_DIR = FIGURES_DIR / "simulation"

REGIONS = ["caiso", "pjm", "bpa"]
REGION_LABELS = {"caiso": "CAISO", "pjm": "PJM", "bpa": "BPA"}
ENSEMBLES = [1, 2, 3, 4]

# Strategy display order and colours
STRATEGY_ORDER = [
    "mip", "oracle_optimal",
    "temporal_only", "spatial_naive", "no_coordination",
]
STRATEGY_LABELS = {
    "mip":             "MIP (ours)",
    "oracle_optimal":  "Oracle Optimal",
    "temporal_only":   "Temporal-Only",
    "spatial_naive":   "Spatial-Naïve",
    "no_coordination": "No Coordination",
}
STRATEGY_COLORS = {
    "mip":             "#2563eb",
    "oracle_optimal":  "#7c3aed",
    "temporal_only":   "#f59e0b",
    "spatial_naive":   "#10b981",
    "no_coordination": "#94a3b8",
}

# Shared rcParams
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_summaries() -> pd.DataFrame:
    """Load all summary JSONs into a flat DataFrame."""
    rows = []
    for region in REGIONS:
        for eid in ENSEMBLES:
            path = SIM_DIR / f"summary_{region}_e{eid}.json"
            if not path.exists():
                logger.warning(f"Missing: {path}")
                continue
            with open(path) as f:
                s = json.load(f)
            for strategy, stats in s["strategies"].items():
                rows.append({
                    "region":    region,
                    "ensemble":  eid,
                    "strategy":  strategy,
                    "n_events":  s["n_events"],
                    "n_hours":   s["n_hours"],
                    **stats,
                })
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} strategy×region×ensemble records")
    return df


def ensemble_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across ensembles for each region × strategy."""
    return (
        df.groupby(["region", "strategy"])
        .agg(
            curtailment_pct_mean=("curtailment_pct_mean", "mean"),
            curtailment_pct_std=("curtailment_pct_mean", "std"),
            total_qos_cost_mean=("total_qos_cost_mean", "mean"),
            total_qos_cost_std=("total_qos_cost_mean", "std"),
            n_migrated_mean=("n_migrated_mean", "mean"),
            n_paused_mean=("n_paused_mean", "mean"),
            feasible_pct=("feasible_pct", "mean"),
        )
        .reset_index()
    )


# ── Figure 1: Curtailment comparison ─────────────────────────────────────────

def plot_curtailment_comparison(avg: pd.DataFrame, out_dir: Path):
    """Grouped bar: mean curtailment% per strategy, grouped by region."""
    fig, ax = plt.subplots(figsize=(12, 5))

    n_strategies = len(STRATEGY_ORDER)
    n_regions    = len(REGIONS)
    group_width  = 0.8
    bar_width    = group_width / n_strategies
    group_positions = np.arange(n_regions)

    for i, strategy in enumerate(STRATEGY_ORDER):
        offsets = group_positions + (i - n_strategies / 2 + 0.5) * bar_width
        vals, errs = [], []
        for region in REGIONS:
            row = avg[(avg["region"] == region) & (avg["strategy"] == strategy)]
            vals.append(float(row["curtailment_pct_mean"].iloc[0]) if len(row) else 0)
            errs.append(float(row["curtailment_pct_std"].iloc[0]) if len(row) else 0)
        ax.bar(
            offsets, vals,
            width=bar_width * 0.9,
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
            yerr=errs, capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.6},
        )

    ax.set_xticks(group_positions)
    ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS])
    ax.set_ylabel("Mean Curtailment Achieved (% of grid load)")
    ax.set_title("Curtailment Performance by Strategy and Region\n(mean ± std across ensembles 1–4)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    path = out_dir / "curtailment_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Figure 2: QoS cost comparison ────────────────────────────────────────────

def plot_qos_comparison(avg: pd.DataFrame, out_dir: Path):
    """Grouped bar: mean QoS cost per strategy, grouped by region."""
    fig, ax = plt.subplots(figsize=(12, 5))

    n_strategies = len(STRATEGY_ORDER)
    n_regions    = len(REGIONS)
    group_width  = 0.8
    bar_width    = group_width / n_strategies
    group_positions = np.arange(n_regions)

    for i, strategy in enumerate(STRATEGY_ORDER):
        offsets = group_positions + (i - n_strategies / 2 + 0.5) * bar_width
        vals, errs = [], []
        for region in REGIONS:
            row = avg[(avg["region"] == region) & (avg["strategy"] == strategy)]
            vals.append(float(row["total_qos_cost_mean"].iloc[0]) if len(row) else 0)
            errs.append(float(row["total_qos_cost_std"].iloc[0]) if len(row) else 0)
        ax.bar(
            offsets, vals,
            width=bar_width * 0.9,
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
            yerr=errs, capsize=2, error_kw={"elinewidth": 0.8, "alpha": 0.6},
        )

    ax.set_xticks(group_positions)
    ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS])
    ax.set_ylabel("Mean QoS Cost (Σ w_j · q_j)")
    ax.set_title("QoS Degradation Cost by Strategy and Region\n(mean ± std across ensembles 1–4, lower is better)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    path = out_dir / "qos_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Figure 3: Oracle gap ──────────────────────────────────────────────────────

def plot_oracle_gap(avg: pd.DataFrame, out_dir: Path):
    """Side-by-side MIP vs Oracle bars per region, annotated with gap-closure %."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for ax, region in zip(axes, REGIONS):
        mip_row = avg[(avg["region"] == region) & (avg["strategy"] == "mip")]
        ora_row = avg[(avg["region"] == region) & (avg["strategy"] == "oracle_optimal")]

        mip_val = float(mip_row["curtailment_pct_mean"].iloc[0])
        ora_val = float(ora_row["curtailment_pct_mean"].iloc[0])
        mip_err = float(mip_row["curtailment_pct_std"].iloc[0])
        ora_err = float(ora_row["curtailment_pct_std"].iloc[0])

        bars = ax.bar(
            [0, 1], [mip_val, ora_val],
            color=[STRATEGY_COLORS["mip"], STRATEGY_COLORS["oracle_optimal"]],
            width=0.5,
            yerr=[mip_err, ora_err],
            capsize=4,
            error_kw={"elinewidth": 1, "alpha": 0.7},
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MIP\n(online)", "Oracle\n(foresight)"])
        ax.set_title(REGION_LABELS[region])
        ax.set_ylabel("Curtailment (%)" if region == "caiso" else "")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

        # Gap closure annotation
        if ora_val > 0:
            closure = 100 * mip_val / ora_val
            gap_label = f"{closure:.0f}% closure"
        else:
            gap_label = "N/A"

        ymax = max(mip_val, ora_val)
        ax.annotate(
            gap_label,
            xy=(0.5, ymax * 1.05),
            ha="center", va="bottom",
            fontsize=9, color="#374151",
            xycoords=("data", "data"),
        )

    fig.suptitle(
        "Online MIP vs. Oracle Optimal Curtailment\n"
        "(gap closure = MIP / Oracle, averaged across ensembles 1–4)",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / "oracle_gap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Figure 4: Ensemble sensitivity ────────────────────────────────────────────

def plot_ensemble_sensitivity(df: pd.DataFrame, out_dir: Path):
    """MIP curtailment% and QoS cost across ensembles 1–4, one line per region."""
    mip = df[df["strategy"] == "mip"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    region_colors = {"caiso": "#2563eb", "pjm": "#f59e0b", "bpa": "#10b981"}
    region_markers = {"caiso": "o", "pjm": "s", "bpa": "^"}

    for region in REGIONS:
        sub = mip[mip["region"] == region].sort_values("ensemble")
        c, m = region_colors[region], region_markers[region]

        axes[0].plot(
            sub["ensemble"], sub["curtailment_pct_mean"],
            color=c, marker=m, linewidth=1.5, markersize=6,
            label=REGION_LABELS[region],
        )
        axes[1].plot(
            sub["ensemble"], sub["total_qos_cost_mean"],
            color=c, marker=m, linewidth=1.5, markersize=6,
            label=REGION_LABELS[region],
        )

    axes[0].set_xlabel("Ensemble ID")
    axes[0].set_ylabel("Mean Curtailment Achieved (%)")
    axes[0].set_title("MIP Curtailment% by Ensemble")
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_xticklabels([
        "E1\n(80% train)", "E2\n(50% train)", "E3\n(50% train\nno Flex0)", "E4\n(90% train)"
    ])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Ensemble ID")
    axes[1].set_ylabel("Mean QoS Cost")
    axes[1].set_title("MIP QoS Cost by Ensemble")
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xticklabels([
        "E1\n(80% train)", "E2\n(50% train)", "E3\n(50% train\nno Flex0)", "E4\n(90% train)"
    ])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("MIP Sensitivity to Workload Ensemble Composition", fontsize=12)
    fig.tight_layout()
    path = out_dir / "ensemble_sensitivity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Figure 5: Action breakdown ────────────────────────────────────────────────

def plot_action_breakdown(df: pd.DataFrame, out_dir: Path):
    """Stacked bar: MIP action mix per region, averaged over ensembles."""
    mip = df[df["strategy"] == "mip"].copy()
    avg_actions = (
        mip.groupby("region")[["n_migrated_mean", "n_paused_mean"]]
        .mean()
        .reindex(REGIONS)
    )

    # Load parquet files to get n_dvfs and n_nothing (not in summary JSON)
    dvfs_nothing = {r: {"dvfs": 0.0, "nothing": 0.0} for r in REGIONS}
    for region in REGIONS:
        dvfs_vals, nothing_vals = [], []
        for eid in ENSEMBLES:
            path = SIM_DIR / f"results_{region}_e{eid}.parquet"
            if not path.exists():
                continue
            results = pd.read_parquet(path)
            mip_rows = results[results["strategy"] == "mip"]
            dvfs_vals.append(mip_rows["n_dvfs"].mean())
            nothing_vals.append(mip_rows["n_nothing"].mean())
        if dvfs_vals:
            dvfs_nothing[region]["dvfs"]    = float(np.mean(dvfs_vals))
            dvfs_nothing[region]["nothing"] = float(np.mean(nothing_vals))

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(REGIONS))
    bar_width = 0.45

    migrated = [avg_actions.loc[r, "n_migrated_mean"] for r in REGIONS]
    dvfs     = [dvfs_nothing[r]["dvfs"]    for r in REGIONS]
    paused   = [avg_actions.loc[r, "n_paused_mean"] for r in REGIONS]
    nothing  = [dvfs_nothing[r]["nothing"] for r in REGIONS]

    action_colors = {
        "migrate": "#2563eb",
        "dvfs":    "#f59e0b",
        "pause":   "#ef4444",
        "nothing": "#d1d5db",
    }

    b1 = ax.bar(x, migrated, bar_width, label="Migrate",  color=action_colors["migrate"])
    b2 = ax.bar(x, dvfs,     bar_width, label="DVFS",     color=action_colors["dvfs"],
                bottom=migrated)
    b3 = ax.bar(x, paused,   bar_width, label="Pause",    color=action_colors["pause"],
                bottom=[m + d for m, d in zip(migrated, dvfs)])
    b4 = ax.bar(x, nothing,  bar_width, label="Nothing",  color=action_colors["nothing"],
                bottom=[m + d + p for m, d, p in zip(migrated, dvfs, paused)])

    ax.set_xticks(x)
    ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS])
    ax.set_ylabel("Mean Jobs per Hour")
    ax.set_title("MIP Action Mix by Region\n(averaged across all stress events and ensembles 1–4)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / "action_breakdown.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df  = load_summaries()
    avg = ensemble_mean(df)

    plot_curtailment_comparison(avg, OUT_DIR)
    plot_qos_comparison(avg, OUT_DIR)
    plot_oracle_gap(avg, OUT_DIR)
    plot_ensemble_sensitivity(df, OUT_DIR)
    plot_action_breakdown(df, OUT_DIR)

    logger.info(f"All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
