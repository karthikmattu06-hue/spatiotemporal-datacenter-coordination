"""
Figures for the curtailment-fraction sweep.

Reads:  data/processed/simulation/sweep_summary.csv
Writes: figures/simulation/sweep_curtailment.png
        figures/simulation/sweep_qos.png
        figures/simulation/sweep_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from src.data_collection.config import PROCESSED_DIR
except ImportError:
    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

SIM_DIR = PROCESSED_DIR / "simulation"
FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "simulation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES      = ["mip", "temporal_only", "no_coordination"]
STRATEGY_LABELS = {
    "mip":             "MIP Coordinator (ours)",
    "temporal_only":   "Temporal-Only (no migration)",
    "no_coordination": "No Coordination",
}
STRATEGY_COLORS = {
    "mip":             "#2563eb",
    "temporal_only":   "#f59e0b",
    "no_coordination": "#9ca3af",
}
STRATEGY_LS = {
    "mip":             "-",
    "temporal_only":   "-.",
    "no_coordination": ":",
}

REGIONS = ["caiso", "pjm", "bpa"]
REGION_LABELS = {"caiso": "CAISO", "pjm": "PJM", "bpa": "BPA"}

# ── Load data ─────────────────────────────────────────────────────────────────

def load_sweep() -> pd.DataFrame:
    path = SIM_DIR / "sweep_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing sweep summary at {path}\n"
            "Run: python -m src.simulation.run_sweep"
        )
    df = pd.read_csv(path)
    return df[df["strategy"].isin(STRATEGIES)]


# ── Figure 1: Curtailment achieved vs. fraction (line chart, 3 regions) ───────

def fig_curtailment(df: pd.DataFrame):
    """
    3-panel figure (one per region).
    X: curtailment fraction requested (%)
    Y: mean curtailment achieved (% of fleet power)
    Shows all strategies + diagonal reference line (perfect tracking).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, region in zip(axes, REGIONS):
        sub = df[df["region"] == region]
        fracs = sorted(sub["curtailment_pct_requested"].unique())

        # Diagonal reference (perfect tracking)
        ax.plot(fracs, fracs, color="#999", lw=1.0, ls="--", label="Perfect tracking", zorder=1)

        for strat in STRATEGIES:
            s = sub[sub["strategy"] == strat].sort_values("curtailment_pct_requested")
            if s.empty:
                continue
            ax.plot(
                s["curtailment_pct_requested"],
                s["curtailment_pct_achieved"],
                color=STRATEGY_COLORS[strat],
                ls=STRATEGY_LS[strat],
                marker="o", ms=4,
                lw=1.8,
                label=STRATEGY_LABELS[strat],
                zorder=3,
            )

        ax.set_title(REGION_LABELS[region], fontsize=11, fontweight="bold")
        ax.set_xlabel("Curtailment Fraction Requested (%)", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Mean Curtailment Achieved (% of fleet)", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.suptitle(
        "Curtailment Tracking Across Demand Response Depths",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = FIG_DIR / "sweep_curtailment.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: QoS cost vs. fraction (line chart, 3 regions) ──────────────────

def fig_qos(df: pd.DataFrame):
    """
    3-panel figure.
    X: curtailment fraction requested (%)
    Y: mean QoS cost (Σ w_j × q_j)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, region in zip(axes, REGIONS):
        sub = df[df["region"] == region]

        for strat in STRATEGIES:
            s = sub[sub["strategy"] == strat].sort_values("curtailment_pct_requested")
            if s.empty:
                continue
            ax.plot(
                s["curtailment_pct_requested"],
                s["qos_cost_mean"],
                color=STRATEGY_COLORS[strat],
                ls=STRATEGY_LS[strat],
                marker="o", ms=4,
                lw=1.8,
                label=STRATEGY_LABELS[strat],
            )

        ax.set_title(REGION_LABELS[region], fontsize=11, fontweight="bold")
        ax.set_xlabel("Curtailment Fraction Requested (%)", fontsize=9)
        ax.set_ylabel("Mean QoS Cost", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.suptitle(
        "QoS Cost vs. DR Curtailment Depth",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = FIG_DIR / "sweep_qos.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2b: QoS cost (LinkedIn version — relabeled for general audience) ──

def fig_qos_linkedin(df: pd.DataFrame):
    """Same as fig_qos but with labels rewritten for a non-specialist audience."""
    linkedin_labels = {
        "mip":             "Multi-region migration",
        "temporal_only":   "Local-only response",
        "no_coordination": "No response",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, region in zip(axes, REGIONS):
        sub = df[df["region"] == region]

        for strat in STRATEGIES:
            s = sub[sub["strategy"] == strat].sort_values("curtailment_pct_requested")
            if s.empty:
                continue
            ax.plot(
                s["curtailment_pct_requested"],
                s["qos_cost_mean"],
                color=STRATEGY_COLORS[strat],
                ls=STRATEGY_LS[strat],
                marker="o", ms=4,
                lw=1.8,
                label=linkedin_labels[strat],
            )

        ax.set_title(REGION_LABELS[region], fontsize=11, fontweight="bold")
        ax.set_xlabel("Power Curtailed (% of fleet)", fontsize=9)
        ax.set_ylabel("Service Degradation Cost", fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.suptitle(
        "Service Impact vs. Power Curtailment Depth\n"
        "Same curtailment achieved — dramatically different service impact",
        fontsize=12, fontweight="bold", y=1.04,
    )
    fig.tight_layout()

    out = FIG_DIR / "sweep_qos_linkedin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Emerald comparison — curtailment + QoS at each fraction, CAISO ─

def fig_emerald_comparison(df: pd.DataFrame):
    """
    Dual-axis bar+line chart for CAISO.
    Bars: curtailment achieved (%) — MIP vs TemporalOnly
    Line overlay: QoS cost ratio (MIP / TemporalOnly)
    X: curtailment fraction
    Reference band at Emerald 25%.
    """
    region = "caiso"
    sub = df[df["region"] == region]
    fracs = sorted(sub["curtailment_pct_requested"].unique())

    mip  = sub[sub["strategy"] == "mip"].sort_values("curtailment_pct_requested")
    temp = sub[sub["strategy"] == "temporal_only"].sort_values("curtailment_pct_requested")

    if mip.empty or temp.empty:
        print("Skipping Emerald comparison figure — missing strategies.")
        return

    x = np.arange(len(fracs))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))

    bars_mip  = ax1.bar(x - width/2, mip["curtailment_pct_achieved"],  width,
                        color=STRATEGY_COLORS["mip"],          label="MIP Coordinator (ours)")
    bars_temp = ax1.bar(x + width/2, temp["curtailment_pct_achieved"], width,
                        color=STRATEGY_COLORS["temporal_only"], label="Temporal-Only (no migration)")

    ax1.set_ylabel("Mean Curtailment Achieved (% of fleet)", fontsize=10)
    ax1.set_xlabel("DR Curtailment Depth Requested", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(f)}%" for f in fracs])
    ax1.set_ylim(0, max(fracs) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    # QoS ratio on secondary axis
    ax2 = ax1.twinx()
    # Ratio = MIP QoS / TemporalOnly QoS (lower = MIP is better)
    qos_ratio = mip["qos_cost_mean"].values / np.where(
        temp["qos_cost_mean"].values > 0, temp["qos_cost_mean"].values, np.nan
    )
    ax2.plot(x, qos_ratio, color="#7c3aed", marker="D", ms=6, lw=1.8,
             ls="-", label="QoS ratio (MIP / Temporal-Only)", zorder=5)
    ax2.axhline(1.0, color="#7c3aed", lw=0.6, ls=":", alpha=0.5)
    ax2.set_ylabel("QoS Cost Ratio (MIP ÷ Temporal-Only)\n[< 1 = MIP is better]",
                   fontsize=9, color="#7c3aed")
    ax2.tick_params(axis="y", labelcolor="#7c3aed")
    ax2.set_ylim(0, 1.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, frameon=True)

    ax1.set_title(
        f"MIP vs. Temporal-Only Across DR Depths — {REGION_LABELS[region]}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    out = FIG_DIR / "sweep_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_sweep()
    print(f"Loaded sweep: {len(df)} rows, fractions={sorted(df['curtailment_pct_requested'].unique())}")
    fig_curtailment(df)
    fig_qos(df)
    fig_qos_linkedin(df)
    fig_emerald_comparison(df)
    print("All sweep figures generated.")


if __name__ == "__main__":
    main()
