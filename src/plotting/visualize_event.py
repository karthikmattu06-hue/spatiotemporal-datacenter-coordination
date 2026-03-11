"""
Emerald-style Fig. 2 reproduction for our spatiotemporal coordinator.

Left panel:  CAISO grid load + datacenter cluster power over time.
             Shows the DR response during a real stress event
             (California heat wave, Sep 6-7 2022).

Right panel: Per-flex-tier QoS performance maintained under MIP coordination.
             Bars show mean (1 - qos_degradation) per tier at the event peak.
             Reference lines at each tier's SLA threshold.

Outputs:
    figures/simulation/event_timeline.png   (standalone timeline)
    figures/simulation/event_combined.png   (2-panel, Emerald Fig. 2 style)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from src.data_collection.config import PROCESSED_DIR
except ImportError:
    PROCESSED_DIR = ROOT / "data" / "processed"

SIM_DIR  = PROCESSED_DIR / "simulation"
FIG_DIR  = ROOT / "figures" / "simulation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "grid_load":    "#7c3aed",   # purple — CAISO grid
    "baseline":     "#9ca3af",   # gray
    "mip":          "#2563eb",   # blue
    "temporal":     "#f59e0b",   # orange
    "event_fill":   "#fef9c3",   # pale yellow
    "buffer_fill":  "#e5e7eb",   # light gray
}
FLEX_COLORS = {0: "#6366f1", 1: "#22c55e", 2: "#f97316", 3: "#2563eb"}
SLA_THRESHOLDS = {0: 100.0, 1: 95.0, 2: 85.0, 3: 70.0}   # min performance %

# ── Chosen event: CAISO Sep 6-7 2022 (California heat wave, event 560) ───────
TARGET_EVENT   = 560
TARGET_REGION  = "caiso"
ENSEMBLE_ID    = 1
CURTAILMENT_F  = 0.10   # 10% DR target (standard)
CONTEXT_HOURS  = 4      # hours of context before/after event


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Load simulation results + CAISO stress data."""
    sim = pd.read_parquet(SIM_DIR / f"results_{TARGET_REGION}_e{ENSEMBLE_ID}_f10.parquet")
    sim["timestamp"] = pd.to_datetime(sim["timestamp"], utc=True)

    stress = pd.read_parquet(
        PROCESSED_DIR / TARGET_REGION / f"{TARGET_REGION}_stress_annotated.parquet"
    )
    return sim, stress


# ── Per-tier QoS from targeted MIP run ────────────────────────────────────────

def compute_tier_performance(sim: pd.DataFrame) -> tuple[dict, dict, object, object]:
    """
    Re-run MIP and TemporalOnly on the event's peak-load hour.
    Returns (mip_tier_perf, temp_tier_perf, mip_result, ensemble).

    Reconstructs the exact ensemble using the same seed used during
    the original simulation run (seed = event_id * 1000 + hour).
    """
    from collections import defaultdict

    from src.simulation.baselines import TemporalOnly
    from src.simulation.mip_coordinator import MIPCoordinator
    from src.simulation.workload import generate_ensemble

    mip_event = sim[(sim["strategy"] == "mip") & (sim["event_id"] == TARGET_EVENT)]
    peak_row  = mip_event.loc[mip_event["stressed_load_mw"].idxmax()]

    hour_ts = pd.Timestamp(peak_row["timestamp"])
    n_jobs  = int(peak_row["n_jobs"])
    seed    = int(TARGET_EVENT * 1000 + hour_ts.hour)

    ensemble = generate_ensemble(
        ensemble_id=ENSEMBLE_ID,
        n_nodes=n_jobs,
        region=TARGET_REGION,
        seed=seed,
    )

    curtailment_target = ensemble.total_power_mw * CURTAILMENT_F
    headroom = {
        "pjm": float(peak_row["headroom_pjm_mw"]),
        "bpa": float(peak_row["headroom_bpa_mw"]),
    }

    # ── MIP ───────────────────────────────────────────────────────────────────
    coord      = MIPCoordinator(log_to_console=False)
    mip_result = coord.solve(
        ensemble=ensemble,
        stressed_region=TARGET_REGION,
        curtailment_target_mw=curtailment_target,
        headroom=headroom,
    )

    # ── TemporalOnly ──────────────────────────────────────────────────────────
    temp        = TemporalOnly()
    temp_result = temp.solve(
        ensemble=ensemble,
        stressed_region=TARGET_REGION,
        curtailment_target_mw=curtailment_target,
        headroom=headroom,
    )

    def _tier_perf(result, ens) -> dict[int, float]:
        by_tier: dict[int, list[float]] = defaultdict(list)
        for job in ens.jobs:
            d = result.decisions.get(job.job_id)
            q = d.qos_degradation if d else 0.0
            by_tier[job.flex_tier.value].append((1.0 - q) * 100.0)
        return {t: float(np.mean(v)) for t, v in by_tier.items()}

    return _tier_perf(mip_result, ensemble), _tier_perf(temp_result, ensemble), mip_result, ensemble


# ── Left panel ────────────────────────────────────────────────────────────────

def build_timeline(sim: pd.DataFrame, stress: pd.DataFrame):
    """Return (window_load, event_hours_mip, event_hours_temp, t_start, t_end)."""
    mip  = sim[(sim["strategy"] == "mip") & (sim["event_id"] == TARGET_EVENT)].sort_values("timestamp")
    temp = sim[(sim["strategy"] == "temporal_only") & (sim["event_id"] == TARGET_EVENT)].sort_values("timestamp")

    t_start = mip["timestamp"].iloc[0]
    t_end   = mip["timestamp"].iloc[-1]

    w_start = t_start - pd.Timedelta(hours=CONTEXT_HOURS)
    w_end   = t_end   + pd.Timedelta(hours=CONTEXT_HOURS)

    window_load = stress.loc[w_start:w_end, "load_mw"].copy()

    # Baseline cluster power (mean of event-hour fleet_power_mw)
    baseline_mw = float(mip["fleet_power_mw"].mean())

    # Build cluster power series over the full window
    cluster_mip  = pd.Series(baseline_mw, index=window_load.index, dtype=float)
    cluster_temp = pd.Series(baseline_mw, index=window_load.index, dtype=float)

    for _, row in mip.iterrows():
        ts = row["timestamp"]
        if ts in cluster_mip.index:
            cluster_mip[ts] = row["fleet_power_mw"] - row["curtailment_achieved_mw"]

    for _, row in temp.iterrows():
        ts = row["timestamp"]
        if ts in cluster_temp.index:
            cluster_temp[ts] = row["fleet_power_mw"] - row["curtailment_achieved_mw"]

    return window_load, cluster_mip, cluster_temp, baseline_mw, t_start, t_end


def plot_timeline_panel(ax, window_load, cluster_mip, cluster_temp,
                        baseline_mw, t_start, t_end):
    """Draw timeline on ax. Returns twin axis."""
    # Convert UTC to Pacific (UTC-7 in September)
    local_idx = window_load.index.tz_convert("America/Los_Angeles")

    # ── Background shading ───────────────────────────────────────────────────
    t_start_l = t_start.tz_convert("America/Los_Angeles")
    t_end_l   = t_end.tz_convert("America/Los_Angeles")
    w_start_l = local_idx[0]
    w_end_l   = local_idx[-1]

    # Buffer regions (gray)
    ax.axvspan(w_start_l, t_start_l, color=COLORS["buffer_fill"], alpha=0.6, zorder=0)
    ax.axvspan(t_end_l,   w_end_l,   color=COLORS["buffer_fill"], alpha=0.6, zorder=0)
    # Event region (yellow)
    ax.axvspan(t_start_l, t_end_l,   color=COLORS["event_fill"],  alpha=0.9, zorder=0)

    # ── Grid load (left axis) ────────────────────────────────────────────────
    ax.plot(local_idx, window_load.values / 1000,
            color=COLORS["grid_load"], lw=2.2, marker="o", ms=4,
            label="CAISO Grid Load", zorder=3)
    ax.set_ylabel("Grid Load (GW)", color=COLORS["grid_load"], fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=COLORS["grid_load"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # ── Cluster power (right axis) ───────────────────────────────────────────
    ax2 = ax.twinx()
    local_idx2 = cluster_mip.index.tz_convert("America/Los_Angeles")

    # Baseline
    ax2.axhline(baseline_mw, color=COLORS["baseline"], lw=1.2, ls="--",
                alpha=0.8, label=f"Baseline ({baseline_mw:.0f} MW)", zorder=2)

    # Temporal-Only
    ax2.plot(local_idx2, cluster_temp.values,
             color=COLORS["temporal"], lw=1.8, ls="-.", ms=3,
             label="Temporal-Only (no migration)", zorder=3)

    # MIP Coordinator
    ax2.plot(local_idx2, cluster_mip.values,
             color=COLORS["mip"], lw=2.2, ms=4,
             label="MIP Coordinator (ours)", zorder=4)

    # Annotate curtailment gap at peak
    peak_local = window_load.idxmax().tz_convert("America/Los_Angeles")
    if peak_local in cluster_mip.index.tz_convert("America/Los_Angeles"):
        peak_pos = cluster_mip.index.tz_convert("America/Los_Angeles").get_loc(peak_local)
        mip_val  = cluster_mip.values[peak_pos] if peak_pos < len(cluster_mip) else None
        if mip_val:
            pct = 100 * (baseline_mw - mip_val) / baseline_mw
            ax2.annotate(
                f"−{pct:.0f}%\ncurtailed",
                xy=(peak_local, (baseline_mw + mip_val) / 2),
                xytext=(peak_local + pd.Timedelta(hours=1.5), (baseline_mw + mip_val) / 2),
                fontsize=8, color=COLORS["mip"],
                arrowprops=dict(arrowstyle="->", color=COLORS["mip"], lw=1),
                va="center",
            )

    ax2.set_ylabel("Datacenter Cluster Power (MW)", fontsize=10, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # ── X-axis ────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
    ax.set_xlabel("Sep 6–7, 2022  (Pacific Time)", fontsize=9)

    # ── Legend ────────────────────────────────────────────────────────────────
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=True,
              framealpha=0.9)

    ax.set_title("CAISO Grid Stress Event — Sep 6 2022 (CA Heat Wave)",
                 fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_facecolor("white")

    return ax2


# ── Right panel ───────────────────────────────────────────────────────────────

def plot_tier_panel(ax, mip_perf: dict[int, float], temp_perf: dict[int, float]):
    """
    Grouped bar chart: MIP vs Temporal-Only performance per flex tier.
    Demonstrates that MIP achieves the same curtailment at far lower QoS cost.
    """
    tiers      = [0, 1, 2, 3]
    tier_names = ["Flex 0\n(inference)", "Flex 1\n(training)", "Flex 2\n(training)", "Flex 3\n(training)"]

    x     = np.arange(len(tiers))
    width = 0.35

    mip_vals  = [mip_perf.get(t, 100.0)  for t in tiers]
    temp_vals = [temp_perf.get(t, 100.0) for t in tiers]

    bars_mip  = ax.bar(x - width/2, mip_vals,  width, label="MIP Coordinator (ours)",
                       color=COLORS["mip"],     zorder=3, edgecolor="white", lw=0.8)
    bars_temp = ax.bar(x + width/2, temp_vals, width, label="Temporal-Only (no migration)",
                       color=COLORS["temporal"], zorder=3, edgecolor="white", lw=0.8,
                       alpha=0.85)

    # Value labels
    for bar, p in zip(bars_mip, mip_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{p:.0f}%", ha="center", va="bottom", fontsize=7.5,
                color=COLORS["mip"], fontweight="bold")
    for bar, p in zip(bars_temp, temp_vals):
        if p < 99.5:   # only label when there's meaningful degradation
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{p:.0f}%", ha="center", va="bottom", fontsize=7.5,
                    color=COLORS["temporal"], fontweight="bold")

    # SLA threshold lines (one per tier group)
    sla_line_colors = {0: "#6366f1", 1: "#22c55e", 2: "#f97316", 3: "#2563eb"}
    sla_labels      = {0: "100% SLA", 1: "95% SLA", 2: "85% SLA", 3: "70% SLA"}
    for i, t in enumerate(tiers):
        ax.hlines(SLA_THRESHOLDS[t], x[i] - 0.45, x[i] + 0.45,
                  colors=sla_line_colors[t], linewidths=2.2, linestyles="--",
                  zorder=4)
        ax.text(x[i] + 0.46, SLA_THRESHOLDS[t] + 0.5, sla_labels[t],
                va="bottom", fontsize=6.5, color=sla_line_colors[t])

    ax.set_xticks(x)
    ax.set_xticklabels(tier_names, fontsize=9)
    ax.set_ylabel("Performance Maintained (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(55, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_title("Job Performance by Flex Tier\n(same 10% curtailment target)",
                 fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_facecolor("white")
    ax.legend(fontsize=9, loc="lower left", frameon=True, framealpha=0.9)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    sim, stress = load_data()

    print("Computing per-tier QoS (running MIP + TemporalOnly on peak hour)...")
    mip_perf, temp_perf, result, ensemble = compute_tier_performance(sim)
    print(f"  MIP  performance: { {t: f'{v:.1f}%' for t, v in mip_perf.items()} }")
    print(f"  Temp performance: { {t: f'{v:.1f}%' for t, v in temp_perf.items()} }")

    print("Building timeline data...")
    window_load, cluster_mip, cluster_temp, baseline_mw, t_start, t_end = \
        build_timeline(sim, stress)

    # ── Standalone timeline ───────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    fig1.patch.set_facecolor("white")
    plot_timeline_panel(ax1, window_load, cluster_mip, cluster_temp,
                        baseline_mw, t_start, t_end)
    fig1.tight_layout()
    out1 = FIG_DIR / "event_timeline.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"Saved: {out1}")

    # ── Combined 2-panel (Emerald Fig. 2 style) ───────────────────────────────
    fig2, (axL, axR) = plt.subplots(
        1, 2,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig2.patch.set_facecolor("white")

    plot_timeline_panel(axL, window_load, cluster_mip, cluster_temp,
                        baseline_mw, t_start, t_end)
    plot_tier_panel(axR, mip_perf, temp_perf)

    fig2.suptitle(
        "Spatiotemporal MIP Coordinator — CAISO Demand Response Event",
        fontsize=12, fontweight="bold", y=1.01, color="#1e293b",
    )
    fig2.tight_layout(w_pad=2)
    fig2.subplots_adjust(right=0.96)

    out2 = FIG_DIR / "event_combined.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # Print summary
    from collections import Counter
    actions = Counter(d.action for d in result.decisions.values())
    print(f"\nEvent peak hour action breakdown: {dict(actions)}")
    print(f"Curtailment: {result.total_curtailment_mw:.1f} MW  "
          f"QoS cost: {result.total_qos_cost:.4f}")


if __name__ == "__main__":
    main()
