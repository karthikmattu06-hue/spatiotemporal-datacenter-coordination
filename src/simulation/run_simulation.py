"""
Simulation driver: run MIP coordinator + all baselines over real stress events.

For each stress event in the specified region:
  1. Look up the actual load at that hour across all three regions
  2. Compute curtailment target (CURTAILMENT_FRACTION × stressed-region load)
  3. Compute headroom at non-stressed regions (P90 threshold − current load)
  4. Generate a synthetic workload ensemble (scaled to stressed-region load)
  5. Run MIP coordinator + all 5 baselines
  6. Record results

Outputs:
  data/processed/simulation/results_{region}_e{ensemble}.parquet
  data/processed/simulation/summary_{region}_e{ensemble}.json

Usage:
    python -m src.simulation.run_simulation --region caiso --ensemble 1
    python -m src.simulation.run_simulation --region pjm --ensemble 2 --max-events 50
    python -m src.simulation.run_simulation --all-regions --ensemble 1
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import ALL_BASELINES
from .mip_coordinator import CURTAILMENT_FRACTION, MIPCoordinator, compute_headroom
from .workload import REGIONS, generate_ensemble

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from src.data_collection.config import PROCESSED_DIR, LOG_LEVEL
except ImportError:
    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    LOG_LEVEL = "INFO"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SIM_OUT_DIR = PROCESSED_DIR / "simulation"
STRATEGIES = ["mip"] + list(ALL_BASELINES.keys())


# ── Data loading ──────────────────────────────────────────────────────────────

def load_stress_data(regions: list[str]) -> dict[str, pd.DataFrame]:
    """Load stress-annotated parquet for each region."""
    data = {}
    for region in regions:
        path = PROCESSED_DIR / region / f"{region}_stress_annotated.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\nRun stress_analysis.py --region {region} first."
            )
        df = pd.read_parquet(path)[["load_mw", "stress_any", "stress_event_id"]]
        data[region] = df
        logger.info(f"Loaded {region}: {len(df)} hours")
    return data


def compute_p90_thresholds(data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute global P90 load threshold per region (used for headroom)."""
    return {
        region: float(df["load_mw"].quantile(0.90))
        for region, df in data.items()
    }


# ── Simulation ────────────────────────────────────────────────────────────────

def run_one_event(
    event_id: int,
    event_hours: pd.DataFrame,
    stressed_region: str,
    all_data: dict[str, pd.DataFrame],
    p90: dict[str, float],
    ensemble_id: int,
    n_nodes: int,
    mip_coordinator: MIPCoordinator,
) -> list[dict]:
    """Run all strategies on one stress event. Returns list of result rows."""
    rows = []

    # Oracle knows the event peak upfront (perfect foresight).
    # peak_load_mw is used to scale Oracle's curtailment target at every hour.
    peak_load_mw = float(event_hours["load_mw"].max())

    for hour_ts, hour_row in event_hours.iterrows():
        stressed_load = hour_row["load_mw"]

        # Headroom at other regions at this same timestamp
        region_loads = {stressed_region: stressed_load}
        for r in REGIONS:
            if r != stressed_region and r in all_data:
                try:
                    region_loads[r] = float(all_data[r].loc[hour_ts, "load_mw"])
                except KeyError:
                    region_loads[r] = p90[r] * 0.85  # fallback: assume 85% load

        headroom = compute_headroom(
            region_loads={r: v for r, v in region_loads.items() if r != stressed_region},
            region_p90={r: p90[r] for r in REGIONS if r != stressed_region},
        )

        # Scale n_nodes so fleet total power ≈ stressed-region actual load
        # (each node ~6 MW average, so n_nodes ≈ load_mw / 6)
        n_jobs = max(20, min(200, int(stressed_load / 6)))
        ensemble = generate_ensemble(
            ensemble_id=ensemble_id,
            n_nodes=n_jobs,
            region=stressed_region,
            seed=int(event_id * 1000 + hour_ts.hour),
        )

        # Curtailment target is fleet-relative: DR signal asks the DC operator
        # to curtail 10% of their own fleet power, not 10% of the entire grid.
        curtailment_target = ensemble.total_power_mw * CURTAILMENT_FRACTION

        base_row = {
            "event_id": event_id,
            "timestamp": hour_ts,
            "stressed_region": stressed_region,
            "stressed_load_mw": stressed_load,
            "curtailment_target_mw": curtailment_target,
            "ensemble_id": ensemble_id,
            "n_jobs": n_jobs,
            **{f"headroom_{r}_mw": headroom.get(r, 0) for r in REGIONS if r != stressed_region},
        }

        # ── MIP ──────────────────────────────────────────────────────────────
        try:
            result = mip_coordinator.solve(
                ensemble=ensemble,
                stressed_region=stressed_region,
                curtailment_target_mw=curtailment_target,
                headroom=headroom,
            )
            rows.append({
                **base_row,
                "strategy": "mip",
                "curtailment_achieved_mw": result.total_curtailment_mw,
                "curtailment_pct": 100 * result.total_curtailment_mw / stressed_load,
                "total_qos_cost": result.total_qos_cost,
                "n_migrated": result.n_migrated,
                "n_paused": result.n_paused,
                "n_dvfs": result.n_dvfs,
                "n_nothing": result.n_nothing,
                "feasible": result.feasible,
                "solve_time_s": result.solve_time_s,
            })
        except Exception as e:
            logger.error(f"MIP failed at event {event_id} {hour_ts}: {e}")

        # ── Baselines ────────────────────────────────────────────────────────
        # Oracle receives the event-peak target at every hour (perfect foresight).
        # All other baselines receive the current-hour target (online).
        oracle_target = ensemble.total_power_mw * CURTAILMENT_FRACTION * (
            peak_load_mw / stressed_load
        )

        for name, baseline in ALL_BASELINES.items():
            try:
                result = baseline.solve(
                    ensemble=ensemble,
                    stressed_region=stressed_region,
                    curtailment_target_mw=oracle_target if name == "oracle_optimal" else curtailment_target,
                    headroom=headroom,
                )
                rows.append({
                    **base_row,
                    "strategy": name,
                    "curtailment_achieved_mw": result.total_curtailment_mw,
                    "curtailment_pct": 100 * result.total_curtailment_mw / stressed_load,
                    "total_qos_cost": result.total_qos_cost,
                    "n_migrated": result.n_migrated,
                    "n_paused": result.n_paused,
                    "n_dvfs": result.n_dvfs,
                    "n_nothing": result.n_nothing,
                    "feasible": result.feasible,
                    "solve_time_s": result.solve_time_s,
                })
            except Exception as e:
                logger.error(f"Baseline {name} failed at event {event_id} {hour_ts}: {e}")

    return rows


def run_simulation(
    stressed_region: str,
    ensemble_id: int,
    n_nodes: int = 100,
    max_events: int | None = None,
) -> pd.DataFrame:
    """Run full simulation for one region and ensemble. Returns results DataFrame."""
    SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for all three regions (headroom needs all of them)
    all_data = load_stress_data(REGIONS)
    p90 = compute_p90_thresholds(all_data)
    logger.info(f"P90 thresholds: { {r: f'{v:.0f} MW' for r, v in p90.items()} }")

    stressed_df = all_data[stressed_region]
    event_ids = sorted(stressed_df.loc[stressed_df["stress_event_id"] > 0, "stress_event_id"].unique())

    if max_events:
        event_ids = event_ids[:max_events]

    logger.info(
        f"Region: {stressed_region.upper()}, Ensemble: {ensemble_id}, "
        f"Events: {len(event_ids)}"
    )

    mip = MIPCoordinator(log_to_console=False)
    all_rows = []

    for i, event_id in enumerate(event_ids):
        event_hours = stressed_df[stressed_df["stress_event_id"] == event_id]
        rows = run_one_event(
            event_id=event_id,
            event_hours=event_hours,
            stressed_region=stressed_region,
            all_data=all_data,
            p90=p90,
            ensemble_id=ensemble_id,
            n_nodes=n_nodes,
            mip_coordinator=mip,
        )
        all_rows.extend(rows)

        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(event_ids)} events")

    results = pd.DataFrame(all_rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = SIM_OUT_DIR / f"results_{stressed_region}_e{ensemble_id}.parquet"
    results.to_parquet(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = build_summary(results, stressed_region, ensemble_id)
    summary_path = SIM_OUT_DIR / f"summary_{stressed_region}_e{ensemble_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")

    print_summary(summary)
    return results


def build_summary(results: pd.DataFrame, region: str, ensemble_id: int) -> dict:
    """Compute per-strategy aggregate metrics."""
    strategies = results["strategy"].unique()
    strategy_stats = {}

    for strategy in strategies:
        s = results[results["strategy"] == strategy]
        strategy_stats[strategy] = {
            "curtailment_pct_mean": round(float(s["curtailment_pct"].mean()), 1),
            "curtailment_pct_p10": round(float(s["curtailment_pct"].quantile(0.10)), 1),
            "total_qos_cost_mean": round(float(s["total_qos_cost"].mean()), 4),
            "n_migrated_mean": round(float(s["n_migrated"].mean()), 1),
            "n_paused_mean": round(float(s["n_paused"].mean()), 1),
            "feasible_pct": round(100 * float(s["feasible"].mean()), 1),
        }

    return {
        "region": region,
        "ensemble_id": ensemble_id,
        "n_events": int(results["event_id"].nunique()),
        "n_hours": len(results[results["strategy"] == strategies[0]]),
        "strategies": strategy_stats,
    }


def print_summary(summary: dict):
    """Print comparison table to stdout."""
    logger.info("\n" + "=" * 70)
    logger.info(f"SIMULATION SUMMARY — {summary['region'].upper()} Ensemble {summary['ensemble_id']}")
    logger.info(f"Events: {summary['n_events']}, Hours: {summary['n_hours']}")
    logger.info("=" * 70)
    logger.info(f"{'Strategy':<22} {'Curtail%':>10} {'QoS cost':>10} {'Migrated':>10} {'Paused':>8}")
    logger.info("-" * 70)
    for strat, stats in summary["strategies"].items():
        logger.info(
            f"{strat:<22} {stats['curtailment_pct_mean']:>9.1f}% "
            f"{stats['total_qos_cost_mean']:>10.4f} "
            f"{stats['n_migrated_mean']:>10.1f} "
            f"{stats['n_paused_mean']:>8.1f}"
        )
    logger.info("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run spatiotemporal coordination simulation")
    parser.add_argument(
        "--region", type=str, default="caiso",
        choices=REGIONS + ["all"],
        help="Stressed region (default: caiso). Use 'all' for all three regions."
    )
    parser.add_argument(
        "--ensemble", type=int, default=1, choices=[1, 2, 3, 4],
        help="Workload ensemble ID from Emerald Table 1 (default: 1)"
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Limit number of stress events (useful for quick tests)"
    )
    args = parser.parse_args()

    regions_to_run = REGIONS if args.region == "all" else [args.region]

    for region in regions_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running simulation: {region.upper()}, Ensemble {args.ensemble}")
        run_simulation(
            stressed_region=region,
            ensemble_id=args.ensemble,
            max_events=args.max_events,
        )


if __name__ == "__main__":
    main()
