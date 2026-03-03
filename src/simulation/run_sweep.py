"""
Curtailment-fraction sweep: run simulation at 5%, 10%, 15%, 20%, 25%, 30%
for all three regions, Ensemble 1 (training-dominant, most representative).

Basis for range:
  - CAISO BIP / CBP: 10–20% (reduce to Firm Service Level)
  - PJM ELRP: 10–15% economic, 20–25% emergency
  - Emerald benchmark: 25% cluster power reduction (high-end emergency)
  - 5%: light price-responsive signal; 30%: near-maximum feasibility limit

Outputs per run:
  data/processed/simulation/results_{region}_e1_f{pct}.parquet
  data/processed/simulation/summary_{region}_e1_f{pct}.json

Aggregated sweep table:
  data/processed/simulation/sweep_summary.csv

Usage:
    python -m src.simulation.run_sweep
    python -m src.simulation.run_sweep --max-events 50    # quick test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .run_simulation import SIM_OUT_DIR, run_simulation
from .workload import REGIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Sweep configuration ───────────────────────────────────────────────────────

SWEEP_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SWEEP_ENSEMBLE = 1     # training-dominant; most representative for DR comparison
SWEEP_REGIONS = REGIONS  # all three


def run_sweep(max_events: int | None = None) -> pd.DataFrame:
    """Run full curtailment-fraction sweep. Returns aggregated summary DataFrame."""
    summary_rows = []

    total = len(SWEEP_REGIONS) * len(SWEEP_FRACTIONS)
    done = 0

    for region in SWEEP_REGIONS:
        for frac in SWEEP_FRACTIONS:
            frac_pct = int(round(frac * 100))
            logger.info(
                f"\n{'='*60}\n"
                f"  [{done+1}/{total}] {region.upper()}  Ensemble {SWEEP_ENSEMBLE}  "
                f"Fraction {frac_pct}%\n"
                f"{'='*60}"
            )

            # Check if already computed (allows resuming interrupted sweep)
            out_path = SIM_OUT_DIR / f"results_{region}_e{SWEEP_ENSEMBLE}_f{frac_pct}.parquet"
            if out_path.exists():
                logger.info(f"  Already computed — loading cached results.")
                results = pd.read_parquet(out_path)
            else:
                results = run_simulation(
                    stressed_region=region,
                    ensemble_id=SWEEP_ENSEMBLE,
                    max_events=max_events,
                    curtailment_fraction=frac,
                )

            # Aggregate key metrics per strategy
            for strategy in results["strategy"].unique():
                s = results[results["strategy"] == strategy]
                summary_rows.append({
                    "region": region,
                    "ensemble_id": SWEEP_ENSEMBLE,
                    "curtailment_fraction": frac,
                    "curtailment_pct_requested": frac_pct,
                    "strategy": strategy,
                    "curtailment_pct_achieved": round(float(s["curtailment_pct"].mean()), 2),
                    "qos_cost_mean": round(float(s["total_qos_cost"].mean()), 4),
                    "n_migrated_mean": round(float(s["n_migrated"].mean()), 1),
                    "n_paused_mean": round(float(s["n_paused"].mean()), 1),
                    "feasible_pct": round(100 * float(s["feasible"].mean()), 1),
                    "n_events": int(results["event_id"].nunique()),
                })

            done += 1

    sweep_df = pd.DataFrame(summary_rows)

    out_csv = SIM_OUT_DIR / "sweep_summary.csv"
    sweep_df.to_csv(out_csv, index=False)
    logger.info(f"\nSweep complete. Summary saved: {out_csv}")

    # Print pivot: MIP vs TemporalOnly at each fraction for CAISO
    _print_pivot(sweep_df, region="caiso")

    return sweep_df


def _print_pivot(df: pd.DataFrame, region: str):
    sub = df[
        (df["region"] == region)
        & (df["strategy"].isin(["mip", "temporal_only", "oracle_optimal"]))
    ].sort_values(["curtailment_fraction", "strategy"])

    logger.info(f"\n{'='*70}")
    logger.info(f"Curtailment sweep — {region.upper()}  (Ensemble {SWEEP_ENSEMBLE})")
    logger.info(f"{'Fraction':>10} {'Strategy':<20} {'Achieved%':>10} {'QoS cost':>10} {'Migrated':>9}")
    logger.info("-" * 70)
    for _, row in sub.iterrows():
        logger.info(
            f"{int(row['curtailment_pct_requested']):>9}%  "
            f"{row['strategy']:<20} "
            f"{row['curtailment_pct_achieved']:>9.1f}%  "
            f"{row['qos_cost_mean']:>10.4f}  "
            f"{row['n_migrated_mean']:>8.1f}"
        )
    logger.info("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curtailment fraction sweep")
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Limit stress events per run (for quick testing)"
    )
    args = parser.parse_args()
    run_sweep(max_events=args.max_events)
