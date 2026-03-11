"""
MIP-based spatiotemporal datacenter coordinator.

Formulates and solves the optimal job dispatch decision as a MILP using
scipy.optimize.milp (HiGHS solver — open-source, no license required).

Decision variables: x[j, k] ∈ {0,1} — binary assignment of job j to action k.

Actions per job (mutually exclusive):
  nothing        q=0.00   δ=0
  dvfs_0.9       q=0.10   δ=P_j × 0.10
  dvfs_0.8       q=0.20   δ=P_j × 0.20
  dvfs_0.7       q=0.30   δ=P_j × 0.30
  dvfs_0.6       q=0.40   δ=P_j × 0.40
  pause          q=1.00   δ=P_j
  migrate_{r'}   q≈0.01   δ=P_j  (load removed from r0, added to r')

Objective: min Σ_j w_j × Σ_k q_k × x[j,k]

Constraints:
  1. Mutual exclusivity: Σ_k x[j,k] = 1  ∀j
  2. Curtailment:        Σ_j Σ_k δ_k(P_j) × x[j,k] ≥ Δ_r
  3. SLA:                Σ_k q_k × x[j,k] ≤ φ_{f_j}  ∀j
  4. Latency:            x[j, migrate_{r'}] = 0  if λ(r0→r') > budget_j
  5. Capacity:           Σ_j P_j × x[j, migrate_{r'}] ≤ headroom_{r'}  ∀r'

No external solver license required. Requires scipy >= 1.9.
  pip install scipy   # already in requirements.txt

Usage:
  python -m src.simulation.mip_coordinator --test
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from .workload import LATENCY_MS, Job, WorkloadEnsemble

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# DVFS cap levels → (qos_degradation, power_reduction_fraction)
DVFS_LEVELS: dict[str, tuple[float, float]] = {
    "dvfs_0.9": (0.10, 0.10),
    "dvfs_0.8": (0.20, 0.20),
    "dvfs_0.7": (0.30, 0.30),
    "dvfs_0.6": (0.40, 0.40),
}

CURTAILMENT_FRACTION = 0.10  # DR signal targets 10% load reduction

# Migration QoS overhead (dimensionless, on [0,1] scale).
# Training:  ~2% throughput loss ≈ 30s downtime / 1800s checkpoint interval
#            Source: TrainMover, arXiv:2412.12636 (measured on GPT-39.1B)
# Inference: ~1% disruption ≈ 5s GPU-snapshot re-warm / 500s session horizon
#            Source: ServerlessLLM, OSDI 2024
MIGRATION_QOS: dict[str, float] = {
    "training":  0.02,
    "inference": 0.01,
}

# ── Result types ──────────────────────────────────────────────────────────────

ActionType = Literal[
    "nothing", "dvfs_0.9", "dvfs_0.8", "dvfs_0.7", "dvfs_0.6", "pause", "migrate"
]

@dataclass
class JobDecision:
    job_id: str
    action: ActionType
    target_region: str | None   # Set for migrate, None otherwise
    qos_degradation: float      # Realized q_j ∈ [0, 1]
    power_reduction_mw: float   # δ_j: load removed from stressed region

@dataclass
class DispatchResult:
    decisions: dict[str, JobDecision]   # job_id → decision
    total_curtailment_mw: float
    total_qos_cost: float               # Σ w_j * q_j
    n_migrated: int
    n_paused: int
    n_dvfs: int
    n_nothing: int
    feasible: bool
    solve_time_s: float

    def summary(self) -> dict:
        return {
            "curtailment_mw": round(self.total_curtailment_mw, 1),
            "total_qos_cost": round(self.total_qos_cost, 4),
            "n_migrated": self.n_migrated,
            "n_paused": self.n_paused,
            "n_dvfs": self.n_dvfs,
            "n_nothing": self.n_nothing,
            "feasible": self.feasible,
            "solve_time_s": round(self.solve_time_s, 4),
        }


# ── MIP Coordinator ───────────────────────────────────────────────────────────

class MIPCoordinator:
    """
    Optimal dispatcher via MILP (HiGHS solver via scipy.optimize.milp).
    No license required — fully open-source.
    """

    def __init__(self, time_limit_s: float = 30.0, log_to_console: bool = False,
                 allow_migration: bool = True):
        self.time_limit_s   = time_limit_s
        self.log_to_console  = log_to_console
        self.allow_migration = allow_migration

    def solve(
        self,
        ensemble: WorkloadEnsemble,
        stressed_region: str,
        curtailment_target_mw: float,
        headroom: dict[str, float],
        objective: Literal["qos", "carbon"] = "qos",
        carbon_intensity: dict[str, float] | None = None,
    ) -> DispatchResult:
        """Solve the dispatch MILP via HiGHS (scipy.optimize.milp).

        Parameters
        ----------
        ensemble              : Current workload at the stressed datacenter
        stressed_region       : Region receiving the DR signal
        curtailment_target_mw : Target load reduction in MW
        headroom              : Available headroom (MW) at each non-stressed region
        objective             : "qos" (default) or "carbon"
        carbon_intensity      : Required when objective="carbon"; {region: gCO2/kWh}
        """
        t0 = time.perf_counter()

        jobs          = ensemble.jobs
        other_regions = [r for r in headroom if r != stressed_region]

        # ── Build action catalogue ────────────────────────────────────────────
        action_catalogue: list[list[tuple[str, str | None, float, float]]] = []

        for job in jobs:
            φ    = job.flex_tier.max_qos_degradation
            acts: list[tuple[str, str | None, float, float]] = []

            # Always available: do nothing
            acts.append(("nothing", None, 0.0, 0.0))

            # DVFS actions (not available for Flex 0)
            if job.flex_tier.value > 0:
                for name, (q, frac) in DVFS_LEVELS.items():
                    if q <= φ:
                        acts.append((name, None, q, job.power_mw * frac))

            # Pause (DR emergency override — available for all non-Flex-0 jobs,
            # bypasses SLA φ check; SLA constraint exemption handled below)
            if job.flex_tier.value > 0:
                acts.append(("pause", None, 1.0, job.power_mw))

            # Migration — available if latency + capacity allow (and not disabled)
            if self.allow_migration:
                for r_prime in other_regions:
                    lat = LATENCY_MS.get((stressed_region, r_prime), 999.0)
                    if lat <= job.latency_budget_ms:
                        q_mig = MIGRATION_QOS.get(job.job_type, 0.02)
                        acts.append(("migrate", r_prime, q_mig, job.power_mw))

            action_catalogue.append(acts)

        # ── Flatten variables: x[offsets[j] + k] = z_{j,k} ──────────────────
        offsets    = []
        total_vars = 0
        for acts in action_catalogue:
            offsets.append(total_vars)
            total_vars += len(acts)

        # ── Objective vector ──────────────────────────────────────────────────
        c_obj = np.zeros(total_vars)
        for j, job in enumerate(jobs):
            for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                idx = offsets[j] + k
                if objective == "qos":
                    c_obj[idx] = job.weight * q
                elif objective == "carbon":
                    if carbon_intensity is None:
                        raise ValueError("carbon_intensity required for carbon objective")
                    loc = (target if (aname == "migrate" and target) else stressed_region)
                    c_obj[idx] = carbon_intensity.get(loc, 0.0) * job.power_mw
                else:
                    raise ValueError(f"Unknown objective: {objective}")

        # ── Constraint matrix (sparse COO → CSR) ─────────────────────────────
        # Row layout:
        #   rows 0..J-1        : mutual exclusivity  (= 1)
        #   row  J             : curtailment          (≥ D)
        #   rows J+1..2J       : SLA per job          (≤ φ_j)
        #   rows 2J+1..2J+|R'| : capacity per region  (≤ h_r')

        n_jobs = len(jobs)
        n_rows = n_jobs + 1 + n_jobs + len(other_regions)

        rows_idx: list[int]   = []
        cols_idx: list[int]   = []
        data_val: list[float] = []
        lb_arr = np.empty(n_rows)
        ub_arr = np.empty(n_rows)

        row = 0

        # 1. Mutual exclusivity
        for j in range(n_jobs):
            for k in range(len(action_catalogue[j])):
                rows_idx.append(row); cols_idx.append(offsets[j] + k); data_val.append(1.0)
            lb_arr[row] = ub_arr[row] = 1.0
            row += 1

        # 2. Curtailment  (≥ target → lb=target, ub=+∞)
        for j, job in enumerate(jobs):
            for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                if delta > 0:
                    rows_idx.append(row); cols_idx.append(offsets[j] + k); data_val.append(delta)
        lb_arr[row] = curtailment_target_mw
        ub_arr[row] = np.inf
        row += 1

        # 3. SLA  (≤ φ_j → lb=-∞, ub=φ_j)
        #    Pause is exempt: DR emergency override intentionally violates SLA bound.
        for j, job in enumerate(jobs):
            φ = job.flex_tier.max_qos_degradation
            for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                if q > 0 and aname != "pause":
                    rows_idx.append(row); cols_idx.append(offsets[j] + k); data_val.append(q)
            lb_arr[row] = -np.inf
            ub_arr[row] = φ
            row += 1

        # 4. Destination capacity  (≤ headroom_r' → lb=-∞, ub=headroom_r')
        for r_prime in other_regions:
            cap = headroom.get(r_prime, 0.0)
            for j, job in enumerate(jobs):
                for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                    if aname == "migrate" and target == r_prime:
                        rows_idx.append(row)
                        cols_idx.append(offsets[j] + k)
                        data_val.append(job.power_mw)
            lb_arr[row] = -np.inf
            ub_arr[row] = cap
            row += 1

        A_sparse = csr_matrix(
            (data_val, (rows_idx, cols_idx)),
            shape=(n_rows, total_vars),
        )

        constraints = LinearConstraint(A_sparse, lb_arr, ub_arr)
        integrality = np.ones(total_vars)          # all binary
        bounds      = Bounds(lb=0.0, ub=1.0)
        options     = {
            "disp":       self.log_to_console,
            "time_limit": self.time_limit_s,
        }

        # ── Solve ─────────────────────────────────────────────────────────────
        result     = milp(c_obj, constraints=constraints,
                          integrality=integrality, bounds=bounds, options=options)
        solve_time = time.perf_counter() - t0

        # scipy status: 0=optimal, 1=time-limit feasible, 2=infeasible, 3=unbounded
        feasible = result.status in (0, 1)

        if not feasible:
            logger.warning(
                f"MIP infeasible (status={result.status}) — curtailment target "
                f"{curtailment_target_mw:.1f} MW may exceed fleet capacity"
            )
            return DispatchResult(
                decisions={}, total_curtailment_mw=0.0, total_qos_cost=0.0,
                n_migrated=0, n_paused=0, n_dvfs=0, n_nothing=n_jobs,
                feasible=False, solve_time_s=solve_time,
            )

        # ── Extract solution ──────────────────────────────────────────────────
        x = result.x
        decisions: dict[str, JobDecision] = {}
        total_curtailment = 0.0
        total_qos_cost    = 0.0
        n_mig = n_pau = n_dvf = n_not = 0

        for j, job in enumerate(jobs):
            chosen_k = int(np.argmax(x[offsets[j]: offsets[j] + len(action_catalogue[j])]))
            aname, target, q, delta = action_catalogue[j][chosen_k]

            decisions[job.job_id] = JobDecision(
                job_id=job.job_id,
                action=aname,
                target_region=target,
                qos_degradation=q,
                power_reduction_mw=delta,
            )
            total_curtailment += delta
            total_qos_cost    += job.weight * q

            if aname == "migrate":   n_mig += 1
            elif aname == "pause":   n_pau += 1
            elif "dvfs" in aname:    n_dvf += 1
            else:                    n_not += 1

        return DispatchResult(
            decisions=decisions,
            total_curtailment_mw=total_curtailment,
            total_qos_cost=total_qos_cost,
            n_migrated=n_mig,
            n_paused=n_pau,
            n_dvfs=n_dvf,
            n_nothing=n_not,
            feasible=True,
            solve_time_s=solve_time,
        )


# ── Headroom utility ──────────────────────────────────────────────────────────

def compute_headroom(
    region_loads: dict[str, float],
    region_p90:   dict[str, float],
) -> dict[str, float]:
    """Compute available headroom (MW) at each region.

    headroom = P90_threshold - current_load (floored at 0)
    """
    return {
        r: max(0.0, region_p90[r] - region_loads[r])
        for r in region_loads
    }


# ── CLI smoke test ────────────────────────────────────────────────────────────

def _test():
    """Quick smoke test on a synthetic 50-job ensemble."""
    from .workload import generate_ensemble
    from collections import Counter

    print("=== MIP Coordinator smoke test (HiGHS / scipy) ===")
    ensemble = generate_ensemble(ensemble_id=1, n_nodes=50, region="caiso")
    print(f"Ensemble: {ensemble.summary()}")

    curtailment_target = ensemble.total_power_mw * CURTAILMENT_FRACTION
    headroom = {"pjm": 2000.0, "bpa": 500.0}

    print(f"Curtailment target: {curtailment_target:.1f} MW")
    print(f"Headroom: {headroom}")

    coord  = MIPCoordinator(log_to_console=False)
    result = coord.solve(
        ensemble=ensemble,
        stressed_region="caiso",
        curtailment_target_mw=curtailment_target,
        headroom=headroom,
    )

    print(f"\nResult: {result.summary()}")
    print(f"Curtailment achieved: {result.total_curtailment_mw:.1f} / {curtailment_target:.1f} MW")
    print(f"QoS cost: {result.total_qos_cost:.4f}")

    actions = Counter(d.action for d in result.decisions.values())
    print(f"Actions: {dict(actions)}")

    migrate_targets = Counter(
        d.target_region for d in result.decisions.values() if d.action == "migrate"
    )
    if migrate_targets:
        print(f"Migration targets: {dict(migrate_targets)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        _test()
