"""
MIP-based spatiotemporal datacenter coordinator.

Formulates and solves the optimal job dispatch decision as a MILP using Gurobi.
Given a stressed region, a curtailment target, and available headroom at other
regions, it chooses the minimum-QoS-degradation action for each job.

Decision variables: z[j, a] ∈ {0,1} — binary assignment of job j to action a.

Actions per job (mutually exclusive):
  nothing        q=0.00   δ=0
  dvfs_0.9       q=0.10   δ=P_j × 0.10
  dvfs_0.8       q=0.20   δ=P_j × 0.20
  dvfs_0.7       q=0.30   δ=P_j × 0.30
  dvfs_0.6       q=0.40   δ=P_j × 0.40
  pause          q=1.00   δ=P_j
  migrate_{r'}   q=0.00   δ=P_j  (load removed from r0, added to r')

Objective: min Σ_j w_j × Σ_a q_a × z[j,a]

Constraints:
  1. Mutual exclusivity: Σ_a z[j,a] = 1  ∀j
  2. Curtailment:        Σ_j Σ_a δ_a(P_j) × z[j,a] ≥ Δ_r
  3. SLA:                Σ_a q_a × z[j,a] ≤ φ_{f_j}  ∀j
                         (Flex 0 jobs: dvfs and pause vars fixed to 0)
  4. Latency:            z[j, migrate_{r'}] = 0  if λ(r0→r') > budget_j
  5. Capacity:           Σ_j P_j × z[j, migrate_{r'}] ≤ headroom_{r'}  ∀r'

Requires Gurobi license. Setup:
  portal.gurobi.com → Licenses → Academic → WLS Academic License
  Download gurobi.lic → place at ~/gurobi.lic
  Verify: python -c "import gurobipy as gp; m = gp.Model(); print('OK')"

Usage:
  python -m src.simulation.mip_coordinator --test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False
    logger.warning("gurobipy not installed — MIPCoordinator will not work.")

from .workload import LATENCY_MS, Job, WorkloadEnsemble

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
    """Optimal dispatcher via MILP. Requires Gurobi license."""

    def __init__(self, time_limit_s: float = 30.0, log_to_console: bool = False):
        if not _GUROBI_AVAILABLE:
            raise RuntimeError(
                "gurobipy is not installed. "
                "Install with: pip install gurobipy"
            )
        self.time_limit_s = time_limit_s
        self.log_to_console = log_to_console
        # Create Env once; reusing it avoids repeated WLS license round-trips.
        self._env = gp.Env()
        self._env.setParam("OutputFlag", 1 if log_to_console else 0)
        self._env.setParam("LogToConsole", 1 if log_to_console else 0)
        self._env.setParam("TimeLimit", time_limit_s)

    def solve(
        self,
        ensemble: WorkloadEnsemble,
        stressed_region: str,
        curtailment_target_mw: float,
        headroom: dict[str, float],          # {region: available_headroom_mw}
        objective: Literal["qos", "carbon"] = "qos",
        carbon_intensity: dict[str, float] | None = None,
    ) -> DispatchResult:
        """Solve the dispatch MIP.

        Parameters
        ----------
        ensemble           : Current workload at the stressed datacenter
        stressed_region    : Region receiving the DR signal
        curtailment_target_mw : Target load reduction in MW
        headroom           : Available headroom (MW) at each non-stressed region
        objective          : "qos" (default) or "carbon" (for CarbonOptimized baseline)
        carbon_intensity   : Required when objective="carbon"; {region: gCO2/kWh}
        """
        import time
        t0 = time.perf_counter()

        jobs = ensemble.jobs
        other_regions = [r for r in headroom if r != stressed_region]

        # ── Build action catalogue for each job ──────────────────────────────
        # actions[j] = list of (action_name, target_region, q, delta_mw)
        action_catalogue: list[list[tuple[str, str | None, float, float]]] = []

        for job in jobs:
            φ = job.flex_tier.max_qos_degradation
            acts: list[tuple[str, str | None, float, float]] = []

            # Always available: do nothing
            acts.append(("nothing", None, 0.0, 0.0))

            # DVFS actions (not available for Flex 0)
            if job.flex_tier.value > 0:
                for name, (q, frac) in DVFS_LEVELS.items():
                    if q <= φ:
                        acts.append((name, None, q, job.power_mw * frac))

            # Pause (not available for Flex 0)
            if job.flex_tier.value > 0 and 1.0 <= φ:
                acts.append(("pause", None, 1.0, job.power_mw))

            # Migration — always available if latency + capacity allow.
            # Non-zero QoS cost reflects throughput loss during checkpoint
            # transfer and re-warm at the destination (see MIGRATION_QOS).
            for r_prime in other_regions:
                lat = LATENCY_MS.get((stressed_region, r_prime), 999.0)
                if lat <= job.latency_budget_ms:
                    q_mig = MIGRATION_QOS.get(job.job_type, 0.02)
                    acts.append(("migrate", r_prime, q_mig, job.power_mw))

            action_catalogue.append(acts)

        # ── Build Gurobi model ───────────────────────────────────────────────
        m = gp.Model(env=self._env)
        m.ModelName = "spatiotemporal_coordinator"

        # Binary variables z[j][k] = 1 if job j takes action k
        z = [
            [m.addVar(vtype=GRB.BINARY, name=f"z_{j}_{k}")
             for k in range(len(action_catalogue[j]))]
            for j in range(len(jobs))
        ]

        # ── Constraint 1: mutual exclusivity ────────────────────────────────
        for j in range(len(jobs)):
            m.addConstr(gp.quicksum(z[j]) == 1, name=f"mutex_{j}")

        # ── Constraint 2: curtailment compliance ────────────────────────────
        curtailment_expr = gp.LinExpr()
        for j, job in enumerate(jobs):
            for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                curtailment_expr += delta * z[j][k]
        m.addConstr(curtailment_expr >= curtailment_target_mw, name="curtailment")

        # ── Constraint 3: SLA (already enforced by action catalogue, but add
        #    as explicit constraint for completeness / dual variable access) ──
        for j, job in enumerate(jobs):
            φ = job.flex_tier.max_qos_degradation
            qos_expr = gp.quicksum(q * z[j][k]
                                   for k, (_, _, q, _) in enumerate(action_catalogue[j]))
            m.addConstr(qos_expr <= φ, name=f"sla_{j}")

        # ── Constraint 4: destination capacity ──────────────────────────────
        for r_prime in other_regions:
            cap = headroom.get(r_prime, 0.0)
            cap_expr = gp.LinExpr()
            for j, job in enumerate(jobs):
                for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                    if aname == "migrate" and target == r_prime:
                        cap_expr += job.power_mw * z[j][k]
            m.addConstr(cap_expr <= cap, name=f"capacity_{r_prime}")

        # ── Objective ────────────────────────────────────────────────────────
        if objective == "qos":
            obj = gp.LinExpr()
            for j, job in enumerate(jobs):
                for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                    obj += job.weight * q * z[j][k]
        elif objective == "carbon":
            if carbon_intensity is None:
                raise ValueError("carbon_intensity required for carbon objective")
            obj = gp.LinExpr()
            for j, job in enumerate(jobs):
                for k, (aname, target, q, delta) in enumerate(action_catalogue[j]):
                    if aname == "migrate" and target is not None:
                        ci = carbon_intensity.get(target, 0.0)
                    else:
                        ci = carbon_intensity.get(stressed_region, 0.0)
                    obj += ci * job.power_mw * z[j][k]
        else:
            raise ValueError(f"Unknown objective: {objective}")

        m.setObjective(obj, GRB.MINIMIZE)

        # ── Solve ────────────────────────────────────────────────────────────
        m.optimize()
        solve_time = time.perf_counter() - t0

        feasible = m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
        if not feasible:
            logger.warning(
                f"MIP infeasible (status={m.Status}) — curtailment target "
                f"{curtailment_target_mw:.1f} MW may exceed fleet capacity"
            )
            return DispatchResult(
                decisions={}, total_curtailment_mw=0.0, total_qos_cost=0.0,
                n_migrated=0, n_paused=0, n_dvfs=0, n_nothing=len(jobs),
                feasible=False, solve_time_s=solve_time,
            )

        # ── Extract solution ─────────────────────────────────────────────────
        decisions: dict[str, JobDecision] = {}
        total_curtailment = 0.0
        total_qos_cost = 0.0
        n_mig = n_pau = n_dvf = n_not = 0

        for j, job in enumerate(jobs):
            chosen_k = max(range(len(action_catalogue[j])),
                           key=lambda k: z[j][k].X)
            aname, target, q, delta = action_catalogue[j][chosen_k]

            decisions[job.job_id] = JobDecision(
                job_id=job.job_id,
                action=aname,
                target_region=target,
                qos_degradation=q,
                power_reduction_mw=delta,
            )
            total_curtailment += delta
            total_qos_cost += job.weight * q

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
    region_p90: dict[str, float],
) -> dict[str, float]:
    """Compute available headroom (MW) at each region.

    headroom = P90_threshold - current_load (floored at 0)
    """
    return {
        r: max(0.0, region_p90[r] - region_loads[r])
        for r in region_loads
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

def _test():
    """Quick smoke test on a synthetic 50-job ensemble."""
    from .workload import generate_ensemble

    print("=== MIP Coordinator smoke test ===")
    ensemble = generate_ensemble(ensemble_id=1, n_nodes=50, region="caiso")
    print(f"Ensemble: {ensemble.summary()}")

    curtailment_target = ensemble.total_power_mw * CURTAILMENT_FRACTION
    headroom = {"pjm": 2000.0, "bpa": 500.0}

    print(f"Curtailment target: {curtailment_target:.1f} MW")
    print(f"Headroom: {headroom}")

    coord = MIPCoordinator(log_to_console=False)
    result = coord.solve(
        ensemble=ensemble,
        stressed_region="caiso",
        curtailment_target_mw=curtailment_target,
        headroom=headroom,
    )

    print(f"\nResult: {result.summary()}")
    print(f"Curtailment achieved: {result.total_curtailment_mw:.1f} / {curtailment_target:.1f} MW")
    print(f"QoS cost: {result.total_qos_cost:.4f}")

    # Show action breakdown
    from collections import Counter
    actions = Counter(d.action for d in result.decisions.values())
    print(f"Actions: {dict(actions)}")

    migrate_targets = Counter(
        d.target_region for d in result.decisions.values()
        if d.action == "migrate"
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
