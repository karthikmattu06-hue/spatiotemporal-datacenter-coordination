"""
Baseline dispatch strategies for comparison against the MIP coordinator.

All baselines share the same interface as MIPCoordinator.solve() and return
a DispatchResult so results are directly comparable.

Baselines:
  1. NoCoordination   — do nothing (0% curtailment)
  2. TemporalOnly     — greedy local actions only, no migration (Emerald replication)
  3. SpatialNaive     — migrate everything possible, ignores SLA
  4. CarbonOptimized  — MIP with carbon-intensity objective (wrong signal, ablation)
  5. OracleOptimal    — MIP with perfect curtailment knowledge (offline upper bound)
"""

from __future__ import annotations

import time
from collections import defaultdict

from .mip_coordinator import (
    CURTAILMENT_FRACTION,
    DVFS_LEVELS,
    MIGRATION_QOS,
    DispatchResult,
    JobDecision,
    MIPCoordinator,
)
from .workload import LATENCY_MS, WorkloadEnsemble


# ── 1. No Coordination ────────────────────────────────────────────────────────

class NoCoordination:
    """Do nothing — represents the pre-DR baseline (0% curtailment achieved)."""

    def solve(
        self,
        ensemble: WorkloadEnsemble,
        stressed_region: str,
        curtailment_target_mw: float,
        headroom: dict[str, float],
        **kwargs,
    ) -> DispatchResult:
        t0 = time.perf_counter()
        decisions = {
            job.job_id: JobDecision(
                job_id=job.job_id,
                action="nothing",
                target_region=None,
                qos_degradation=0.0,
                power_reduction_mw=0.0,
            )
            for job in ensemble.jobs
        }
        return DispatchResult(
            decisions=decisions,
            total_curtailment_mw=0.0,
            total_qos_cost=0.0,
            n_migrated=0, n_paused=0, n_dvfs=0, n_nothing=len(ensemble.jobs),
            feasible=True,
            solve_time_s=time.perf_counter() - t0,
        )


# ── 2. Temporal-Only (Emerald replication) ────────────────────────────────────

class TemporalOnly:
    """Greedy local-only curtailment: DVFS then pause, highest-flex jobs first.

    No migration. Replicates Emerald's 'DVFS + Job Pausing, Fair' policy.
    Priority: Flex 3 → Flex 2 → Flex 1 → Flex 0 (never touch Flex 0 locally).
    Within a tier, apply the mildest action that contributes to the target.
    """

    def solve(
        self,
        ensemble: WorkloadEnsemble,
        stressed_region: str,
        curtailment_target_mw: float,
        headroom: dict[str, float],
        **kwargs,
    ) -> DispatchResult:
        t0 = time.perf_counter()

        # Sort: highest flex tier first (most tolerant), lowest power last
        jobs_sorted = sorted(
            ensemble.jobs,
            key=lambda j: (-j.flex_tier.value, -j.power_mw),
        )

        decisions: dict[str, JobDecision] = {}
        remaining = curtailment_target_mw

        for job in jobs_sorted:
            φ = job.flex_tier.max_qos_degradation
            if φ == 0.0:
                # Flex 0: no local action allowed
                decisions[job.job_id] = JobDecision(
                    job.job_id, "nothing", None, 0.0, 0.0
                )
                continue

            if remaining <= 0:
                decisions[job.job_id] = JobDecision(
                    job.job_id, "nothing", None, 0.0, 0.0
                )
                continue

            # Try mildest DVFS level that satisfies SLA
            action_taken = None
            for name, (q, frac) in sorted(DVFS_LEVELS.items()):
                if q <= φ:
                    delta = job.power_mw * frac
                    action_taken = (name, None, q, delta)
                    break

            # If DVFS isn't sufficient or available, try pause (last resort)
            if action_taken is None and φ >= 1.0:
                action_taken = ("pause", None, 1.0, job.power_mw)

            if action_taken:
                aname, target, q, delta = action_taken
                remaining -= delta
                decisions[job.job_id] = JobDecision(
                    job.job_id, aname, target, q, delta
                )
            else:
                decisions[job.job_id] = JobDecision(
                    job.job_id, "nothing", None, 0.0, 0.0
                )

        return _build_result(decisions, ensemble, time.perf_counter() - t0)


# ── 3. Spatial-Naive ─────────────────────────────────────────────────────────

class SpatialNaive:
    """Migrate as many jobs as possible to highest-headroom region, ignoring SLA.

    Demonstrates why unconstrained migration causes SLA violations:
    Flex 0 inference jobs get migrated even though that's fine (q=0),
    but no prioritization means high-flex local actions are skipped.
    """

    def solve(
        self,
        ensemble: WorkloadEnsemble,
        stressed_region: str,
        curtailment_target_mw: float,
        headroom: dict[str, float],
        **kwargs,
    ) -> DispatchResult:
        t0 = time.perf_counter()

        other_regions = [r for r in headroom if r != stressed_region]
        if not other_regions:
            return NoCoordination().solve(
                ensemble, stressed_region, curtailment_target_mw, headroom
            )

        # Pick region with most headroom
        target_region = max(other_regions, key=lambda r: headroom.get(r, 0))

        decisions: dict[str, JobDecision] = {}
        remaining_headroom = headroom.get(target_region, 0.0)
        remaining_curtailment = curtailment_target_mw

        for job in sorted(ensemble.jobs, key=lambda j: -j.power_mw):
            lat = LATENCY_MS.get((stressed_region, target_region), 999.0)
            can_migrate = (
                lat <= job.latency_budget_ms
                and remaining_headroom >= job.power_mw
                and remaining_curtailment > 0
            )
            if can_migrate:
                remaining_headroom -= job.power_mw
                remaining_curtailment -= job.power_mw
                q_mig = MIGRATION_QOS.get(job.job_type, 0.02)
                decisions[job.job_id] = JobDecision(
                    job.job_id, "migrate", target_region, q_mig, job.power_mw
                )
            else:
                decisions[job.job_id] = JobDecision(
                    job.job_id, "nothing", None, 0.0, 0.0
                )

        return _build_result(decisions, ensemble, time.perf_counter() - t0)


# ── 4. Carbon-Optimized ───────────────────────────────────────────────────────

class CarbonOptimized:
    """MIP coordinator with carbon-minimization objective instead of QoS.

    Included as an ablation to show that optimizing for the wrong signal
    (carbon intensity) degrades QoS outcomes vs. the load-stress signal.
    Uses placeholder carbon intensities; real values come from EIA fuel-type data.
    """

    # Placeholder carbon intensities (gCO2/kWh) — rough regional averages
    DEFAULT_CARBON = {
        "caiso": 210.0,   # CA grid: ~35% renewables
        "pjm":   400.0,   # PJM: ~40% coal/gas mix
        "bpa":   30.0,    # BPA: ~90% hydro
    }

    def __init__(self, carbon_intensity: dict[str, float] | None = None):
        self.carbon_intensity = carbon_intensity or self.DEFAULT_CARBON
        self._coord = MIPCoordinator(log_to_console=False)

    def solve(
        self,
        ensemble: WorkloadEnsemble,
        stressed_region: str,
        curtailment_target_mw: float,
        headroom: dict[str, float],
        **kwargs,
    ) -> DispatchResult:
        return self._coord.solve(
            ensemble=ensemble,
            stressed_region=stressed_region,
            curtailment_target_mw=curtailment_target_mw,
            headroom=headroom,
            objective="carbon",
            carbon_intensity=self.carbon_intensity,
        )


# ── 5. Oracle Optimal ────────────────────────────────────────────────────────

class OracleOptimal:
    """MIP with perfect curtailment knowledge — theoretical offline upper bound.

    Difference from MIPCoordinator: uses the actual (known) peak curtailment
    demand across the full event duration rather than just the current-hour
    target. This gives the best possible outcome, serving as a ceiling.
    """

    def __init__(self, peak_multiplier: float = 1.0):
        # peak_multiplier > 1 means Oracle knows the event peaks higher
        self.peak_multiplier = peak_multiplier
        self._coord = MIPCoordinator(log_to_console=False)

    def solve(
        self,
        ensemble: WorkloadEnsemble,
        stressed_region: str,
        curtailment_target_mw: float,
        headroom: dict[str, float],
        **kwargs,
    ) -> DispatchResult:
        # Oracle uses the event peak — same target for now (run_simulation
        # can override curtailment_target_mw with the event peak when calling)
        return self._coord.solve(
            ensemble=ensemble,
            stressed_region=stressed_region,
            curtailment_target_mw=curtailment_target_mw * self.peak_multiplier,
            headroom=headroom,
            objective="qos",
        )


# ── Helper ───────────────────────────────────────────────────────────────────

def _build_result(
    decisions: dict[str, JobDecision],
    ensemble: WorkloadEnsemble,
    solve_time: float,
) -> DispatchResult:
    total_curtailment = sum(d.power_reduction_mw for d in decisions.values())
    total_qos = sum(
        j.weight * decisions[j.job_id].qos_degradation
        for j in ensemble.jobs
        if j.job_id in decisions
    )
    n_mig = sum(1 for d in decisions.values() if d.action == "migrate")
    n_pau = sum(1 for d in decisions.values() if d.action == "pause")
    n_dvf = sum(1 for d in decisions.values() if "dvfs" in d.action)
    n_not = sum(1 for d in decisions.values() if d.action == "nothing")

    return DispatchResult(
        decisions=decisions,
        total_curtailment_mw=total_curtailment,
        total_qos_cost=total_qos,
        n_migrated=n_mig,
        n_paused=n_pau,
        n_dvfs=n_dvf,
        n_nothing=n_not,
        feasible=True,
        solve_time_s=solve_time,
    )


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_BASELINES: dict[str, object] = {
    "no_coordination": NoCoordination(),
    "temporal_only":   TemporalOnly(),
    "spatial_naive":   SpatialNaive(),
    "oracle_optimal":  OracleOptimal(),
}
