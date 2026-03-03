"""
Workload data structures and synthetic ensemble generation.

Replicates Emerald Table 1 workload ensembles:
  Ensemble 1: 80% training, 20% inference  (some Flex 0 inference)
  Ensemble 2: 50% training, 50% inference  (more Flex 0 inference)
  Ensemble 3: 50% training, 50% inference  (NO Flex 0 — all Flex 1-3)
  Ensemble 4: 90% training, 10% inference  (few Flex 0 inference)

Flex tier is a business decision, not a workload property — same inference job
can be Flex 0 (latency-critical) or Flex 3 (batch, tolerant of degradation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal


# ── Flex tiers ────────────────────────────────────────────────────────────────

class FlexTier(IntEnum):
    """QoS flexibility tier assigned by the operator, not the job type."""
    FLEX_0 = 0  # No local degradation; migration only
    FLEX_1 = 1  # Up to 5% degradation OK
    FLEX_2 = 2  # Up to 15% degradation OK
    FLEX_3 = 3  # Up to 30% degradation OK

    @property
    def max_qos_degradation(self) -> float:
        """Maximum allowable QoS degradation fraction (φ_{f})."""
        return {0: 0.00, 1: 0.05, 2: 0.15, 3: 0.30}[self.value]

    @property
    def weight(self) -> float:
        """Objective weight w_j — higher weight = more important to protect."""
        return 1.0 / (self.value + 1)  # Flex 0 → 1.0, Flex 3 → 0.25


# ── Job ───────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    """A single ML workload job running in a datacenter."""
    job_id: str
    job_type: Literal["training", "inference"]
    flex_tier: FlexTier
    power_mw: float        # Current power consumption in MW
    region: str            # Home region ("caiso", "pjm", "bpa")

    # Latency budget for migration (ms). Inference is latency-sensitive.
    latency_budget_ms: float = field(init=False)

    def __post_init__(self):
        # MLPerf TPOT budgets: inference ~50ms, training ~500ms (can tolerate migration lag)
        self.latency_budget_ms = 50.0 if self.job_type == "inference" else 500.0

    @property
    def weight(self) -> float:
        return self.flex_tier.weight

    def __repr__(self):
        return (
            f"Job({self.job_id}, {self.job_type}, Flex{self.flex_tier.value}, "
            f"{self.power_mw:.1f}MW, {self.region})"
        )


# ── Ensemble ──────────────────────────────────────────────────────────────────

@dataclass
class WorkloadEnsemble:
    """A collection of jobs representing a datacenter fleet state."""
    ensemble_id: int
    jobs: list[Job]
    region: str

    @property
    def total_power_mw(self) -> float:
        return sum(j.power_mw for j in self.jobs)

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    def summary(self) -> dict:
        from collections import Counter
        flex_counts = Counter(j.flex_tier.value for j in self.jobs)
        type_counts = Counter(j.job_type for j in self.jobs)
        return {
            "ensemble_id": self.ensemble_id,
            "n_jobs": self.n_jobs,
            "total_power_mw": round(self.total_power_mw, 1),
            "flex_distribution": dict(flex_counts),
            "type_distribution": dict(type_counts),
        }


# ── Power sampling ────────────────────────────────────────────────────────────

# Per-node power in MW: training is GPU-heavy (~8 MW), inference lighter (~4 MW)
_POWER_PARAMS = {
    "training":  {"mean": 8.0, "std": 1.5, "min": 4.0, "max": 12.0},
    "inference": {"mean": 4.0, "std": 0.8, "min": 2.0, "max":  6.0},
}


def _sample_power(job_type: str, rng: random.Random) -> float:
    p = _POWER_PARAMS[job_type]
    val = rng.gauss(p["mean"], p["std"])
    return max(p["min"], min(p["max"], round(val, 2)))


# ── Ensemble generation ───────────────────────────────────────────────────────

def _make_flex_tier(job_type: str, flex0_ids: set[int], idx: int) -> FlexTier:
    """Assign flex tier: Flex 0 for designated inference nodes, random 1-3 for others."""
    if job_type == "inference" and idx in flex0_ids:
        return FlexTier.FLEX_0
    if job_type == "inference":
        # Inference not Flex 0: mostly Flex 1, some Flex 2
        return FlexTier.FLEX_1
    # Training: distributed across Flex 1-3
    return FlexTier(1 + (idx % 3))


def generate_ensemble(
    ensemble_id: int,
    n_nodes: int = 100,
    region: str = "caiso",
    seed: int = 42,
) -> WorkloadEnsemble:
    """Generate a synthetic workload ensemble matching Emerald Table 1.

    Parameters
    ----------
    ensemble_id : 1-4 (matches Emerald Table 1 ensembles)
    n_nodes     : Total number of compute nodes (jobs)
    region      : Home region for all jobs
    seed        : RNG seed for reproducibility

    Ensemble compositions:
        1: 80% training, 20% inference — 20% of inference = Flex 0 (6 nodes in Emerald)
        2: 50% training, 50% inference — ~28% of inference = Flex 0 (14 nodes in Emerald)
        3: 50% training, 50% inference — NO Flex 0 (all inference = Flex 1-3)
        4: 90% training, 10% inference — ~40% of inference = Flex 0 (4 nodes in Emerald)
    """
    if ensemble_id not in (1, 2, 3, 4):
        raise ValueError(f"ensemble_id must be 1-4, got {ensemble_id}")

    rng = random.Random(seed)

    # Training / inference split
    training_fractions = {1: 0.80, 2: 0.50, 3: 0.50, 4: 0.90}
    n_training = round(n_nodes * training_fractions[ensemble_id])
    n_inference = n_nodes - n_training

    # How many inference nodes are Flex 0 (from Emerald Table 1, scaled to n_nodes=30)
    # Emerald used 30 nodes total; scale proportionally
    flex0_fractions = {1: 6/6, 2: 14/15, 3: 0.0, 4: 4/3}
    # Actual counts
    if ensemble_id == 3:
        n_flex0_inference = 0
    else:
        # In Emerald: ensemble1=6 Flex0 out of ~6 inference nodes → ~100%
        # ensemble2=14 Flex0 out of ~15 inference nodes → ~93%
        # ensemble4=4 Flex0 out of ~3 inference nodes → cap at n_inference
        raw = {1: 1.0, 2: 0.93, 4: 0.40}[ensemble_id]
        n_flex0_inference = min(n_inference, round(n_inference * raw))

    flex0_inference_ids = set(rng.sample(range(n_inference), n_flex0_inference))

    jobs = []
    for i in range(n_training):
        jid = f"train_{i:04d}"
        tier = FlexTier(1 + (i % 3))  # Flex 1/2/3 round-robin for training
        jobs.append(Job(
            job_id=jid,
            job_type="training",
            flex_tier=tier,
            power_mw=_sample_power("training", rng),
            region=region,
        ))

    for i in range(n_inference):
        jid = f"infer_{i:04d}"
        tier = FlexTier.FLEX_0 if i in flex0_inference_ids else FlexTier.FLEX_1
        jobs.append(Job(
            job_id=jid,
            job_type="inference",
            flex_tier=tier,
            power_mw=_sample_power("inference", rng),
            region=region,
        ))

    rng.shuffle(jobs)
    return WorkloadEnsemble(ensemble_id=ensemble_id, jobs=jobs, region=region)


# ── Latency matrix ────────────────────────────────────────────────────────────

# P99 inter-region latency in ms (from CLAUDE.md)
LATENCY_MS: dict[tuple[str, str], float] = {
    ("caiso", "pjm"): 70.0,
    ("pjm", "caiso"): 70.0,
    ("caiso", "bpa"): 20.0,
    ("bpa", "caiso"): 20.0,
    ("pjm", "bpa"):   62.0,
    ("bpa", "pjm"):   62.0,
}

REGIONS = ["caiso", "pjm", "bpa"]
