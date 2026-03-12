# Spatiotemporal Coordination for Grid-Responsive AI Datacenters

**Joint temporal + spatial workload optimization for multi-region datacenter demand response**

---

## Overview

AI datacenter electricity demand is projected to reach 100 GW by 2030. During grid stress events,
temporal-only strategies (GPU frequency scaling, job pausing) are limited to workloads that can
tolerate local QoS degradation, leaving latency-sensitive inference ("Flex 0") untouched.

This project adds a **spatial control knob — geographic migration** — formulated as a MILP that
jointly optimizes DVFS, pausing, and cross-region migration subject to SLA, latency, and capacity
constraints. Using 5 years of public grid data from CAISO, PJM, and BPA, we show:

- Grid stress peaks are **temporally offset** by 4–6 hours between regions (ρ < 0.3, simultaneous
  stress < 5% of hours), providing a stable migration target.
- The MIP coordinator achieves **93% lower QoS cost** than temporal-only methods at the same
  10% curtailment depth (clean ablation: same MILP solver, only migration disabled).
- All results use open-source data and the HiGHS solver (no commercial license required).

---

## Quick Start

```bash
git clone https://github.com/karthikmattu/spatiotemporal-datacenter-coordination.git
cd spatiotemporal-datacenter-coordination

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # no Gurobi, no license needed

# Verify MIP solver works (solves a 50-job instance in ~12 ms)
python -m src.simulation.mip_coordinator --test

# Run full curtailment-fraction sweep (3 regions × 6 depths × 3 strategies)
python -m src.simulation.run_sweep

# Generate all figures
python -m src.plotting.visualize_sweep
python -m src.plotting.visualize_event
python -m src.plotting.visualize_simulation
```

Pre-fetched grid data and simulation results are required before the sweep.
See [Pipeline](#pipeline) below.

---

## Key Results

| Strategy | Curtailment @ 10% | QoS cost | Notes |
|---|---|---|---|
| **MIP Coordinator (ours)** | **10.1%** | **0.25** | Optimal; uses migration for Flex 0 |
| Temporal-Only (no migration) | 10.0% | 3.69 | Same MILP, migration disabled |
| No Coordination | 0.0% | 0.00 | Status quo |

CAISO, 957 stress events, Ensemble 1. QoS reduction from migration: **93%**.
Verified on 192 simultaneous-stress hours: MIP = Temporal-Only (delta = 0),
confirming the gap is purely attributable to the spatial degree of freedom.

---

## Pipeline

Run steps in order. Each step's output is the next step's input.

```bash
# Step 1 — Fetch raw grid data (one-time; ~minutes)
python src/data_collection/fetch_caiso.py
python src/data_collection/fetch_pjm.py
python src/data_collection/fetch_bpa.py

# Step 2 — Clean and merge to hourly parquet
python src/data_collection/clean_caiso.py   # → data/processed/caiso/caiso_merged_hourly.parquet
# (analogous scripts for pjm, bpa)

# Step 3 — Annotate stress events (P90 threshold, ≥2h duration filter)
python src/analysis/stress_analysis.py --region caiso
python src/analysis/stress_analysis.py --region pjm
python src/analysis/stress_analysis.py --region bpa

# Step 4 — Run simulation (MIP + all baselines, one region)
python -m src.simulation.run_simulation --region caiso --ensemble 1

# Step 5 — Run curtailment-fraction sweep (all regions, 5–30%)
python -m src.simulation.run_sweep

# Step 6 — Generate figures
python -m src.plotting.visualize_sweep       # sweep_curtailment/qos/comparison.png
python -m src.plotting.visualize_event       # event_combined.png
python -m src.plotting.visualize_simulation  # curtailment/qos_comparison, action_breakdown.png
```

---

## Repository Structure

```
src/
  data_collection/    fetch_{caiso,pjm,bpa}.py  clean_{caiso,pjm,bpa}.py
  analysis/           stress_analysis.py  complementarity.py
  simulation/         workload.py  mip_coordinator.py  baselines.py
                      run_simulation.py  run_sweep.py
  plotting/           visualize_stress.py  visualize_sweep.py  visualize_event.py
                      visualize_simulation.py
data/
  raw/{caiso,pjm,bpa}/            monthly CSVs  (.gitignored)
  processed/{caiso,pjm,bpa}/      parquet files (.gitignored)
  processed/simulation/           results_*.parquet  sweep_summary.csv
figures/
  caiso/   pjm/   bpa/   simulation/
paper/
  main.tex  refs.bib
```

---

## MIP Formulation

Binary variable **x[j,k] ∈ {0,1}** — assignment of job j to action k.

Actions per job (mutually exclusive):

| Action | QoS degradation q | Power reduction δ |
|---|---|---|
| nothing | 0.00 | 0 |
| dvfs_0.9 | 0.10 | P_j × 0.10 |
| dvfs_0.8 | 0.20 | P_j × 0.20 |
| dvfs_0.7 | 0.30 | P_j × 0.30 |
| dvfs_0.6 | 0.40 | P_j × 0.40 |
| pause | 1.00 | P_j |
| migrate_r′ | 0.01–0.02 | P_j (removed from r₀) |

**Objective:** min Σ_j w_j · Σ_k q_k · x[j,k]

**Constraints:**
1. Mutual exclusivity: Σ_k x[j,k] = 1 ∀j
2. Curtailment: Σ_j Σ_k δ_k · x[j,k] ≥ Δ_r
3. SLA: Σ_k q_k · x[j,k] ≤ φ_{f_j} ∀j
4. Latency: x[j, migrate_r′] = 0 if λ(r₀→r′) > budget_j
5. Capacity: Σ_j P_j · x[j, migrate_r′] ≤ headroom_r′ ∀r′

Solved with **HiGHS** via `scipy.optimize.milp` — fully open-source, no license needed.
Typical solve time: 10–30 ms for 20–200 jobs.

---

## Workload Ensembles

Following Table 1 of Colangelo et al. (2025):

| ID | Training | Inference | Flex 0 inference |
|---|---|---|---|
| E1 | 80% | 20% | 10% of fleet |
| E2 | 50% | 50% | 14% of fleet |
| E3 | 50% | 50% | 0% (no Flex 0) |
| E4 | 90% | 10% | 4% of fleet |

Flex tiers: 0 → φ=0 (migration only), 1 → φ=0.05, 2 → φ=0.15, 3 → φ=0.30

---

## Data Sources

All grid data is publicly available, no registration required for CAISO and BPA.

| Region | Source | Coverage |
|---|---|---|
| CAISO | [gridstatus](https://github.com/kmax12/gridstatus) / OASIS | 2020–2024, hourly |
| PJM | [gridstatus](https://github.com/kmax12/gridstatus) / Data Miner 2 | 2020–2024, hourly |
| BPA | EIA API `electricity/rto/region-data` | 2020–2024, hourly |

---

## Citation

If you use this code or data, please cite:

```bibtex
@misc{mattu2025spatiotemporal,
  title  = {Spatiotemporal Coordination for Grid-Responsive {AI} Datacenters},
  author = {Mattu, Karthik},
  year   = {2025},
  url    = {https://github.com/karthikmattu/spatiotemporal-datacenter-coordination}
}
```

This work builds directly on:

```bibtex
@misc{emerald2025,
  title         = {Turning {AI} Data Centers into Grid-Interactive Assets: Results from a Field
                   Demonstration in {Phoenix, Arizona}},
  author        = {Colangelo, Philip and Coskun, Ayse K. and Megrue, Jack and others},
  year          = {2025},
  eprint        = {2507.00909},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

**Author:** Karthik Mattu — M.S. Student, Golisano College of Computing and Information Sciences,
Rochester Institute of Technology
