# Spatiotemporal Coordination for Grid-Responsive AI Datacenters

**Unlocking flexibility for latency-sensitive AI workloads through multi-region workload migration**

---

## Overview

AI datacenter electricity demand is projected to reach 100 GW by 2030, but grid infrastructure cannot keep pace — new datacenters face 5–8 year interconnection queues. Recent work by [Emerald AI](https://emeraldai.com) demonstrated that software-only orchestration can reduce datacenter power by **25% during peak grid events** through temporal flexibility (GPU frequency scaling, job pausing, resource reallocation).

However, this approach leaves **Flex 0 workloads** — real-time inference with strict latency SLAs — completely untouched. As inference grows to dominate datacenter compute, this gap becomes critical.

This project extends temporal flexibility with **spatial flexibility**: migrating Flex 0 workloads to geographically distant datacenters in non-stressed grid regions, rather than throttling them. By exploiting complementary grid stress patterns across U.S. regions (PJM, CAISO, BPA), the framework targets **35–40% total power reduction** — a 60% improvement over temporal-only methods — while preserving P99 latency SLAs.

## Research Question

> Can hyperscalers leverage their existing multi-region infrastructure to achieve 35–40% grid stress reduction by coordinating workload migration between datacenters with complementary grid stress patterns, thereby unlocking flexibility for Flex 0 workloads while preserving end-user performance?

## Approach

The system extends Emerald Conductor's architecture with a fourth control knob — **geographic migration** — and formulates the joint temporal-spatial orchestration as a mixed-integer program (MIP):

- **Temporal knobs** (DVFS, pausing, reallocation) handle Flex 1–3 workloads locally, replicating Emerald's demonstrated 25% reduction
- **Spatial knob** (migration) routes Flex 0 workloads to non-stressed regions where they run at full performance
- **Optimization** minimizes total QoS degradation subject to curtailment targets, per-job SLA constraints, latency budgets, and destination capacity limits

## Key Components

| Component | Description | Status |
|-----------|-------------|--------|
| Grid Stress Complementarity Analysis | Empirical validation using PJM/CAISO/BPA hourly data (2020–2024) | 🔄 In Progress |
| MIP Formulation | Joint temporal-spatial optimization with SLA constraints | ✅ Complete |
| Simulation Framework | Workload migration simulator with realistic latency models | ⬜ Not Started |
| Baseline Comparisons | No coordination, temporal-only, spatial-naive, carbon-optimized, oracle | ⬜ Not Started |

## Experimental Baselines

| # | Baseline | Expected Reduction |
|---|----------|--------------------|
| 1 | No Coordination (status quo) | 0% |
| 2 | Temporal-Only (Emerald replication) | ~25% |
| 3 | Spatial-Only, Naive (no constraints) | Uncontrolled |
| 4 | Carbon-Optimized (CarbonClipper/CICS) | Moderate |
| 5 | Oracle Offline Optimal | Theoretical max |

## Repository Structure

```
├── src/
│   ├── data_collection/      # Grid data download scripts (PJM, CAISO, BPA)
│   ├── analysis/             # Complementarity analysis, correlation metrics
│   ├── simulation/           # MIP solver, workload migration simulator
│   └── plotting/             # Figure generation for paper
├── data/                     # Raw and processed grid data (.gitignored)
├── figures/                  # Generated plots
├── notebooks/                # Exploratory analysis
└── docs/                     # Research notes, references
```

## Data Sources

All grid data is freely available from public sources:

| Source | Portal | Data Available |
|--------|--------|----------------|
| PJM | [Data Miner 2](https://dataminer2.pjm.com) | Hourly load, LMPs, reserves (1993–present) |
| CAISO | [OASIS](https://oasis.caiso.com) | Hourly load, LMPs, reserves, net demand |
| BPA | [transmission.bpa.gov](https://transmission.bpa.gov/business/operations/wind/) | 5-min load, hydro, wind, thermal generation |

See [`src/data_collection/README.md`](src/data_collection/README.md) for download instructions.

## Setup

```bash
git clone https://github.com/<your-username>/spatiotemporal-datacenter-coordination.git
cd spatiotemporal-datacenter-coordination
pip install -r requirements.txt

# Download grid data
python src/data_collection/fetch_pjm.py
python src/data_collection/fetch_caiso.py
python src/data_collection/fetch_bpa.py
```

## Built On

This research directly extends the limitations identified in **Section 3.5** of:

> Colangelo, Sivaram et al., "Turning AI Data Centers into Grid-Interactive Assets: Results from a Field Demonstration in Phoenix, Arizona," arXiv:2507.00909, July 2025.

## References

- Colangelo et al., [Emerald AI Phoenix Demonstration](https://arxiv.org/abs/2507.00909) (2025)
- Norris et al., [100 GW Flexible Load Opportunity](https://hdl.handle.net/10161/32077) — Duke Nicholas Institute (2025)
- Zheng et al., [Load Migration Between Data Centers](https://doi.org/10.1016/j.joule.2020.08.001) — Joule (2020)
- Lechowicz et al., [CarbonClipper: SOAD Algorithms](https://doi.org/10.1145/3626776) — ACM SIGMETRICS (2024)

## Author

**Karthik Mattu**
M.S. Student, Golisano College of Computing and Information Sciences
Rochester Institute of Technology

## License

MIT License — see [LICENSE](LICENSE).
