# Comparative Evaluation of Classical and Evolutionary Algorithms for Dynamic Production Scheduling Under Machine Disruptions

## Project Overview
This repository implements a rigorous comparative evaluation framework for **Dynamic Job Shop Scheduling Problems (DJSSP)** under stochastic/deterministic **machine breakdown disruptions**.

The project evaluates and compares:
1. **Classical Priority Dispatching Rules**: First-Come First-Served (**FCFS**) and Shortest Processing Time (**SPT**).
2. **Single-Objective Evolutionary Algorithm**: Standard Genetic Algorithm (**GA**) minimizing Makespan ($C_{\max}$).
3. **Reactive Shift Rescheduling Baseline**: Right-Shift Rescheduling (preserving sequence order).
4. **Multi-Objective Evolutionary Algorithm**: **NSGA-II** simultaneously optimizing:
   - **Makespan ($f_1 = C_{\max}$)**: Minimizing total production completion time.
   - **Total Tardiness ($f_2 = \sum T_j$)**: Minimizing customer delivery delays beyond due dates ($d_j$).
   - **Schedule Instability ($f_3 = \Delta S$)**: Minimizing shop floor nervousness / start-time deviations from the baseline schedule.

---

## Directory Layout
```
d:/SEM 5/LNO PROJECT/
├── data/
│   └── benchmarks/             # Standard JSPLib instances (FT06, FT10, LA01, LA02)
├── src/
│   ├── problem.py              # Operation, Job, and JSPInstance models + due-date generator (TWK)
│   ├── schedule.py             # Operation-based chromosome decoder (Bierwirth representation)
│   ├── baselines.py            # FCFS and SPT priority dispatching rules
│   ├── disruption.py           # Machine breakdown model & Right-Shift Rescheduling
│   ├── ga.py                   # Single-Objective GA with Job-based Crossover (POX/JOX)
│   ├── nsga2.py                # Fast Non-Dominated Sorting & Crowding Distance MOEA
│   └── visualize.py            # High-resolution Gantt charts & 3D/2D Pareto Front plotters
├── experiments/
│   └── run_experiments.py      # Multi-seed batch experiment runner
├── results/                    # Generated charts (PNG) and tabular metrics (CSV)
├── run_demo.py                 # End-to-end prototype demo script
└── LITERATURE_REVIEW.md        # Comprehensive literature survey & dataset selection justification
```

---

## Quickstart

### 1. Run the End-to-End Prototype
```bash
python run_demo.py
```
This executes FCFS, SPT, Single-Objective GA, injects a machine breakdown on Machine 2, evaluates Right-Shift Rescheduling, runs NSGA-II to discover Pareto trade-offs, and saves all Gantt charts to `results/`.

### 2. Run Batch Benchmark Experiments across Random Seeds
```bash
python experiments/run_experiments.py
```
Outputs aggregated comparative metrics to `results/experiment_summary_ft06.csv`.

---

## Key Experimental Results on FT06 Benchmark

| Method | Phase | Makespan ($C_{\max}$) | Total Tardiness ($\sum T_j$) | Instability ($\Delta S$) | Average Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **FCFS** | Pre-Disruption Baseline | 65 | 58 | - | 0.10 ms |
| **SPT** | Pre-Disruption Baseline | 109 | 131 | - | 0.08 ms |
| **Single-Obj GA** | Pre-Disruption (Predictive) | 59 | 36 | - | 0.16 s |
| **Right-Shift** | Post-Disruption (Reactive) | 68 | 75 | 97 | 0.05 ms |
| **NSGA-II (Min Makespan)** | Post-Disruption (Multi-Obj) | **61** | **48** | 122 | 5.67 s (front) |
| **NSGA-II (Min Instability)** | Post-Disruption (Multi-Obj) | **65** | **51** | **99** | 5.67 s (front) |

> **Key Research Finding**: When Machine 2 breaks down for 15 time units, simple Right-Shift rescheduling inflates makespan to 68 and tardiness to 75. NSGA-II discovers non-dominated schedules that recover makespan back down to 61 and tardiness down to 48 while keeping shop floor instability under controlled limits.


## Evaluation-friendly comparison charts and tests

The original `python run_demo.py` still produces the five existing figures.
It now additionally generates `results/06_algorithm_comparison.png`,
`results/07_post_disruption_comparison.png` and `results/comparison_summary.csv`.
The first chart labels pre-disruption and post-disruption phases separately;
**pre-disruption and post-disruption metrics are not a controlled head-to-head comparison**.
Charts are generated from the live run (not hardcoded README values).

Run basic regression tests with:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
python run_demo.py
```

The tests cover chromosome counts, operation precedence, machine overlaps,
hand-computed metrics, and simple right-shift disruption cases. They do not
claim that the full dynamic schedule is formally verified. The breakdown
model treats an interrupted operation as resume-after-repair; the current
Gantt representation uses single start/end rectangles, so uninterrupted
processing should not be inferred from an interrupted block's drawing.
