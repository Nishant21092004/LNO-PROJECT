"""
run_experiments.py: Systematic multi-seed benchmark experiment runner.
Executes FCFS, SPT, Single-Objective GA, Right-Shift, and NSGA-II
across standard benchmarks (FT06, FT10, LA01) and saves results to CSV.
"""

import os
import time
import pandas as pd
import numpy as np
from src.problem import load_jsp_instance
from src.baselines import solve_fcfs, solve_spt
from src.ga import run_genetic_algorithm
from src.disruption import BreakdownDisruption, apply_right_shift, partition_disrupted_state
from src.nsga2 import run_nsga2


def evaluate_instance_experiment(
    benchmark_name: str,
    file_path: str,
    disruption_machine: int,
    disruption_time: int,
    disruption_duration: int,
    num_seeds: int = 5,
    ga_pop: int = 50,
    ga_gen: int = 80,
    nsga_pop: int = 60,
    nsga_gen: int = 70,
):
    print(f"\n=======================================================")
    print(f" EXPERIMENT: {benchmark_name} (Disruption M{disruption_machine} at t={disruption_time}, dur={disruption_duration})")
    print(f"=======================================================")

    instance = load_jsp_instance(file_path, tightness_factor=1.5)
    disruption = BreakdownDisruption(
        machine_id=disruption_machine,
        breakdown_start=disruption_time,
        duration=disruption_duration,
    )

    records = []

    # 1. Deterministic baselines
    t0 = time.time()
    sched_fcfs = solve_fcfs(instance)
    fcfs_time = time.time() - t0
    records.append({
        "Instance": benchmark_name,
        "Algorithm": "FCFS",
        "Phase": "Pre-Disruption",
        "Seed": "-",
        "Makespan": sched_fcfs.makespan,
        "Tardiness": sched_fcfs.total_tardiness,
        "Instability": 0,
        "Runtime_sec": fcfs_time,
    })

    t0 = time.time()
    sched_spt = solve_spt(instance)
    spt_time = time.time() - t0
    records.append({
        "Instance": benchmark_name,
        "Algorithm": "SPT",
        "Phase": "Pre-Disruption",
        "Seed": "-",
        "Makespan": sched_spt.makespan,
        "Tardiness": sched_spt.total_tardiness,
        "Instability": 0,
        "Runtime_sec": spt_time,
    })

    # 2. Stochastic algorithms over random seeds
    for seed in range(num_seeds):
        # A. Single-Objective GA Initial
        t0 = time.time()
        initial_ga_sched, _, _ = run_genetic_algorithm(
            instance,
            pop_size=ga_pop,
            generations=ga_gen,
            seed=seed * 17 + 1,
        )
        ga_time = time.time() - t0
        records.append({
            "Instance": benchmark_name,
            "Algorithm": "Single-Obj GA",
            "Phase": "Pre-Disruption",
            "Seed": seed,
            "Makespan": initial_ga_sched.makespan,
            "Tardiness": initial_ga_sched.total_tardiness,
            "Instability": 0,
            "Runtime_sec": ga_time,
        })

        # B. Right-Shift Baseline
        t0 = time.time()
        sched_rs = apply_right_shift(initial_ga_sched, disruption, instance)
        rs_time = time.time() - t0
        rs_instab = sched_rs.compute_instability(initial_ga_sched, impacted_only_since=disruption.breakdown_start)
        records.append({
            "Instance": benchmark_name,
            "Algorithm": "Right-Shift",
            "Phase": "Post-Disruption",
            "Seed": seed,
            "Makespan": sched_rs.makespan,
            "Tardiness": sched_rs.total_tardiness,
            "Instability": rs_instab,
            "Runtime_sec": rs_time,
        })

        # C. NSGA-II Multi-Objective Rescheduling
        completed_ops, m_ready, j_ready, j_offsets, rem_pool = partition_disrupted_state(
            initial_ga_sched, disruption, instance
        )
        t0 = time.time()
        pareto_scheds, pareto_objs = run_nsga2(
            instance,
            baseline_schedule=initial_ga_sched,
            pop_size=nsga_pop,
            generations=nsga_gen,
            machine_ready_times=m_ready,
            job_ready_times=j_ready,
            job_op_offsets=j_offsets,
            disruption_time=disruption.breakdown_start,
            seed=seed * 23 + 7,
        )
        nsga_time = time.time() - t0

        # Record representative solutions on Pareto front
        # 1. Best Makespan
        records.append({
            "Instance": benchmark_name,
            "Algorithm": "NSGA-II (Min Makespan)",
            "Phase": "Post-Disruption",
            "Seed": seed,
            "Makespan": pareto_objs[0, 0],
            "Tardiness": pareto_objs[0, 1],
            "Instability": pareto_objs[0, 2],
            "Runtime_sec": nsga_time,
        })

        # 2. Best Stability (Min Instability)
        best_stab_idx = int(np.argmin(pareto_objs[:, 2]))
        records.append({
            "Instance": benchmark_name,
            "Algorithm": "NSGA-II (Min Instability)",
            "Phase": "Post-Disruption",
            "Seed": seed,
            "Makespan": pareto_objs[best_stab_idx, 0],
            "Tardiness": pareto_objs[best_stab_idx, 1],
            "Instability": pareto_objs[best_stab_idx, 2],
            "Runtime_sec": nsga_time,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Run for FT06
    df_ft06 = evaluate_instance_experiment(
        benchmark_name="FT06 (6x6)",
        file_path="data/benchmarks/ft06.txt",
        disruption_machine=2,
        disruption_time=20,
        disruption_duration=15,
        num_seeds=3,
        ga_pop=40,
        ga_gen=60,
        nsga_pop=50,
        nsga_gen=50,
    )

    csv_path = "results/experiment_summary_ft06.csv"
    df_ft06.to_csv(csv_path, index=False)
    print(f"\n[DONE] Experiment results saved to {csv_path}")

    # Display aggregated results
    summary = df_ft06.groupby(["Algorithm", "Phase"])[["Makespan", "Tardiness", "Instability", "Runtime_sec"]].mean().reset_index()
    print("\nAGGREGATED SUMMARY TABLE:")
    print(summary.to_string(index=False))
