"""
run_demo.py: End-to-end prototype running all algorithms on FT06 with machine disruption.
Demonstrates:
  1. Classical Baselines (FCFS, SPT)
  2. Single-Objective GA (Predictive Schedule)
  3. Machine Breakdown Disruption Injection
  4. Right-Shift Rescheduling Baseline
  5. NSGA-II Multi-Objective Rescheduling (Pareto Front)
  6. Automatic Generation of Gantt Charts and Pareto Front Plots
"""

import os
import time
import numpy as np
from tabulate import tabulate

from src.problem import load_jsp_instance
from src.baselines import solve_fcfs, solve_spt
from src.ga import run_genetic_algorithm
from src.disruption import BreakdownDisruption, apply_right_shift, partition_disrupted_state
from src.nsga2 import run_nsga2
from src.visualize import plot_gantt_chart, plot_pareto_front


def main():
    print("=" * 75)
    print(" DYNAMIC JOB SHOP SCHEDULING UNDER MACHINE DISRUPTIONS: PROTOTYPE DEMO")
    print("=" * 75)

    # 1. Load Benchmark Instance (FT06 - 6 jobs, 6 machines, 36 ops)
    benchmark_path = "data/benchmarks/ft06.txt"
    print(f"\n[1] Loading Benchmark: {benchmark_path} ...")
    instance = load_jsp_instance(benchmark_path, tightness_factor=1.5)
    print(f"    - Jobs: {instance.num_jobs}, Machines: {instance.num_machines}, Ops: {instance.total_operations}")
    for j in instance.jobs:
        print(f"      Job {j.job_id}: Processing={j.total_processing_time}, Due Date={j.due_date}")

    # 2. Evaluate Classical Baselines (FCFS & SPT) on Initial State
    print("\n[2] Evaluating Classical Dispatching Heuristics (Pre-Breakdown)...")
    t0 = time.time()
    sched_fcfs = solve_fcfs(instance)
    fcfs_time = time.time() - t0

    t0 = time.time()
    sched_spt = solve_spt(instance)
    spt_time = time.time() - t0

    print(f"    - FCFS: Makespan = {sched_fcfs.makespan}, Tardiness = {sched_fcfs.total_tardiness} (Runtime: {fcfs_time*1000:.2f}ms)")
    print(f"    - SPT : Makespan = {sched_spt.makespan}, Tardiness = {sched_spt.total_tardiness} (Runtime: {spt_time*1000:.2f}ms)")

    # 3. Single-Objective GA: Produce Initial Predictive Schedule
    print("\n[3] Running Single-Objective GA (Optimizing Makespan)...")
    t0 = time.time()
    initial_ga_sched, best_chrom, ga_hist = run_genetic_algorithm(
        instance,
        pop_size=50,
        generations=80,
        crossover_rate=0.85,
        mutation_rate=0.15,
        seed=42,
    )
    ga_time = time.time() - t0
    print(f"    - GA Initial: Makespan = {initial_ga_sched.makespan}, Tardiness = {initial_ga_sched.total_tardiness} (Runtime: {ga_time:.2f}s)")

    # Save Gantt chart for initial schedule
    os.makedirs("results", exist_ok=True)
    plot_gantt_chart(
        initial_ga_sched,
        title="Initial Predictive Schedule (Single-Objective GA)",
        save_path="results/01_initial_ga_schedule.png",
    )
    print("    -> Saved plot: results/01_initial_ga_schedule.png")

    # 4. Inject Machine Disruption
    # Breakdown occurs on Machine 2 at t=20 for duration D=15 (offline from t=20 to 35)
    disruption = BreakdownDisruption(machine_id=2, breakdown_start=20, duration=15)
    print(f"\n[4] Injecting Disruption: {disruption}")

    # 5. Reactive Baseline: Right-Shift Rescheduling
    print("\n[5] Computing Right-Shift Rescheduling Baseline...")
    t0 = time.time()
    sched_right_shift = apply_right_shift(initial_ga_sched, disruption, instance)
    rs_time = time.time() - t0
    rs_instability = sched_right_shift.compute_instability(initial_ga_sched, impacted_only_since=disruption.breakdown_start)
    print(f"    - Right-Shift: Makespan = {sched_right_shift.makespan}, Tardiness = {sched_right_shift.total_tardiness}, Instability = {rs_instability}")

    plot_gantt_chart(
        sched_right_shift,
        disruption=disruption,
        title="Rescheduled: Right-Shift Baseline",
        save_path="results/02_right_shift_schedule.png",
    )
    print("    -> Saved plot: results/02_right_shift_schedule.png")

    # 6. Partition Disrupted State for Re-Optimization
    completed_ops, m_ready, j_ready, j_offsets, rem_pool = partition_disrupted_state(
        initial_ga_sched, disruption, instance
    )
    print(f"\n[6] Rescheduling State Partitioned at t={disruption.breakdown_start}:")
    print(f"    - Completed Ops Frozen: {len(completed_ops)}")
    print(f"    - Remaining Ops to Reschedule: {len(rem_pool)}")
    print(f"    - Machine Ready Times: {m_ready}")

    # 7. Run NSGA-II Multi-Objective Rescheduling
    print("\n[7] Running NSGA-II (Objectives: Makespan, Tardiness, Instability)...")
    t0 = time.time()
    pareto_schedules, pareto_objs = run_nsga2(
        instance,
        baseline_schedule=initial_ga_sched,
        pop_size=60,
        generations=70,
        crossover_rate=0.85,
        mutation_rate=0.20,
        machine_ready_times=m_ready,
        job_ready_times=j_ready,
        job_op_offsets=j_offsets,
        disruption_time=disruption.breakdown_start,
        seed=42,
    )
    nsga2_time = time.time() - t0
    print(f"    - NSGA-II Finished in {nsga2_time:.2f}s. Discovered {len(pareto_schedules)} distinct Pareto trade-off solutions.")

    # 8. Plot Pareto Trade-Off Front
    baseline_pts = {
        "Right-Shift": (sched_right_shift.makespan, sched_right_shift.total_tardiness, rs_instability),
    }
    plot_pareto_front(
        pareto_objs,
        baseline_points=baseline_pts,
        save_path="results/03_pareto_frontier.png",
        title="Dynamic Rescheduling Pareto Frontier (NSGA-II) vs Right-Shift",
    )
    print("    -> Saved plot: results/03_pareto_frontier.png")

    # Save Gantt charts for extreme Pareto solutions
    # Solution A: Best Makespan on front
    best_ms_sched = pareto_schedules[0]
    plot_gantt_chart(
        best_ms_sched,
        disruption=disruption,
        title=f"NSGA-II Solution (Min Makespan = {best_ms_sched.makespan})",
        save_path="results/04_nsga2_min_makespan.png",
    )
    print("    -> Saved plot: results/04_nsga2_min_makespan.png")

    # Solution B: Best Stability (Lowest Instability) on front
    best_stab_idx = int(np.argmin(pareto_objs[:, 2]))
    best_stab_sched = pareto_schedules[best_stab_idx]
    plot_gantt_chart(
        best_stab_sched,
        disruption=disruption,
        title=f"NSGA-II Solution (Min Instability = {int(pareto_objs[best_stab_idx, 2])})",
        save_path="results/05_nsga2_min_instability.png",
    )
    print("    -> Saved plot: results/05_nsga2_min_instability.png")

    # 9. Comparative Summary Table
    print("\n" + "=" * 75)
    print(" FINAL COMPARATIVE SUMMARY")
    print("=" * 75)

    table_data = [
        ["FCFS (Pre-Disruption)", sched_fcfs.makespan, sched_fcfs.total_tardiness, "-", f"{fcfs_time*1000:.2f} ms"],
        ["SPT (Pre-Disruption)", sched_spt.makespan, sched_spt.total_tardiness, "-", f"{spt_time*1000:.2f} ms"],
        ["Single-Obj GA (Initial)", initial_ga_sched.makespan, initial_ga_sched.total_tardiness, "-", f"{ga_time:.2f} s"],
        ["Right-Shift (Post-Disruption)", sched_right_shift.makespan, sched_right_shift.total_tardiness, rs_instability, f"{rs_time*1000:.2f} ms"],
    ]

    for i, sol in enumerate(pareto_schedules):
        ms = int(pareto_objs[i, 0])
        tar = int(pareto_objs[i, 1])
        instab = int(pareto_objs[i, 2])
        label = f"NSGA-II Pareto Sol #{i+1}"
        if i == 0:
            label += " (Min Makespan)"
        elif i == best_stab_idx:
            label += " (Min Instability)"
        table_data.append([label, ms, tar, instab, f"{nsga2_time:.2f} s (front)"])

    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    headers = ["Algorithm / Solution", "Makespan (C_max)", "Total Tardiness (sum T_j)", "Instability (Delta S)", "Runtime"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 75)
    print("\nPrototype verification complete! All results and Gantt charts generated in `results/`.")


if __name__ == "__main__":
    main()
