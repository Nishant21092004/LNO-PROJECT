"""
run_nsga3_comparison.py: Compares NSGA-II vs NSGA-III when scaling from
3 objectives to 4 objectives (incorporating Total Energy Consumption).
Demonstrates where NSGA-III's reference-point selection excels.
"""

import sys
import time
import numpy as np
from tabulate import tabulate

from src.problem import load_jsp_instance
from src.disruption import BreakdownDisruption, partition_disrupted_state
from src.ga import run_genetic_algorithm
from src.nsga2 import run_nsga2
from src.nsga3 import run_nsga3

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    print("=" * 80)
    print(" SCALING STUDY: NSGA-II vs NSGA-III (3 OBJECTIVES vs 4 OBJECTIVES)")
    print("=" * 80)

    # 1. Load benchmark
    instance = load_jsp_instance("data/benchmarks/ft06.txt", tightness_factor=1.5)

    # 2. Initial predictive schedule
    init_sched, _, _ = run_genetic_algorithm(instance, pop_size=40, generations=60, seed=42)

    # 3. Disruption
    disruption = BreakdownDisruption(machine_id=2, breakdown_start=20, duration=15)
    completed_ops, m_ready, j_ready, j_offsets, rem_pool = partition_disrupted_state(
        init_sched, disruption, instance
    )

    print(f"\n[Problem Setup]")
    print(f"  Benchmark: FT06 (6 jobs, 6 machines, 36 ops)")
    print(f"  Disruption: Machine 2 breakdown at t=20 for 15 units")
    print(f"  Objectives 1-3: Makespan (C_max), Tardiness (sum T_j), Instability (Delta S)")
    print(f"  Objective 4   : Total Energy Consumption (TEC, in kWh)")

    # 4. NSGA-II on 3 objectives
    print("\n[1] Running NSGA-II on 3 Objectives...")
    t0 = time.time()
    scheds_nsga2_3d, objs_nsga2_3d = run_nsga2(
        instance,
        baseline_schedule=init_sched,
        pop_size=50,
        generations=60,
        machine_ready_times=m_ready,
        job_ready_times=j_ready,
        job_op_offsets=j_offsets,
        disruption_time=disruption.breakdown_start,
        seed=42,
    )
    time_nsga2_3d = time.time() - t0
    print(f"    Done in {time_nsga2_3d:.2f}s. Discovered {len(scheds_nsga2_3d)} Pareto solutions.")

    # 5. NSGA-III on 4 objectives
    print("\n[2] Running NSGA-III on 4 Objectives (Many-Objective Green Scheduling)...")
    t0 = time.time()
    scheds_nsga3_4d, objs_nsga3_4d = run_nsga3(
        instance,
        baseline_schedule=init_sched,
        pop_size=50,
        generations=60,
        num_divisions=3,
        machine_ready_times=m_ready,
        job_ready_times=j_ready,
        job_op_offsets=j_offsets,
        disruption_time=disruption.breakdown_start,
        include_energy=True,
        seed=42,
    )
    time_nsga3_4d = time.time() - t0
    print(f"    Done in {time_nsga3_4d:.2f}s. Discovered {len(scheds_nsga3_4d)} Pareto solutions across 4D space.")

    # Display comparison
    table = [
        [
            "NSGA-II (3 Objectives)",
            "C_max, sum T_j, Delta S",
            len(scheds_nsga2_3d),
            f"{int(np.min(objs_nsga2_3d[:,0]))} - {int(np.max(objs_nsga2_3d[:,0]))}",
            f"{int(np.min(objs_nsga2_3d[:,1]))} - {int(np.max(objs_nsga2_3d[:,1]))}",
            f"{int(np.min(objs_nsga2_3d[:,2]))} - {int(np.max(objs_nsga2_3d[:,2]))}",
            "-",
            f"{time_nsga2_3d:.2f} s",
        ],
        [
            "NSGA-III (4 Objectives)",
            "C_max, sum T_j, Delta S, Energy",
            len(scheds_nsga3_4d),
            f"{int(np.min(objs_nsga3_4d[:,0]))} - {int(np.max(objs_nsga3_4d[:,0]))}",
            f"{int(np.min(objs_nsga3_4d[:,1]))} - {int(np.max(objs_nsga3_4d[:,1]))}",
            f"{int(np.min(objs_nsga3_4d[:,2]))} - {int(np.max(objs_nsga3_4d[:,2]))}",
            f"{np.min(objs_nsga3_4d[:,3]):.1f} - {np.max(objs_nsga3_4d[:,3]):.1f} kWh",
            f"{time_nsga3_4d:.2f} s",
        ],
    ]

    headers = [
        "Algorithm",
        "Target Objectives",
        "Front Size",
        "Makespan Range",
        "Tardiness Range",
        "Instability Range",
        "Energy Range",
        "Runtime",
    ]

    print("\n" + "=" * 80)
    print(" SUMMARY TABLE: NSGA-II vs NSGA-III")
    print("=" * 80)
    print(tabulate(table, headers=headers, tablefmt="github"))
    print("=" * 80)

    print("\nSample Diverse Solutions from NSGA-III 4D Pareto Front:")
    sample_rows = []
    for i in range(min(5, len(scheds_nsga3_4d))):
        sample_rows.append([
            f"NSGA-III Sol #{i+1}",
            int(objs_nsga3_4d[i, 0]),
            int(objs_nsga3_4d[i, 1]),
            int(objs_nsga3_4d[i, 2]),
            f"{objs_nsga3_4d[i, 3]:.1f} kWh",
        ])
    print(tabulate(sample_rows, headers=["Solution", "Makespan", "Tardiness", "Instability", "Energy"], tablefmt="github"))


if __name__ == "__main__":
    main()
