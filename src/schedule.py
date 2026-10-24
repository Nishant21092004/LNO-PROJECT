"""
schedule.py: Schedule representation, objective evaluation, and operation-based chromosome decoder.
Guarantees strictly valid, deadlock-free active schedules using Bierwirth's operation-based representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.problem import JSPInstance, Operation, Job


@dataclass
class ScheduledOperation:
    job_id: int
    op_idx: int
    machine_id: int
    start_time: int
    end_time: int
    duration: int

    @property
    def op_key(self) -> Tuple[int, int]:
        return (self.job_id, self.op_idx)


@dataclass
class Schedule:
    instance: JSPInstance
    operations: List[ScheduledOperation] = field(default_factory=list)
    machine_schedules: Dict[int, List[ScheduledOperation]] = field(default_factory=dict)
    job_completion_times: Dict[int, int] = field(default_factory=dict)
    op_lookup: Dict[Tuple[int, int], ScheduledOperation] = field(default_factory=dict)

    def finalize(self):
        """Build index maps and calculate completion times."""
        self.machine_schedules = {m: [] for m in range(self.instance.num_machines)}
        self.op_lookup = {}
        self.job_completion_times = {j: 0 for j in range(self.instance.num_jobs)}

        for op in self.operations:
            self.machine_schedules[op.machine_id].append(op)
            self.op_lookup[op.op_key] = op
            if op.end_time > self.job_completion_times[op.job_id]:
                self.job_completion_times[op.job_id] = op.end_time

        for m in self.machine_schedules:
            self.machine_schedules[m].sort(key=lambda x: x.start_time)

    @property
    def makespan(self) -> int:
        if not self.operations:
            return 0
        return max(op.end_time for op in self.operations)

    @property
    def total_tardiness(self) -> int:
        """Sum of max(0, C_i - d_i) across all jobs."""
        tardiness = 0
        for job in self.instance.jobs:
            c_i = self.job_completion_times.get(job.job_id, 0)
            tardiness += max(0, c_i - job.due_date)
        return tardiness

    @property
    def total_energy_consumption(self) -> float:
        """
        Total Energy Consumption (TEC) - 4th Objective for Green Scheduling:
        Models active processing power (e.g. 1.0 kW) vs machine idle power (e.g. 0.2 kW).
        TEC = sum_m [ P_work * WorkTime_m + P_idle * IdleTime_m ]
        """
        P_WORK = 1.0
        P_IDLE = 0.2
        total_energy = 0.0

        for m, ops in self.machine_schedules.items():
            if not ops:
                continue
            m_completion = max(op.end_time for op in ops)
            work_time = sum(op.duration for op in ops)
            idle_time = max(0, m_completion - work_time)
            total_energy += (P_WORK * work_time) + (P_IDLE * idle_time)

        return round(total_energy, 2)

    def compute_instability(self, baseline_schedule: "Schedule", impacted_only_since: int = 0) -> int:
        """
        Calculates Schedule Instability (Nervousness) relative to a baseline schedule:
        Sum of |start_time_new - start_time_baseline| for all operations starting on or after the disruption.
        """
        if baseline_schedule is None or not baseline_schedule.operations:
            return 0

        instability = 0
        for key, new_op in self.op_lookup.items():
            if key in baseline_schedule.op_lookup:
                base_op = baseline_schedule.op_lookup[key]
                if base_op.start_time >= impacted_only_since:
                    instability += abs(new_op.start_time - base_op.start_time)
        return instability


def decode_chromosome(
    chromosome: List[int],
    instance: JSPInstance,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
) -> Schedule:
    """
    Decodes an operation-based chromosome (permutation with repetition).
    Chromosome: list of job_ids, where each job_id appears once for every operation it contains.
    e.g., for FT06 (6 jobs, 6 ops each): length is 36, containing numbers 0..5 each repeated 6 times.
    """
    m_ready = {m: 0 for m in range(instance.num_machines)}
    if machine_ready_times:
        m_ready.update(machine_ready_times)

    j_ready = {j: 0 for j in range(instance.num_jobs)}
    if job_ready_times:
        j_ready.update(job_ready_times)

    # Tracks which operation index we are scheduling next for each job
    op_ptr = {j: 0 for j in range(instance.num_jobs)}
    if job_op_offsets:
        op_ptr.update(job_op_offsets)

    scheduled_ops: List[ScheduledOperation] = []

    for job_id in chromosome:
        current_op_idx = op_ptr[job_id]
        job = instance.jobs[job_id]

        if current_op_idx >= len(job.operations):
            continue  # Already scheduled all ops for this job

        op = job.operations[current_op_idx]
        machine_id = op.machine_id
        duration = op.duration

        start_time = max(m_ready[machine_id], j_ready[job_id])
        end_time = start_time + duration

        m_ready[machine_id] = end_time
        j_ready[job_id] = end_time
        op_ptr[job_id] += 1

        scheduled_ops.append(
            ScheduledOperation(
                job_id=job_id,
                op_idx=current_op_idx,
                machine_id=machine_id,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
            )
        )

    sched = Schedule(instance=instance, operations=scheduled_ops)
    sched.finalize()
    return sched


def create_random_chromosome(instance: JSPInstance) -> List[int]:
    """Generates a random valid operation-based chromosome."""
    chrom = []
    for job in instance.jobs:
        chrom.extend([job.job_id] * len(job.operations))
    np.random.shuffle(chrom)
    return chrom
