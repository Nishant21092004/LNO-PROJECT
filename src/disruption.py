"""
disruption.py: Models machine breakdown disruptions and the Right-Shift Rescheduling baseline.
Implements the predictive-reactive rescheduling mechanics formulated by Vieira et al. (2003).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from src.problem import JSPInstance, Operation
from src.schedule import Schedule, ScheduledOperation


@dataclass
class BreakdownDisruption:
    machine_id: int
    breakdown_start: int  # t_d
    duration: int         # D

    @property
    def breakdown_end(self) -> int:
        return self.breakdown_start + self.duration

    def __repr__(self):
        return f"Breakdown(M{self.machine_id} offline from t={self.breakdown_start} to t={self.breakdown_end}, dur={self.duration})"


def apply_right_shift(
    initial_schedule: Schedule,
    disruption: BreakdownDisruption,
    instance: JSPInstance,
) -> Schedule:
    """
    Implements Right-Shift Rescheduling (the standard reactive benchmark):
    Preserves the EXACT operation order on all machines, but shifts operations
    to accommodate the machine repair window and ripple effects through job precedences.
    """
    # 1. Topological order of all operations according to the initial schedule
    # Operations are ordered by their initial start times
    sorted_ops = sorted(initial_schedule.operations, key=lambda x: (x.start_time, x.end_time))

    new_ops: List[ScheduledOperation] = []
    machine_avail = {m: 0 for m in range(instance.num_machines)}
    job_avail = {j: 0 for j in range(instance.num_jobs)}

    # When the broken machine becomes available
    # It cannot process anything between [breakdown_start, breakdown_end]
    for op in sorted_ops:
        m = op.machine_id
        j = op.job_id
        dur = op.duration

        # If operation was already finished BEFORE breakdown occurred, it remains untouched
        if op.end_time <= disruption.breakdown_start:
            start_t = op.start_time
            end_t = op.end_time
        else:
            # Operation starts at earliest feasible time respecting job precedence & machine availability
            earliest_t = max(machine_avail[m], job_avail[j])

            if m == disruption.machine_id:
                # If earliest_t falls inside or crosses the breakdown window
                if op.start_time < disruption.breakdown_start and op.end_time > disruption.breakdown_start:
                    # Interrupted operation: resumes after breakdown
                    # For non-preemptive with resume: remaining duration
                    completed_portion = disruption.breakdown_start - op.start_time
                    remaining_dur = dur - completed_portion
                    start_t = op.start_time
                    end_t = disruption.breakdown_end + remaining_dur
                elif earliest_t < disruption.breakdown_end:
                    # Starts after breakdown finishes
                    start_t = max(earliest_t, disruption.breakdown_end)
                    end_t = start_t + dur
                else:
                    start_t = earliest_t
                    end_t = start_t + dur
            else:
                start_t = earliest_t
                end_t = start_t + dur

        machine_avail[m] = end_t
        job_avail[j] = end_t

        new_ops.append(
            ScheduledOperation(
                job_id=j,
                op_idx=op.op_idx,
                machine_id=m,
                start_time=start_t,
                end_time=end_t,
                duration=dur,
            )
        )

    rescheduled = Schedule(instance=instance, operations=new_ops)
    rescheduled.finalize()
    return rescheduled


def partition_disrupted_state(
    initial_schedule: Schedule,
    disruption: BreakdownDisruption,
    instance: JSPInstance,
) -> Tuple[List[ScheduledOperation], Dict[int, int], Dict[int, int], Dict[int, int], List[int]]:
    """
    Partitions the schedule at t = breakdown_start:
    - completed_ops: frozen operations that finished before t_d
    - machine_ready_times: initial availability for remaining operations
    - job_ready_times: when jobs become ready for remaining operations
    - job_op_offsets: next operation index to schedule per job
    - remaining_chromosome_pool: list of job_ids that still need scheduling
    """
    completed_ops: List[ScheduledOperation] = []
    job_op_offsets = {j: 0 for j in range(instance.num_jobs)}
    machine_ready = {m: 0 for m in range(instance.num_machines)}
    job_ready = {j: 0 for j in range(instance.num_jobs)}

    # Machine under breakdown cannot start any new operation until breakdown_end
    machine_ready[disruption.machine_id] = disruption.breakdown_end

    for op in sorted(initial_schedule.operations, key=lambda x: x.end_time):
        if op.end_time <= disruption.breakdown_start:
            completed_ops.append(op)
            job_op_offsets[op.job_id] = max(job_op_offsets[op.job_id], op.op_idx + 1)
            machine_ready[op.machine_id] = max(machine_ready[op.machine_id], op.end_time)
            job_ready[op.job_id] = max(job_ready[op.job_id], op.end_time)

    # For any job, ready time cannot be earlier than breakdown_start if it was in progress
    for j in range(instance.num_jobs):
        job_ready[j] = max(job_ready[j], disruption.breakdown_start)

    # Remaining operations to be rescheduled
    remaining_chrom_pool: List[int] = []
    for job in instance.jobs:
        rem_count = len(job.operations) - job_op_offsets[job.job_id]
        remaining_chrom_pool.extend([job.job_id] * rem_count)

    return completed_ops, machine_ready, job_ready, job_op_offsets, remaining_chrom_pool
