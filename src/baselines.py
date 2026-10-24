"""
baselines.py: Classical Priority Dispatching Rules (FCFS and SPT) for Job Shop Scheduling.
These provide established non-evolutionary baselines for production scheduling.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from src.problem import JSPInstance, Operation
from src.schedule import Schedule, ScheduledOperation


def solve_fcfs(
    instance: JSPInstance,
    initial_machine_avail: Optional[Dict[int, int]] = None,
    initial_job_avail: Optional[Dict[int, int]] = None,
    initial_op_ptr: Optional[Dict[int, int]] = None,
) -> Schedule:
    """
    First-Come, First-Served (FCFS) Dispatching Rule:
    At any dispatching decision point, the operation whose preceding step
    finished earliest (i.e. entered the machine queue first) is scheduled next.
    """
    m_avail = {m: 0 for m in range(instance.num_machines)}
    if initial_machine_avail:
        m_avail.update(initial_machine_avail)

    j_avail = {j: 0 for j in range(instance.num_jobs)}
    if initial_job_avail:
        j_avail.update(initial_job_avail)

    op_ptr = {j: 0 for j in range(instance.num_jobs)}
    if initial_op_ptr:
        op_ptr.update(initial_op_ptr)

    scheduled_ops: List[ScheduledOperation] = []
    total_ops = instance.total_operations - sum(op_ptr.values())

    while len(scheduled_ops) < total_ops:
        # Find all operations that are currently ready (their previous op is complete)
        candidate_ops: List[Tuple[int, Operation, int]] = []  # (job_id, Operation, arrival_time)
        for j_id in range(instance.num_jobs):
            idx = op_ptr[j_id]
            if idx < len(instance.jobs[j_id].operations):
                op = instance.jobs[j_id].operations[idx]
                arrival_time = j_avail[j_id]
                candidate_ops.append((j_id, op, arrival_time))

        if not candidate_ops:
            break

        # In FCFS, sort candidate operations primarily by arrival time at machine queue
        candidate_ops.sort(key=lambda item: item[2])
        chosen_job_id, chosen_op, _ = candidate_ops[0]

        m_id = chosen_op.machine_id
        start_time = max(m_avail[m_id], j_avail[chosen_job_id])
        end_time = start_time + chosen_op.duration

        m_avail[m_id] = end_time
        j_avail[chosen_job_id] = end_time
        op_ptr[chosen_job_id] += 1

        scheduled_ops.append(
            ScheduledOperation(
                job_id=chosen_job_id,
                op_idx=chosen_op.op_idx,
                machine_id=m_id,
                start_time=start_time,
                end_time=end_time,
                duration=chosen_op.duration,
            )
        )

    sched = Schedule(instance=instance, operations=scheduled_ops)
    sched.finalize()
    return sched


def solve_spt(
    instance: JSPInstance,
    initial_machine_avail: Optional[Dict[int, int]] = None,
    initial_job_avail: Optional[Dict[int, int]] = None,
    initial_op_ptr: Optional[Dict[int, int]] = None,
) -> Schedule:
    """
    Shortest Processing Time (SPT) Dispatching Rule:
    At any dispatching decision point, the operation with the shortest
    duration among ready candidate jobs is prioritized.
    """
    m_avail = {m: 0 for m in range(instance.num_machines)}
    if initial_machine_avail:
        m_avail.update(initial_machine_avail)

    j_avail = {j: 0 for j in range(instance.num_jobs)}
    if initial_job_avail:
        j_avail.update(initial_job_avail)

    op_ptr = {j: 0 for j in range(instance.num_jobs)}
    if initial_op_ptr:
        op_ptr.update(initial_op_ptr)

    scheduled_ops: List[ScheduledOperation] = []
    total_ops = instance.total_operations - sum(op_ptr.values())

    while len(scheduled_ops) < total_ops:
        candidate_ops: List[Tuple[int, Operation, int]] = []
        for j_id in range(instance.num_jobs):
            idx = op_ptr[j_id]
            if idx < len(instance.jobs[j_id].operations):
                op = instance.jobs[j_id].operations[idx]
                candidate_ops.append((j_id, op, op.duration))

        if not candidate_ops:
            break

        # SPT prioritizes minimal duration; tie-break with job ID
        candidate_ops.sort(key=lambda item: (item[2], item[0]))
        chosen_job_id, chosen_op, _ = candidate_ops[0]

        m_id = chosen_op.machine_id
        start_time = max(m_avail[m_id], j_avail[chosen_job_id])
        end_time = start_time + chosen_op.duration

        m_avail[m_id] = end_time
        j_avail[chosen_job_id] = end_time
        op_ptr[chosen_job_id] += 1

        scheduled_ops.append(
            ScheduledOperation(
                job_id=chosen_job_id,
                op_idx=chosen_op.op_idx,
                machine_id=m_id,
                start_time=start_time,
                end_time=end_time,
                duration=chosen_op.duration,
            )
        )

    sched = Schedule(instance=instance, operations=scheduled_ops)
    sched.finalize()
    return sched
