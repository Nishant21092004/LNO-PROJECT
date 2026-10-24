"""
problem.py: Core data models and instance loader for Job Shop Scheduling Problems (JSSP).
Supports standard OR-Library / JSPLIB formats.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os


@dataclass
class Operation:
    job_id: int
    op_idx: int
    machine_id: int
    duration: int

    def __repr__(self):
        return f"Op(J{self.job_id}, #{self.op_idx}, M{self.machine_id}, dur={self.duration})"


@dataclass
class Job:
    job_id: int
    operations: List[Operation] = field(default_factory=list)
    due_date: int = 0
    release_time: int = 0

    @property
    def total_processing_time(self) -> int:
        return sum(op.duration for op in self.operations)


@dataclass
class JSPInstance:
    name: str
    num_jobs: int
    num_machines: int
    jobs: List[Job] = field(default_factory=list)

    @property
    def total_operations(self) -> int:
        return sum(len(job.operations) for job in self.jobs)

    def assign_due_dates(self, tightness_factor: float = 1.5, min_due_date: int = 0):
        """
        Total Work Content (TWK) rule: Due date = release_time + k * total_processing_time.
        Standard academic formula for introducing tardiness in JSSP benchmarks.
        """
        for job in self.jobs:
            raw_due = int(job.release_time + tightness_factor * job.total_processing_time)
            job.due_date = max(min_due_date, raw_due)


def load_jsp_instance(file_path: str, tightness_factor: float = 1.5) -> JSPInstance:
    """
    Parses standard JSPLib / OR-Library JSSP benchmark text files.
    Format:
      # Comments starting with #
      num_jobs num_machines
      For each job row: pairs of (machine_id, duration)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    lines = []
    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)

    if not lines:
        raise ValueError(f"No valid data lines found in {file_path}")

    # First non-comment line has num_jobs and num_machines
    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])

    jobs = []
    job_lines = lines[1 : num_jobs + 1]

    for j_id, line in enumerate(job_lines):
        tokens = [int(tok) for tok in line.split()]
        ops = []
        op_idx = 0
        for i in range(0, len(tokens), 2):
            m_id = tokens[i]
            dur = tokens[i + 1]
            ops.append(Operation(job_id=j_id, op_idx=op_idx, machine_id=m_id, duration=dur))
            op_idx += 1
        jobs.append(Job(job_id=j_id, operations=ops))

    instance = JSPInstance(
        name=instance_name,
        num_jobs=num_jobs,
        num_machines=num_machines,
        jobs=jobs,
    )
    instance.assign_due_dates(tightness_factor=tightness_factor)
    return instance
