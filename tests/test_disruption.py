"""Regression tests for the simple right-shift breakdown model."""
from src.problem import JSPInstance, Job, Operation
from src.schedule import decode_chromosome
from src.disruption import BreakdownDisruption, apply_right_shift


def test_right_shift_keeps_previously_completed_operation():
    inst = JSPInstance("toy", 1, 1, [
        Job(0, [Operation(0, 0, 0, 3), Operation(0, 1, 0, 4)], due_date=20)
    ])
    initial = decode_chromosome([0, 0], inst)
    disruption = BreakdownDisruption(machine_id=0, breakdown_start=4, duration=3)
    changed = apply_right_shift(initial, disruption, inst)
    assert changed.op_lookup[(0, 0)].start_time == 0
    assert changed.op_lookup[(0, 0)].end_time == 3
    # The second operation has already started by t=4. Under resume-after-repair
    # semantics it cannot finish earlier than its original end + downtime.
    assert changed.op_lookup[(0, 1)].end_time >= 10


def test_right_shift_delays_not_yet_started_operations():
    inst = JSPInstance("toy", 1, 1, [
        Job(0, [Operation(0, 0, 0, 3), Operation(0, 1, 0, 4)], due_date=20)
    ])
    initial = decode_chromosome([0, 0], inst)
    disruption = BreakdownDisruption(machine_id=0, breakdown_start=2, duration=5)
    shifted = apply_right_shift(initial, disruption, inst)
    assert shifted.op_lookup[(0, 1)].start_time >= 7
