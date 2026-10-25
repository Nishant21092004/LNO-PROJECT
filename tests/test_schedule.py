"""Small, hand-checkable scheduling invariants using the existing JSP model."""
from collections import Counter
import pytest
from src.problem import JSPInstance, Job, Operation
from src.schedule import decode_chromosome, create_random_chromosome
from src.ga import job_based_crossover


@pytest.fixture
def toy():
    return JSPInstance("toy", 2, 2, [
        Job(0, [Operation(0, 0, 0, 3), Operation(0, 1, 1, 2)], due_date=4),
        Job(1, [Operation(1, 0, 1, 4)], due_date=8),
    ])


def validate_schedule(schedule, instance):
    expected = {(j.job_id, i) for j in instance.jobs for i in range(len(j.operations))}
    assert {op.op_key for op in schedule.operations} == expected
    assert len(schedule.operations) == len(expected)
    for op in schedule.operations:
        source = instance.jobs[op.job_id].operations[op.op_idx]
        assert op.machine_id == source.machine_id
        assert op.duration == source.duration
        assert op.end_time - op.start_time == op.duration
        for earlier in schedule.operations:
            if earlier.op_key == (op.job_id, op.op_idx - 1):
                assert earlier.end_time <= op.start_time
    for machine in schedule.machine_schedules.values():
        for a, b in zip(machine, machine[1:]):
            assert a.end_time <= b.start_time


def test_chromosome_counts_and_crossover(toy):
    chrom = create_random_chromosome(toy)
    assert Counter(chrom) == Counter({0: 2, 1: 1})
    a, b = job_based_crossover([0, 1, 0], [1, 0, 0], toy)
    assert Counter(a) == Counter(chrom)
    assert Counter(b) == Counter(chrom)


def test_decoder_machine_and_precedence_feasibility(toy):
    for genes in ([0, 1, 0], [1, 0, 0], [0, 0, 1]):
        sched = decode_chromosome(genes, toy)
        validate_schedule(sched, toy)


def test_hand_calculated_makespan_and_tardiness(toy):
    sched = decode_chromosome([0, 1, 0], toy)
    # J0: M0 [0,3] then M1 [4,6]; J1: M1 [0,4].
    assert sched.makespan == 6
    assert sched.total_tardiness == 2
