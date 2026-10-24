"""
ga.py: Single-Objective Genetic Algorithm for Job Shop Scheduling (minimizing Makespan).
Uses Job-based Precedence Crossover (JOX/POX) and Swap Mutation.
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from src.problem import JSPInstance
from src.schedule import Schedule, decode_chromosome, create_random_chromosome


def job_based_crossover(parent1: List[int], parent2: List[int], instance: JSPInstance) -> Tuple[List[int], List[int]]:
    """
    Job-based Order Crossover (JOX/POX):
    Selects a random subset of jobs. In child 1, jobs in the subset keep their positions
    from parent 1, while other positions are filled by the remaining jobs in the order
    they appear in parent 2. This guarantees offspring feasibility.
    """
    all_jobs = list(range(instance.num_jobs))
    subset_size = random.randint(1, max(1, instance.num_jobs - 1))
    job_subset = set(random.sample(all_jobs, subset_size))

    def make_child(p1: List[int], p2: List[int]) -> List[int]:
        child = [None] * len(p1)
        # 1. Copy genes from p1 belonging to job_subset
        for i, gene in enumerate(p1):
            if gene in job_subset:
                child[i] = gene

        # 2. Fill empty slots with remaining genes in order of p2
        p2_remaining = [gene for gene in p2 if gene not in job_subset]
        p2_idx = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = p2_remaining[p2_idx]
                p2_idx += 1
        return child

    child1 = make_child(parent1, parent2)
    child2 = make_child(parent2, parent1)
    return child1, child2


def swap_mutation(chromosome: List[int], mutation_rate: float = 0.1) -> List[int]:
    """Swaps two randomly chosen genes with probability mutation_rate."""
    chrom = list(chromosome)
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(chrom)), 2)
        chrom[idx1], chrom[idx2] = chrom[idx2], chrom[idx1]
    return chrom


def tournament_selection(population: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
    """Selects the best individual among k randomly drawn candidates."""
    selected_indices = random.sample(range(len(population)), k)
    best_idx = min(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_idx]


def run_genetic_algorithm(
    instance: JSPInstance,
    pop_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.15,
    tournament_k: int = 3,
    elite_count: int = 2,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
    seed: Optional[int] = None,
) -> Tuple[Schedule, List[int], List[float]]:
    """
    Executes Single-Objective Genetic Algorithm minimizing Makespan.
    Returns:
      best_schedule: The optimal/near-optimal schedule found
      best_chromosome: Chromosome encoding
      history: Best makespan per generation
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Initialize Population
    # Construct base template multiset respecting remaining ops if offsets are provided
    base_pool: List[int] = []
    for job in instance.jobs:
        offset = job_op_offsets[job.job_id] if job_op_offsets else 0
        rem_ops = len(job.operations) - offset
        base_pool.extend([job.job_id] * rem_ops)

    population: List[List[int]] = []
    for _ in range(pop_size):
        individual = list(base_pool)
        np.random.shuffle(individual)
        population.append(individual)

    def evaluate(ind: List[int]) -> int:
        sched = decode_chromosome(
            ind,
            instance,
            machine_ready_times=machine_ready_times,
            job_ready_times=job_ready_times,
            job_op_offsets=job_op_offsets,
        )
        return sched.makespan

    fitnesses = [evaluate(ind) for ind in population]
    best_idx = int(np.argmin(fitnesses))
    best_chromosome = list(population[best_idx])
    best_makespan = fitnesses[best_idx]
    history: List[float] = [best_makespan]

    for gen in range(generations):
        # Elitism
        sorted_indices = np.argsort(fitnesses)
        new_population = [list(population[idx]) for idx in sorted_indices[:elite_count]]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses, k=tournament_k)
            p2 = tournament_selection(population, fitnesses, k=tournament_k)

            if random.random() < crossover_rate:
                c1, c2 = job_based_crossover(p1, p2, instance)
            else:
                c1, c2 = list(p1), list(p2)

            c1 = swap_mutation(c1, mutation_rate)
            c2 = swap_mutation(c2, mutation_rate)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population
        fitnesses = [evaluate(ind) for ind in population]

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_makespan:
            best_makespan = fitnesses[gen_best_idx]
            best_chromosome = list(population[gen_best_idx])

        history.append(best_makespan)

    final_schedule = decode_chromosome(
        best_chromosome,
        instance,
        machine_ready_times=machine_ready_times,
        job_ready_times=job_ready_times,
        job_op_offsets=job_op_offsets,
    )
    return final_schedule, best_chromosome, history
