"""
nsga2.py: Multi-Objective Evolutionary Algorithm (NSGA-II) for Dynamic Job Shop Scheduling.
Optimizes three conflicting objectives:
  1. Makespan (C_max)
  2. Total Tardiness (sum max(0, C_i - d_i))
  3. Schedule Instability / Nervousness (Delta S relative to baseline schedule)
Based on Deb et al. (2002).
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from src.problem import JSPInstance
from src.schedule import Schedule, decode_chromosome
from src.ga import job_based_crossover, swap_mutation


class Individual:
    def __init__(self, chromosome: List[int]):
        self.chromosome: List[int] = list(chromosome)
        self.objectives: np.ndarray = np.zeros(3)  # [Makespan, Total Tardiness, Instability]
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.schedule: Optional[Schedule] = None

    def dominates(self, other: "Individual") -> bool:
        """Pareto dominance: True if self is <= other in all objectives and < in at least one."""
        return np.all(self.objectives <= other.objectives) and np.any(self.objectives < other.objectives)


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Computes Pareto Fronts using Deb et al.'s O(MN^2) fast non-dominated sorting algorithm.
    Returns: List of fronts [F_1, F_2, ...], where F_1 is the non-dominated Pareto front.
    """
    domination_count = {id(ind): 0 for ind in population}
    dominated_solutions = {id(ind): [] for ind in population}
    fronts: List[List[Individual]] = [[]]

    for p in population:
        p_id = id(p)
        for q in population:
            q_id = id(q)
            if p.dominates(q):
                dominated_solutions[p_id].append(q)
            elif q.dominates(p):
                domination_count[p_id] += 1

        if domination_count[p_id] == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front: List[Individual] = []
        for p in fronts[i]:
            p_id = id(p)
            for q in dominated_solutions[p_id]:
                q_id = id(q)
                domination_count[q_id] -= 1
                if domination_count[q_id] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove empty trailing front
    if not fronts[-1]:
        fronts.pop()

    return fronts


def calculate_crowding_distance(front: List[Individual]):
    """Assigns crowding distances to individuals in a Pareto front."""
    l = len(front)
    if l == 0:
        return
    for ind in front:
        ind.crowding_distance = 0.0

    num_objectives = len(front[0].objectives)

    for m in range(num_objectives):
        front.sort(key=lambda ind: ind.objectives[m])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        obj_range = obj_max - obj_min

        if obj_range == 0:
            continue

        for i in range(1, l - 1):
            if front[i].crowding_distance != float("inf"):
                dist = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
                front[i].crowding_distance += dist


def crowded_tournament_selection(population: List[Individual], k: int = 2) -> Individual:
    """Crowded comparison operator: chooses lower rank; ties broken by higher crowding distance."""
    candidates = random.sample(population, k)
    best = candidates[0]
    for other in candidates[1:]:
        if other.rank < best.rank:
            best = other
        elif other.rank == best.rank and other.crowding_distance > best.crowding_distance:
            best = other
    return best


def run_nsga2(
    instance: JSPInstance,
    baseline_schedule: Optional[Schedule] = None,
    pop_size: int = 60,
    generations: int = 80,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.2,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
    disruption_time: int = 0,
    seed: Optional[int] = None,
) -> Tuple[List[Schedule], np.ndarray]:
    """
    Executes NSGA-II to find the Pareto optimal front across:
      f1: Makespan
      f2: Total Tardiness
      f3: Schedule Instability (deviation from baseline_schedule)

    Returns:
      pareto_schedules: List of schedules on the non-dominated front (Rank 0)
      pareto_objectives: 2D numpy array of shape (P, 3)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Base pool of remaining operations
    base_pool: List[int] = []
    for job in instance.jobs:
        offset = job_op_offsets[job.job_id] if job_op_offsets else 0
        rem_ops = len(job.operations) - offset
        base_pool.extend([job.job_id] * rem_ops)

    def evaluate_individual(ind: Individual):
        sched = decode_chromosome(
            ind.chromosome,
            instance,
            machine_ready_times=machine_ready_times,
            job_ready_times=job_ready_times,
            job_op_offsets=job_op_offsets,
        )
        ind.schedule = sched
        f1_makespan = sched.makespan
        f2_tardiness = sched.total_tardiness
        f3_instability = sched.compute_instability(baseline_schedule, impacted_only_since=disruption_time)
        ind.objectives = np.array([f1_makespan, f2_tardiness, f3_instability], dtype=float)

    # 2. Initial Population
    population: List[Individual] = []
    for _ in range(pop_size):
        chrom = list(base_pool)
        np.random.shuffle(chrom)
        ind = Individual(chrom)
        evaluate_individual(ind)
        population.append(ind)

    # Initial sorting and crowding
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        calculate_crowding_distance(front)

    # 3. Generational Loop
    for gen in range(generations):
        # Generate offspring Qt (size pop_size)
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = crowded_tournament_selection(population)
            p2 = crowded_tournament_selection(population)

            if random.random() < crossover_rate:
                c1_chrom, c2_chrom = job_based_crossover(p1.chromosome, p2.chromosome, instance)
            else:
                c1_chrom, c2_chrom = list(p1.chromosome), list(p2.chromosome)

            c1_chrom = swap_mutation(c1_chrom, mutation_rate)
            c2_chrom = swap_mutation(c2_chrom, mutation_rate)

            ind1 = Individual(c1_chrom)
            evaluate_individual(ind1)
            offspring.append(ind1)

            if len(offspring) < pop_size:
                ind2 = Individual(c2_chrom)
                evaluate_individual(ind2)
                offspring.append(ind2)

        # Combine parent + offspring: R_t = P_t U Q_t (size 2 * pop_size)
        combined_pool = population + offspring
        fronts = fast_non_dominated_sort(combined_pool)

        # Select next generation P_{t+1} of size pop_size
        new_population: List[Individual] = []
        for front in fronts:
            calculate_crowding_distance(front)
            if len(new_population) + len(front) <= pop_size:
                new_population.extend(front)
            else:
                # Front must be partially selected based on crowding distance
                remaining_slots = pop_size - len(new_population)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                new_population.extend(front[:remaining_slots])
                break

        population = new_population

    # Extract final Rank 0 Pareto front
    final_fronts = fast_non_dominated_sort(population)
    pareto_front = final_fronts[0]

    # Filter distinct objective trade-offs
    unique_pareto: List[Individual] = []
    seen = set()
    for ind in pareto_front:
        key = tuple(ind.objectives)
        if key not in seen:
            seen.add(key)
            unique_pareto.append(ind)

    unique_pareto.sort(key=lambda ind: ind.objectives[0])  # Sort by makespan
    pareto_schedules = [ind.schedule for ind in unique_pareto]
    pareto_objs = np.array([ind.objectives for ind in unique_pareto])

    return pareto_schedules, pareto_objs
