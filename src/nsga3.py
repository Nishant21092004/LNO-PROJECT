"""
nsga3.py: Many-Objective Evolutionary Algorithm (NSGA-III) for Dynamic Job Shop Scheduling.
Specifically designed for 4 or more objectives to overcome the curse of dimensionality
where NSGA-II's crowding distance operator degrades.
Based on Deb & Jain (2014), IEEE TEVC.
Optimizes:
  f1: Makespan (C_max)
  f2: Total Tardiness (sum max(0, C_i - d_i))
  f3: Schedule Instability (Delta S)
  f4: Total Energy Consumption (TEC / Green scheduling)
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from src.problem import JSPInstance
from src.schedule import Schedule, decode_chromosome
from src.ga import job_based_crossover, swap_mutation
from src.nsga2 import Individual, fast_non_dominated_sort


def generate_reference_points(num_objectives: int, num_divisions: int) -> np.ndarray:
    """
    Generates uniformly spaced reference points on a normalized unit simplex
    using Das and Dennis's systematic approach (sum(x_i) = 1).
    Number of points H = (M + p - 1) choose p.
    """
    def _generate_recursive(dim: int, depth: int, left: int, total: int):
        if depth == dim - 1:
            return [[left / total]]
        res = []
        for i in range(left + 1):
            for suffix in _generate_recursive(dim, depth + 1, left - i, total):
                res.append([i / total] + suffix)
        return res

    points = _generate_recursive(num_objectives, 0, num_divisions, num_divisions)
    return np.array(points, dtype=float)


def associate_to_reference_points(
    population: List[Individual],
    reference_points: np.ndarray,
    ideal_point: np.ndarray,
    nadir_point: np.ndarray,
) -> Tuple[List[int], List[float]]:
    """
    Normalizes objectives and associates each individual with its nearest reference ray.
    Returns:
      associated_rp_idx: Index of closest reference point for each individual
      perpendicular_distances: Distance to that reference ray
    """
    ranges = nadir_point - ideal_point
    ranges[ranges == 0] = 1.0  # Avoid division by zero

    norm_objs = (np.array([ind.objectives for ind in population]) - ideal_point) / ranges

    associated_rp: List[int] = []
    distances: List[float] = []

    for obj in norm_objs:
        # Distance from point to reference rays
        # For each reference point r, ray is line from origin through r:
        # proj = (obj . r) / (||r||^2) * r
        # d_perp = || obj - proj ||
        best_rp = 0
        min_dist = float("inf")

        for r_idx, r in enumerate(reference_points):
            r_norm_sq = np.dot(r, r)
            if r_norm_sq == 0:
                continue
            proj_scale = np.dot(obj, r) / r_norm_sq
            proj = proj_scale * r
            dist = float(np.linalg.norm(obj - proj))
            if dist < min_dist:
                min_dist = dist
                best_rp = r_idx

        associated_rp.append(best_rp)
        distances.append(min_dist)

    return associated_rp, distances


def run_nsga3(
    instance: JSPInstance,
    baseline_schedule: Optional[Schedule] = None,
    pop_size: int = 60,
    generations: int = 70,
    num_divisions: int = 3,  # For 4 objectives, H = binom(4+3-1, 3) = binom(6,3) = 20 ref points
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.20,
    machine_ready_times: Optional[Dict[int, int]] = None,
    job_ready_times: Optional[Dict[int, int]] = None,
    job_op_offsets: Optional[Dict[int, int]] = None,
    disruption_time: int = 0,
    include_energy: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[Schedule], np.ndarray]:
    """
    Runs NSGA-III for Many-Objective Scheduling (3 or 4 objectives).
    If include_energy is True:
      f1: Makespan
      f2: Total Tardiness
      f3: Schedule Instability
      f4: Total Energy Consumption (TEC)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_objectives = 4 if include_energy else 3
    ref_points = generate_reference_points(num_objectives, num_divisions)

    # Base multiset of remaining ops
    base_pool: List[int] = []
    for job in instance.jobs:
        offset = job_op_offsets[job.job_id] if job_op_offsets else 0
        rem_ops = len(job.operations) - offset
        base_pool.extend([job.job_id] * rem_ops)

    def evaluate_ind(ind: Individual):
        sched = decode_chromosome(
            ind.chromosome,
            instance,
            machine_ready_times=machine_ready_times,
            job_ready_times=job_ready_times,
            job_op_offsets=job_op_offsets,
        )
        ind.schedule = sched
        f1 = sched.makespan
        f2 = sched.total_tardiness
        f3 = sched.compute_instability(baseline_schedule, impacted_only_since=disruption_time)
        if include_energy:
            f4 = sched.total_energy_consumption
            ind.objectives = np.array([f1, f2, f3, f4], dtype=float)
        else:
            ind.objectives = np.array([f1, f2, f3], dtype=float)

    # Initial Population
    population: List[Individual] = []
    for _ in range(pop_size):
        chrom = list(base_pool)
        np.random.shuffle(chrom)
        ind = Individual(chrom)
        evaluate_ind(ind)
        population.append(ind)

    # Generational Loop
    for gen in range(generations):
        # Generate offspring
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = random.choice(population)
            p2 = random.choice(population)

            if random.random() < crossover_rate:
                c1_chrom, c2_chrom = job_based_crossover(p1.chromosome, p2.chromosome, instance)
            else:
                c1_chrom, c2_chrom = list(p1.chromosome), list(p2.chromosome)

            c1_chrom = swap_mutation(c1_chrom, mutation_rate)
            c2_chrom = swap_mutation(c2_chrom, mutation_rate)

            ind1 = Individual(c1_chrom)
            evaluate_ind(ind1)
            offspring.append(ind1)
            if len(offspring) < pop_size:
                ind2 = Individual(c2_chrom)
                evaluate_ind(ind2)
                offspring.append(ind2)

        # Combine R_t = P_t U Q_t
        combined_pool = population + offspring
        fronts = fast_non_dominated_sort(combined_pool)

        # Select fronts until capacity exceeded
        selected_pool: List[Individual] = []
        last_front_idx = 0
        for f_idx, front in enumerate(fronts):
            if len(selected_pool) + len(front) <= pop_size:
                selected_pool.extend(front)
            else:
                last_front_idx = f_idx
                break

        if len(selected_pool) == pop_size:
            population = selected_pool
            continue

        # Niche-preservation selection on the splitting front F_l
        front_l = fronts[last_front_idx]
        needed_slots = pop_size - len(selected_pool)

        # Compute ideal and approximate nadir points from combined pool
        all_objs = np.array([ind.objectives for ind in combined_pool])
        ideal_pt = np.min(all_objs, axis=0)
        nadir_pt = np.max(all_objs, axis=0)

        # Associate selected_pool + front_l with reference points
        eval_subset = selected_pool + front_l
        assoc_rp, dists = associate_to_reference_points(eval_subset, ref_points, ideal_pt, nadir_pt)

        # Count niche occupancy for already selected individuals
        niche_counts = np.zeros(len(ref_points), dtype=int)
        for i in range(len(selected_pool)):
            niche_counts[assoc_rp[i]] += 1

        # Candidates from front_l
        front_l_candidates = []
        for i in range(len(front_l)):
            idx_in_eval = len(selected_pool) + i
            front_l_candidates.append({
                "ind": front_l[i],
                "rp": assoc_rp[idx_in_eval],
                "dist": dists[idx_in_eval],
            })

        # Iteratively pick candidates from front_l based on lowest niche count
        while needed_slots > 0 and front_l_candidates:
            # Find reference points associated with at least one remaining candidate
            active_rps = set(c["rp"] for c in front_l_candidates)
            min_niche = min(niche_counts[rp] for rp in active_rps)
            candidate_rps = [rp for rp in active_rps if niche_counts[rp] == min_niche]

            chosen_rp = random.choice(candidate_rps)

            # Find candidates associated with chosen_rp
            matching = [c for c in front_l_candidates if c["rp"] == chosen_rp]
            if niche_counts[chosen_rp] == 0:
                # Pick candidate with minimum perpendicular distance
                best_cand = min(matching, key=lambda c: c["dist"])
            else:
                # Pick random
                best_cand = random.choice(matching)

            selected_pool.append(best_cand["ind"])
            niche_counts[chosen_rp] += 1
            front_l_candidates.remove(best_cand)
            needed_slots -= 1

        population = selected_pool

    # Extract final Rank 0 front
    final_fronts = fast_non_dominated_sort(population)
    pareto_front = final_fronts[0]

    unique_pareto: List[Individual] = []
    seen = set()
    for ind in pareto_front:
        key = tuple(ind.objectives)
        if key not in seen:
            seen.add(key)
            unique_pareto.append(ind)

    unique_pareto.sort(key=lambda ind: ind.objectives[0])
    pareto_scheds = [ind.schedule for ind in unique_pareto]
    pareto_objs = np.array([ind.objectives for ind in unique_pareto])

    return pareto_scheds, pareto_objs
