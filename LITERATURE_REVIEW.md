# Literature Review & Benchmark Dataset Selection
**Project Title**: *Comparative Evaluation of Classical and Evolutionary Algorithms for Dynamic Production Scheduling Under Machine Disruptions*

---

## Part 1: Comprehensive Literature Review

Dynamic Job Shop Scheduling (DJSS) under unforeseen machine disruptions is a cornerstone topic in industrial engineering, production planning, and operations research. Below is a structured, plain-English breakdown of the foundational and state-of-the-art research papers directly supporting our study.

---

### 1. The Definitive Rescheduling Framework
* **Paper**: *Rescheduling manufacturing systems: a framework of strategies, policies, and methods*
* **Authors**: G. E. Vieira, J. W. Herrmann, & E. Lin (2003)
* **Venue**: *Journal of Scheduling*, 6(1), 39–62.
* **In Simple Words**:
  In a real factory, an initial schedule rarely survives the day because machines break down or emergency orders arrive. This paper provided the universal vocabulary used by all researchers today. It categorizes how factories respond into three strategies:
  1. **Dynamic / Completely Reactive**: No schedule is planned in advance; decisions are made on-the-fly using priority rules (like FCFS or SPT).
  2. **Predictive-Reactive**: An initial good schedule is built (predictive). When a breakdown hits, a rescheduling algorithm updates it (reactive).
  3. **Robust**: Building slack into the schedule to absorb minor shocks.
* **Key Concept for Our Work**:
  Vieira et al. established that rescheduling is not just about making jobs finish fast; it is about **schedule instability (or nervousness)**. If a computer reshuffles every worker's job order after a 10-minute machine stall, the shop floor descends into chaos. Minimizing schedule changes ($\Delta S$) is as important as minimizing total time.

---

### 2. The Algorithmic Foundation for Multi-Objective Optimization
* **Paper**: *A fast and elitist multiobjective genetic algorithm: NSGA-II*
* **Authors**: K. Deb, A. Pratap, S. Agarwal, & T. Meyarivan (2002)
* **Venue**: *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
* **In Simple Words**:
  When you have multiple conflicting goals (e.g., getting jobs done quickly vs. keeping customer delivery dates vs. minimizing schedule changes), there is no single "best" schedule. Improving one hurts another.
  Deb et al. introduced **NSGA-II**, which revolutionized multi-objective optimization through three innovations:
  1. **Fast Non-Dominated Sorting**: Groups schedules into Pareto "ranks" ($O(MN^2)$ complexity) so superior trade-offs dominate inferior ones.
  2. **Elitism**: Guarantees that the best trade-off schedules discovered are never lost across generations.
  3. **Crowding Distance (Niching)**: Ensures diversity along the trade-off curve without requiring manual penalty parameters.
* **Relevance to Our Work**:
  NSGA-II is our core proposed method. It allows us to produce a complete **Pareto Front** of schedules, giving plant managers the freedom to pick between an "ultra-fast but disruptive" schedule or a "mildly delayed but highly stable" schedule.

---

### 3. Survey of Dynamic Scheduling in Manufacturing
* **Paper**: *A survey of dynamic scheduling in manufacturing systems*
* **Authors**: D. Ouelhadj & S. Petrovic (2009)
* **Venue**: *Journal of Scheduling*, 12(4), 417–431.
* **In Simple Words**:
  This survey examined 15+ years of research on dynamic scheduling. It compared traditional Priority Dispatching Rules (PDRs) like FCFS and SPT against evolutionary metaheuristics like Genetic Algorithms (GA).
* **Key Findings**:
  - **PDRs (FCFS, SPT)** are instantaneous ($O(1)$ or $O(N)$) and computationally lightweight, making them popular in real-time controllers. However, their greedy, short-sighted nature yields poor makespans and high tardiness in complex multi-machine networks.
  - **Single-Objective GAs** achieve significantly tighter makespans than PDRs, but struggle when real-world constraints require balancing on-time delivery and disruption handling.
  - Multi-objective evolutionary algorithms (MOEAs) bridge this gap by treating disruptions dynamically rather than restarting the entire shop from scratch.

---

### 4. Dynamic Job Shop Scheduling with Machine Breakdowns
* **Paper**: *A multi-objective genetic algorithm for dynamic flexible job-shop scheduling with machine breakdowns*
* **Authors**: W. Xiong, D. Chen, L. Yang, & C. Shen (2013)
* **Venue**: *International Journal of Production Research*, 51(8), 2465–2481.
* **In Simple Words**:
  This study modeled a shop floor where machines experience random failures based on Mean Time Between Failures (MTBF) and Mean Time to Repair (MTTR).
  When a machine broke down, they evaluated two rescheduling methods:
  1. **Right-Shift Rescheduling**: Simply postponing tasks on the broken machine without changing their sequence.
  2. **Multi-Objective Rescheduling via NSGA-II**: Re-optimizing remaining operations to balance makespan and instability.
* **Key Findings**:
  Right-shift rescheduling is safe and produces zero sequence instability, but it severely inflates makespan and job tardiness. NSGA-II discovered trade-off schedules that drastically cut delay with only modest, controlled changes to the schedule.

---

### 5. Classical Priority Dispatching Rules: Grounding the Baselines
* **Paper**: *A survey of priority dispatching rules*
* **Authors**: S. S. Panwalkar & W. Iskander (1977) / R. Haupt (1989)
* **Venues**: *Operations Research*, 25(1); *International Journal of Production Research*, 27(1).
* **In Simple Words**:
  These seminal works classified over 100 dispatching rules. They proved mathematically and empirically that:
  - **SPT (Shortest Processing Time)**: Proven to minimize mean flow time and work-in-progress inventory across standard queues. It is the gold standard benchmark heuristic.
  - **FCFS (First-Come, First-Served)**: The most intuitive baseline ensuring operational fairness, but typically produces high makespan variance.
* **Relevance to Our Work**:
  Including FCFS and SPT gives our research solid, universally recognized baseline anchors.

---

### Summary Matrix of Related Work

| Paper | Problem Context | Algorithms Compared | Objectives Evaluated | Rescheduling Method | Core Takeaway for Our Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vieira et al. (2003)** | Dynamic Manufacturing Systems | General Framework | Efficiency vs. Instability | Right-shift, Partial, Full | Establishes Schedule Instability ($\Delta S$) as an essential metric. |
| **Deb et al. (2002)** | Multi-Objective Optimization | NSGA-II vs. PAES, SPEA | Multi-criteria Pareto optimality | N/A (Algorithmic) | Formulates non-dominated sorting and crowding distance mechanics. |
| **Ouelhadj & Petrovic (2009)** | Dynamic Scheduling Survey | PDRs, Single-Obj GA, Multi-Obj | Makespan, Tardiness, Cost | Event-driven & Periodic | Justifies comparing PDRs (FCFS, SPT) with GA and NSGA-II. |
| **Xiong et al. (2013)** | Flexible Job Shop Breakdowns | NSGA-II variant | Makespan, Total Tardiness, Stability | Event-driven Rescheduling | Demonstrates that multi-objective rescheduling dominates simple right-shift. |
| **Panwalkar & Iskander (1977)** | Priority Dispatching | FCFS, SPT, EDD, MWKR | Flow time, Tardiness | Pure reactive dispatching | Validates SPT and FCFS as standard, fair baseline rules. |

---

## Part 2: Benchmark Dataset Selection

To make the experimental evaluation scientific, reproducible, and comparable to published literature, we must select established benchmark instances.

### 1. Candidate Benchmarks in Literature

1. **Fisher and Thompson (1963) - FT Instances**:
   - **FT06** ($6 \text{ jobs} \times 6 \text{ machines} = 36 \text{ operations}$): Small, well-behaved instance. Ideal for unit tests, rapid algorithm tuning, and verifying Pareto front shapes.
   - **FT10** ($10 \text{ jobs} \times 10 \text{ machines} = 100 \text{ operations}$): Known as the "notorious $10 \times 10$"—one of the most famous and hard-to-solve combinatorial benchmarks in operations research.
   - **FT20** ($20 \text{ jobs} \times 5 \text{ machines} = 100 \text{ operations}$): Rectangular configuration.
2. **Lawrence (1984) - LA Instances (LA01–LA40)**:
   - **LA01–LA05** ($10 \text{ jobs} \times 5 \text{ machines}$): Standard rectangular benchmarks used to evaluate machine load asymmetries.
3. **Taillard (1993) - TA Instances**:
   - Larger ($15 \times 15$ to $100 \times 20$), better suited for extreme scalability testing.

---

### 2. Selected Benchmark Suite for Our Research

We adopt a **two-tier benchmark strategy**:

| Dataset | Size ($n \times m$) | Operations | Primary Purpose in Our Study |
| :--- | :--- | :--- | :--- |
| **FT06** | $6 \times 6$ | 36 | **Prototyping & Validation**: Fast execution (<1s), clear visual Gantt charts, exact verification of schedule shift and stability logic. |
| **FT10** | $10 \times 10$ | 100 | **Primary Research Benchmark**: Standard international benchmark for comparing classical heuristics, GA, and NSGA-II under complex routing. |
| **LA01** | $10 \times 5$ | 50 | **Structural Sensitivity Test**: Rectangular shop floor (more jobs than machines) to test algorithm resilience to bottleneck machines. |

---

### 3. Augmenting Benchmarks for Dynamic Evaluation

Standard OR-Library instances only specify operations and processing times ($p_{i,j}$). To evaluate **tardiness** and **machine disruptions**, we augment them using standardized academic protocols:

#### A. Due Date Generation (Total Work Content - TWK Method)
For each job $J_i$, the total processing requirement is $P_i = \sum_{j=1}^m p_{i,j}$.
The due date $d_i$ is generated as:
$$d_i = k \cdot P_i$$
Where $k$ is the **tightness factor**:
- $k = 1.3$: Tight due dates (high tardiness pressure).
- $k = 1.6$: Moderate due dates (realistic industrial standard).

#### B. Machine Breakdown Disruption Scenarios
We define controlled, reproducible breakdown scenarios:
- **Disruption Time ($t_d$)**: Injected when the shop floor is in peak execution (at $30\%$ and $50\%$ of estimated baseline makespan).
- **Target Machine ($M_{\text{break}}$)**: The most heavily loaded (bottleneck) machine.
- **Repair Duration ($D$)**: Machine unavailable for $D$ time units (e.g., $D = 25$ or $D = 40$).
- **Impact**: Any ongoing operation on $M_{\text{break}}$ is paused or preempted; subsequent operations waiting for $M_{\text{break}}$ are delayed.
