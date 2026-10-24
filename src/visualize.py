"""
visualize.py: Visualization suite for Gantt charts and Pareto Trade-Off Frontiers.
Produces presentation-ready plots for academic reports and slide decks.
"""

from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from src.schedule import Schedule, ScheduledOperation
from src.disruption import BreakdownDisruption


def plot_gantt_chart(
    schedule: Schedule,
    disruption: Optional[BreakdownDisruption] = None,
    title: str = "Production Schedule Gantt Chart",
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """
    Renders an publication-quality Gantt chart of the job shop schedule.
    Highlights machine breakdown window if provided.
    """
    num_machines = schedule.instance.num_machines
    fig, ax = plt.subplots(figsize=(12, max(4, num_machines * 0.8)), dpi=150)

    # Distinct color palette per job
    cmap = plt.cm.get_cmap("tab20" if schedule.instance.num_jobs > 10 else "tab10")
    job_colors = {j: cmap(j % 20) for j in range(schedule.instance.num_jobs)}

    # Plot operations
    for op in schedule.operations:
        y_pos = op.machine_id - 0.35
        height = 0.7
        width = op.duration
        x_pos = op.start_time

        color = job_colors[op.job_id]
        rect = patches.Rectangle(
            (x_pos, y_pos),
            width,
            height,
            linewidth=1,
            edgecolor="#222222",
            facecolor=color,
            alpha=0.9,
        )
        ax.add_patch(rect)

        # Label inside block: J{i}O{j}
        text_label = f"J{op.job_id}O{op.op_idx}"
        ax.text(
            x_pos + width / 2,
            y_pos + height / 2,
            text_label,
            ha="center",
            va="center",
            color="black",
            fontsize=8,
            fontweight="bold",
        )

    # Highlight machine breakdown window
    if disruption is not None:
        y_pos = disruption.machine_id - 0.45
        height = 0.9
        x_pos = disruption.breakdown_start
        width = disruption.duration

        breakdown_patch = patches.Rectangle(
            (x_pos, y_pos),
            width,
            height,
            linewidth=1.5,
            edgecolor="#b30000",
            facecolor="#ff4d4d",
            hatch="//",
            alpha=0.6,
            zorder=5,
        )
        ax.add_patch(breakdown_patch)
        ax.text(
            x_pos + width / 2,
            y_pos + height / 2,
            f"M{disruption.machine_id} FAIL\n({x_pos}-{x_pos+width})",
            ha="center",
            va="center",
            color="white",
            fontsize=7,
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="black", alpha=0.7),
            zorder=6,
        )

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)], fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (Timeline Units)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Production Machines", fontsize=10, fontweight="bold")
    ax.set_title(
        f"{title}\nMakespan: {schedule.makespan} | Tardiness: {schedule.total_tardiness}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlim(0, max(schedule.makespan + 5, 20))
    ax.set_ylim(-0.6, num_machines - 0.4)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def plot_pareto_front(
    pareto_objs: np.ndarray,
    baseline_points: Optional[Dict[str, Tuple[float, float, float]]] = None,
    save_path: Optional[str] = None,
    title: str = "NSGA-II Pareto Trade-Off Frontier vs Baselines",
):
    """
    Plots the trade-off frontier between Makespan and Tardiness,
    with marker size/color representing Schedule Instability.
    Overlays discrete points for baselines (Right-Shift, FCFS, SPT, GA).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # 1. NSGA-II Pareto Points
    # pareto_objs: [Makespan, Tardiness, Instability]
    sc = ax.scatter(
        pareto_objs[:, 0],
        pareto_objs[:, 1],
        c=pareto_objs[:, 2],
        cmap="viridis",
        s=120,
        edgecolors="black",
        linewidths=1.5,
        alpha=0.9,
        label="NSGA-II Non-Dominated Front",
        zorder=4,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Schedule Instability (ΔS)", fontsize=10, fontweight="bold")

    # Connect Pareto points with a dashed line to indicate front topology
    sorted_order = np.argsort(pareto_objs[:, 0])
    ax.plot(
        pareto_objs[sorted_order, 0],
        pareto_objs[sorted_order, 1],
        linestyle="--",
        color="#2b5c8f",
        alpha=0.6,
        zorder=3,
    )

    # 2. Overlay Baselines
    if baseline_points:
        markers = {"Right-Shift": "s", "FCFS": "^", "SPT": "D", "Single-Obj GA": "X"}
        colors = {"Right-Shift": "#e63946", "FCFS": "#e76f51", "SPT": "#f4a261", "Single-Obj GA": "#2a9d8f"}

        for name, pt in baseline_points.items():
            mk = markers.get(name, "o")
            col = colors.get(name, "#333333")
            ax.scatter(
                pt[0],
                pt[1],
                marker=mk,
                color=col,
                s=160,
                edgecolors="black",
                linewidths=1.8,
                label=f"{name} (Instability={int(pt[2])})",
                zorder=5,
            )
            ax.annotate(
                name,
                (pt[0], pt[1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=col),
            )

    ax.set_xlabel("Makespan (C_max) → (Minimizing)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Tardiness (Σ T_j) → (Minimizing)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
