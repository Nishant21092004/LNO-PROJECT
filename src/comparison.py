"""Small, demo-friendly comparison reports. No optimization algorithms live here."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_comparison(rows, out_dir="results"):
    """Rows contain label, phase, makespan, tardiness, instability and runtime_s."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "comparison_summary.csv"
    fields = ["algorithm", "phase", "makespan", "tardiness", "instability", "runtime_s"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"{r['algorithm']}\n({r['phase']})" for r in rows]
    values = [r["makespan"] for r in rows]
    bars = ax.bar(range(len(rows)), values, color=[
        "#6487ba" if r["phase"] == "pre" else "#ed9e52" for r in rows
    ])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:g}", ha="center", fontsize=9)
    ax.set_xticks(range(len(rows)), labels, fontsize=9)
    ax.set_ylabel("Makespan (lower is better)")
    ax.set_title("Initial scheduling vs breakdown recovery (different phases)")
    ax.set_ylim(0, max(values) * 1.17)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "06_algorithm_comparison.png", dpi=160)
    plt.close(fig)

    post = [r for r in rows if r["phase"] == "post"]
    if post:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, metric, name in zip(axes, ["tardiness", "instability"],
                                    ["Total tardiness", "Schedule instability"]):
            values = [r[metric] for r in post]
            ax.bar(range(len(post)), values, color="#547aa5")
            ax.set_xticks(range(len(post)), [r["algorithm"] for r in post],
                          rotation=15, ha="right", fontsize=9)
            ax.set_title(name + " (post-disruption only)")
            ax.set_ylim(0, max(values + [1]) * 1.18)
            ax.grid(axis="y", alpha=0.2)
            for i, value in enumerate(values):
                ax.text(i, value + max(values + [1]) * 0.03, f"{value:g}", ha="center")
        fig.tight_layout()
        fig.savefig(out / "07_post_disruption_comparison.png", dpi=160)
        plt.close(fig)
    return csv_path
