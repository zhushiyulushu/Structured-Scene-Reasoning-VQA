from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    out_dir = Path("outputs/figures/paper_figures")
    table_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Values are percentages for plotting.
    # F1-score is also multiplied by 100 for a unified percentage-style bar chart.
    methods = [
        "Heuristic",
        "RetailDet (Vis)",
        "GT-Planogram",
        "Our SSR Grid",
    ]

    metrics = ["Recall", "Precision", "F1-score"]

    # Three independent runs. Deterministic baselines are repeated with identical values.
    runs = {
        "Run 1": {
            "Heuristic":       {"Recall": 40.29, "Precision": 15.99, "F1-score": 22.90},
            "RetailDet (Vis)": {"Recall": 96.44, "Precision": 99.50, "F1-score": 97.94},
            "GT-Planogram":    {"Recall": 84.94, "Precision": 92.47, "F1-score": 88.54},
            "Our SSR Grid":    {"Recall": 98.43, "Precision": 97.98, "F1-score": 98.20},
        },
        "Run 2": {
            "Heuristic":       {"Recall": 40.29, "Precision": 15.99, "F1-score": 22.90},
            "RetailDet (Vis)": {"Recall": 96.44, "Precision": 99.50, "F1-score": 97.94},
            "GT-Planogram":    {"Recall": 84.94, "Precision": 92.47, "F1-score": 88.54},
            "Our SSR Grid":    {"Recall": 98.32, "Precision": 97.88, "F1-score": 98.10},
        },
        "Run 3": {
            "Heuristic":       {"Recall": 40.29, "Precision": 15.99, "F1-score": 22.90},
            "RetailDet (Vis)": {"Recall": 82.77, "Precision": 96.81, "F1-score": 89.24},
            "GT-Planogram":    {"Recall": 84.94, "Precision": 92.47, "F1-score": 88.54},
            "Our SSR Grid":    {"Recall": 95.04, "Precision": 92.64, "F1-score": 93.83},
        },
    }

    # Save the plotting source values for traceability.
    rows = []
    for run_name, run_data in runs.items():
        for method in methods:
            row = {"Run": run_name, "Method": method}
            row.update(run_data[method])
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = table_dir / "fig4_no_blip_seed_values.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Unified soft color palette.
    colors = {
        "Heuristic": "#D8C1B2",
        "RetailDet (Vis)": "#AFC6DD",
        "GT-Planogram": "#F1E3A3",
        "Our SSR Grid": "#A8D99C",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)

    bar_width = 0.18
    x = np.arange(len(metrics))

    for ax, (run_name, run_data) in zip(axes, runs.items()):
        for i, method in enumerate(methods):
            offset = (i - 1.5) * bar_width
            values = [run_data[method][m] for m in metrics]
            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                label=method,
                color=colors[method],
                edgecolor="#555555",
                linewidth=0.6,
                alpha=0.95,
            )

            # 柱子上方数值标签保持13号
            for b, v in zip(bars, values):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 1.2,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    rotation=90,
                )

        # 子图标题改为16号，取消加粗
        ax.set_title(run_name, fontsize=16)
        ax.set_xticks(x)
        # X轴分类文字改为16号
        ax.set_xticklabels(metrics, fontsize=16)
        # 新增：XY轴刻度数字字号13
        ax.tick_params(axis="both", labelsize=13)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_axisbelow(True)

    # Y轴总标签改为16号
    axes[0].set_ylabel("Percentage / Score (%)", fontsize=16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=15,  # 图例改为15号
        bbox_to_anchor=(0.5, 1.05),
    )

    # 注释掉全局大标题
    # fig.suptitle(
    #     "Performance Comparison Across Three Independent Runs",
    #     fontsize=15,
    #     fontweight="bold",
    #     y=1.13,
    # )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    png_path = out_dir / "fig4_comparison_seed_bars_no_blip.png"
    pdf_path = out_dir / "fig4_comparison_seed_bars_no_blip.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    # Also overwrite the paper-style filename used by the manuscript.
    fig.savefig(out_dir / "fig4_comparison_seed_bars_paper_style.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_comparison_seed_bars_paper_style.pdf", bbox_inches="tight")

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", out_dir / "fig4_comparison_seed_bars_paper_style.png")
    print("Saved:", out_dir / "fig4_comparison_seed_bars_paper_style.pdf")
    print("Source CSV:", csv_path)


if __name__ == "__main__":
    main()