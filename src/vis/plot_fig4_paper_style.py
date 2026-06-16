from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SEEDS = [42, 2024, 3407]

METHODS = [
    "Heuristic",
    "RetailDet (Vis)",
    "Baseline (Row)",
    "BLIP (VLM)",
    "Our SSR Grid",
]

COLORS = {
    "Heuristic": "#ead7cf",
    "RetailDet (Vis)": "#dbe5ef",
    "Baseline (Row)": "#fff6c9",
    "BLIP (VLM)": "#c9d8ee",
    "Our SSR Grid": "#b8e8aa",
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_single_row_csv(path):
    df = pd.read_csv(path)
    return df.iloc[0].to_dict()


def get_baseline_row(df, method):
    return df[df["Method"] == method].iloc[0].to_dict()


def add_bar_labels(ax, bars, values):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height < 3:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.3,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="#333333",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])
    table_dir = output_root / "tables"

    baseline_df = pd.read_csv(table_dir / "baseline_comparison_iou_only_test.csv")
    heuristic = get_baseline_row(baseline_df, "Heuristic Rule-based tuned")
    row_align = get_baseline_row(baseline_df, "GT-Planogram Alignment")

    blip = pd.read_csv(table_dir / "blip_baseline_summary_test_-1.csv")
    blip_avg = float(blip["accuracy"].mean())

    run_data = {}

    for seed in SEEDS:
        retail = read_single_row_csv(table_dir / f"yoloworld_visual_test_seed_{seed}.csv")
        ours = read_single_row_csv(table_dir / f"ours_ssr_grid_test_seed_{seed}.csv")

        run_data[seed] = {
            "Heuristic": {
                "Recall": float(heuristic["Recall"]),
                "Precision": float(heuristic["Precision"]),
                "F1-Score": float(heuristic["F1"]),
            },
            "RetailDet (Vis)": {
                "Recall": float(retail["Recall"]),
                "Precision": float(retail["Precision"]),
                "F1-Score": float(retail["F1"]),
            },
            "Baseline (Row)": {
                "Recall": float(row_align["Recall"]),
                "Precision": float(row_align["Precision"]),
                "F1-Score": float(row_align["F1"]),
            },
            # BLIP 不输出缺货框，这里使用结构化 VQA 平均准确率作为统一得分显示
            "BLIP (VLM)": {
                "Recall": blip_avg,
                "Precision": blip_avg,
                "F1-Score": blip_avg,
            },
            "Our SSR Grid": {
                "Recall": float(ours["Recall"]),
                "Precision": float(ours["Precision"]),
                "F1-Score": float(ours["F1"]),
            },
        }

    metrics = ["Recall", "Precision", "F1-Score"]
    x = np.arange(len(metrics))
    width = 0.145

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharey=True)

    for ax_idx, seed in enumerate(SEEDS):
        ax = axes[ax_idx]

        for i, method in enumerate(METHODS):
            values = [run_data[seed][method][m] * 100 for m in metrics]
            offset = (i - 2) * width

            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=method if ax_idx == 0 else None,
                color=COLORS[method],
                edgecolor="#777777",
                linewidth=0.8,
            )

            add_bar_labels(ax, bars, values)

        # 子图标题字号改为15
        ax.set_title(f"Run {ax_idx + 1}", fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        # X轴刻度文字15号
        ax.set_xticklabels(metrics, fontsize=15)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", linestyle="--", alpha=0.22)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            # Y轴标签字号15
            ax.set_ylabel("Percentage / Score (%)", fontsize=15)

        for spine in ax.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(1.0)

    # 注释掉总大标题
    # fig.suptitle(
    #     "Performance Comparison Across 3 Random Seeds",
    #     fontsize=15,
    #     fontweight="bold",
    #     y=1.04,
    # )

    # 图例字体改为13
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out_dir = output_root / "figures" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "fig4_comparison_seed_bars_paper_style.png"
    out_pdf = out_dir / "fig4_comparison_seed_bars_paper_style.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()