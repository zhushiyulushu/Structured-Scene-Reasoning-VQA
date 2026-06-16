from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SEEDS = [42, 2024, 3407]

CURVE_COLORS = {
    42: "#7aa6c2",
    2024: "#e3a384",
    3407: "#8fc97a",
}

MEAN_COLOR = "#333333"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find columns {candidates}. Existing: {list(df.columns)}")


def load_curves(output_root):
    curves = {}

    for seed in SEEDS:
        path = output_root / "logs" / f"learnable_grid_train_seed_{seed}.csv"
        df = pd.read_csv(path)

        epoch_col = pick_column(df, ["epoch", "Epoch"])
        loss_col = pick_column(df, ["loss", "train_loss", "Loss"])

        # 优先用 accuracy；没有 accuracy 时用 val_f1
        metric_col = None
        metric_name = None

        for c in ["val_accuracy", "val_acc", "accuracy", "acc"]:
            if c in df.columns:
                metric_col = c
                metric_name = "Validation Accuracy"
                break

        if metric_col is None:
            metric_col = pick_column(df, ["val_f1", "F1", "val_F1", "Val_F1"])
            metric_name = "Validation Accuracy (F1)"

        curves[seed] = {
            "epoch": df[epoch_col].values,
            "loss": df[loss_col].values,
            "metric": df[metric_col].values,
            "metric_name": metric_name,
        }

    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])

    curves = load_curves(output_root)
    epochs = curves[SEEDS[0]]["epoch"]

    loss_values = np.stack([curves[s]["loss"] for s in SEEDS], axis=0)
    metric_values = np.stack([curves[s]["metric"] for s in SEEDS], axis=0)

    loss_mean = loss_values.mean(axis=0)
    metric_mean = metric_values.mean(axis=0)

    metric_name = curves[SEEDS[0]]["metric_name"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # 左图：Loss
    ax = axes[0]
    for seed in SEEDS:
        ax.plot(
            curves[seed]["epoch"],
            curves[seed]["loss"],
            linestyle="--",
            linewidth=1.4,
            color=CURVE_COLORS[seed],
            alpha=0.9,
            label=f"Seed {seed}",
        )

    ax.plot(
        epochs,
        loss_mean,
        linestyle="-",
        linewidth=1.5,  # 👈 已改细
        color=MEAN_COLOR,
        label="Mean",
    )

    ax.set_title("(a) Training Loss", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=13)

    # 右图：Accuracy / F1
    ax = axes[1]
    for seed in SEEDS:
        ax.plot(
            curves[seed]["epoch"],
            curves[seed]["metric"],
            linestyle="--",
            linewidth=1.4,
            color=CURVE_COLORS[seed],
            alpha=0.9,
            label=f"Seed {seed}",
        )

    ax.plot(
        epochs,
        metric_mean,
        linestyle="-",
        linewidth=1.5,
        color=MEAN_COLOR,
        label="Mean",
    )

    ax.set_title(f"(b) {metric_name}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel(metric_name, fontsize=15)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=13)

    # 改动点：x坐标左移12个epoch长度，文字不会贴右边界
    ax.text(
        epochs[-1] - 23,
        metric_mean[-1] + 0.02,
        f"mean={metric_mean[-1]:.3f}",
        fontsize=15,
        va="center",
        color=MEAN_COLOR,
    )

    for ax in axes:
        # 这一行是新增：控制横轴、纵轴上面数字的字号
        ax.tick_params(axis='both', labelsize=14)
        for spine in ax.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(1.0)

    # ========== 删除下面这整块大标题代码 ==========
    # fig.suptitle(
    #     "Training Dynamics of Learnable Grid Completion",
    #     fontsize=18,
    #     fontweight="bold",
    #     y=1.03,
    # )

    plt.tight_layout()

    out_dir = output_root / "figures" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "fig3_loss_accuracy_combined.png"
    out_pdf = out_dir / "fig3_loss_accuracy_combined.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()
