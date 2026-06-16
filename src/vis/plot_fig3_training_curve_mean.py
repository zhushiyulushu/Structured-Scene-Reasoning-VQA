from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SEEDS = [42, 2024, 3407]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])

    curves = []

    for s in SEEDS:
        path = output_root / "logs" / f"learnable_grid_train_seed_{s}.csv"
        df = pd.read_csv(path)
        curves.append(df[["epoch", "val_f1"]].copy())

    epochs = curves[0]["epoch"].values
    values = np.stack([c["val_f1"].values for c in curves], axis=0)
    mean_curve = values.mean(axis=0)

    final_mean = mean_curve[-1]

    out_dir = output_root / "figures" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for s, vals in zip(SEEDS, values):
        plt.plot(epochs, vals, linestyle="--", linewidth=1.2, label=f"Seed {s}")

    plt.plot(epochs, mean_curve, linewidth=2.5, label="Mean")

    plt.xlabel("Epoch")
    plt.ylabel("Validation F1")
    plt.ylim(0, 1.0)
    plt.title("Learnable Grid Completion training under three seeds")
    plt.legend()

    plt.text(
        epochs[-1],
        final_mean,
        f" mean={final_mean:.3f}",
        va="center",
        ha="left"
    )

    plt.tight_layout()

    out_path = out_dir / "fig3_training_curve_mean.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()