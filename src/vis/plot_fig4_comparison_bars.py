from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])
    table_dir = output_root / "tables"

    df = pd.read_csv(table_dir / "table2_comparison_seed_results.csv")

    methods = [
        "Heuristic (Rule-based)",
        "BLIP (Vision-Language)",
        "Baseline RetailDet (Visual)",
        "Baseline GT-Planogram Alignment",
        "Ours / Our SSR Grid"
    ]

    means = []
    stds = []
    seed_values = []

    for m in methods:
        vals = df[df["Method"] == m]["Score"].values.astype(float)
        means.append(vals.mean())
        stds.append(vals.std(ddof=0))
        seed_values.append(vals)

    x = np.arange(len(methods))

    out_dir = output_root / "figures" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=5)

    for i, vals in enumerate(seed_values):
        jitter = np.linspace(-0.12, 0.12, len(vals))
        plt.scatter(np.full(len(vals), x[i]) + jitter, vals, marker="o")

    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Comparison under three random seeds")

    for i, m in enumerate(means):
        plt.text(x[i], m + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    out_path = out_dir / "fig4_comparison_seed_bars.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()