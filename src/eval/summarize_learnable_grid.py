from pathlib import Path
import argparse
import yaml
import pandas as pd
import re


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_seed(path):
    m = re.search(r"seed_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])
    table_dir = output_root / "tables"

    files = sorted(table_dir.glob("learnable_grid_summary_seed_*.csv"))

    if not files:
        raise FileNotFoundError("No learnable_grid_summary_seed_*.csv found.")

    all_rows = []
    for f in files:
        seed = extract_seed(f)
        df = pd.read_csv(f)
        df["Seed"] = seed
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)

    out_all = table_dir / "learnable_grid_all_seeds.csv"
    all_df.to_csv(out_all, index=False, encoding="utf-8-sig")

    metrics = ["Precision", "Recall", "F1", "MeanIoU", "MedianIoU"]

    summary_rows = []
    for split, g in all_df.groupby("Split"):
        row = {"Split": split, "NumSeeds": g["Seed"].nunique()}
        for m in metrics:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=0)
            row[f"{m}_mean_std"] = f"{g[m].mean():.4f} ± {g[m].std(ddof=0):.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_summary = table_dir / "learnable_grid_mean_std.csv"
    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[All seeds]")
    print(all_df)
    print("\n[Mean ± Std]")
    print(summary_df)
    print("=" * 80)
    print("Saved:", out_all)
    print("Saved:", out_summary)


if __name__ == "__main__":
    main()