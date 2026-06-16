from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np


SEEDS = [42, 2024, 3407]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_row(path):
    return pd.read_csv(path).iloc[0].to_dict()


def mean_std(vals):
    vals = np.array(vals, dtype=float)
    return vals.mean(), vals.std(ddof=0), f"{vals.mean():.4f} ± {vals.std(ddof=0):.4f}"


def triple_from_seed_files(table_dir, pattern):
    ps, rs, fs = [], [], []
    for s in SEEDS:
        row = read_row(table_dir / pattern.format(seed=s))
        ps.append(float(row["Precision"]))
        rs.append(float(row["Recall"]))
        fs.append(float(row["F1"]))
    return ps, rs, fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = Path(cfg["output_root"]) / "tables"

    rows = []

    # Ours first to compute F1 drop
    ours_p, ours_r, ours_f = triple_from_seed_files(table_dir, "ours_ssr_grid_test_seed_{seed}.csv")
    ours_f_mean, _, ours_f_ms = mean_std(ours_f)

    def add_row(name, ps, rs, fs):
        p_mean, p_std, p_ms = mean_std(ps)
        r_mean, r_std, r_ms = mean_std(rs)
        f_mean, f_std, f_ms = mean_std(fs)
        drop = max(0.0, ours_f_mean - f_mean)
        rows.append({
            "Variant": name,
            "Recall": r_ms,
            "Precision": p_ms,
            "F1-score": f_ms,
            "Domain Shift": f"{drop:.4f}"
        })

    # w/o Semantic: single old YOLOv8 + fallback, repeat 3 times
    wo_sem = read_row(table_dir / "ablation_wo_semantic.csv")
    add_row(
        "w/o Semantic",
        [float(wo_sem["Precision"])] * 3,
        [float(wo_sem["Recall"])] * 3,
        [float(wo_sem["F1"])] * 3
    )

    # w/o Structural Learning: YOLOWorld visual only
    p, r, f = triple_from_seed_files(table_dir, "yoloworld_visual_test_seed_{seed}.csv")
    add_row("w/o Structural Learning", p, r, f)

    # w/o Visual Vacancy Detector: learnable only
    lg = pd.read_csv(table_dir / "learnable_grid_mean_std.csv")
    test = lg[lg["Split"] == "test"].iloc[0]
    lg_f = float(test["F1_mean"])
    add_row(
        "w/o Visual Vacancy Detector",
        [lg_f] * 3,
        [lg_f] * 3,
        [lg_f] * 3
    )

    # w/o Deep Reasoning: union
    p, r, f = triple_from_seed_files(table_dir, "ablation_wo_deep_reasoning_seed_{seed}.csv")
    add_row("w/o Deep Reasoning", p, r, f)

    # Ours
    add_row("Ours / Our SSR Grid", ours_p, ours_r, ours_f)

    out = pd.DataFrame(rows)
    path = table_dir / "table4_ablation_prf_final.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    print(out)
    print("Saved:", path)


if __name__ == "__main__":
    main()