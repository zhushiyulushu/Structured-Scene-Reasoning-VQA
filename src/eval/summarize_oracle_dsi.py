from pathlib import Path
import argparse
import json
import yaml
import pandas as pd


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["output_root"])

    rows = []
    for split in ["train", "val", "test"]:
        path = output_root / "results" / f"oracle_dsi_{split}.json"
        if not path.exists():
            print(f"[跳过] 找不到 {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)

        rows.append({
            "Split": split,
            "Samples": r["num_samples"],
            "TP": r["TP"],
            "FP": r["FP"],
            "FN": r["FN"],
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1": round(r["f1"], 4),
            "Count Acc": round(r["vqa_count_acc"], 4),
            "Locate Acc": round(r["vqa_locate_acc"], 4),
            "Verify Acc": round(r["vqa_verify_acc"], 4),
            "Identify Acc": round(r["vqa_identify_acc"], 4),
        })

    df = pd.DataFrame(rows)

    out_dir = output_root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "oracle_dsi_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(df)
    print("=" * 80)
    print(f"[完成] 表格已保存：{out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()