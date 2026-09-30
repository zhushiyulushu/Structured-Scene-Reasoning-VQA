from pathlib import Path
import argparse
import json
import yaml
import pandas as pd


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_jsonl(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def audit_split(split, cfg):
    root = Path(cfg["processed_root"])

    ann_path = root / "annotations" / f"{split}_objects.csv"
    grid_path = root / "grid_labels" / f"{split}_grid.jsonl"
    masked_path = root / "masked" / f"{split}_masked.jsonl"
    qa_path = root / "vqa" / f"{split}_qa.jsonl"

    print("\n" + "=" * 80)
    print(f"[Audit] split = {split}")

    df = pd.read_csv(ann_path)
    print("objects.csv images:", df["image_id"].nunique())
    print("objects.csv boxes:", len(df))
    print("avg boxes/image:", round(len(df) / df["image_id"].nunique(), 2))

    grid_records = read_jsonl(grid_path)
    masked_records = read_jsonl(masked_path)
    qa_records = read_jsonl(qa_path)

    print("grid samples:", len(grid_records))
    print("masked samples:", len(masked_records))
    print("qa samples:", len(qa_records))
    print("qa per masked:", round(len(qa_records) / max(len(masked_records), 1), 2))

    bad_missing = 0
    bad_observed = 0

    for rec in masked_records:
        if rec["num_missing"] != len(rec["missing_objects"]):
            bad_missing += 1

        total_after_plus_missing = len(rec["observed_objects"]) + len(rec["missing_objects"])
        if total_after_plus_missing <= len(rec["observed_objects"]):
            bad_observed += 1

        for m in rec["missing_objects"]:
            if m["row"] < 0 or m["col"] < 0:
                bad_missing += 1

    print("bad missing records:", bad_missing)
    print("bad observed records:", bad_observed)

    type_count = {}
    for qa in qa_records:
        type_count[qa["type"]] = type_count.get(qa["type"], 0) + 1
    print("QA type count:", type_count)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    for split in ["train", "val", "test"]:
        audit_split(split, cfg)


if __name__ == "__main__":
    main()