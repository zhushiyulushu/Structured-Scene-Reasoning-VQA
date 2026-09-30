from pathlib import Path
import argparse
import json
import yaml
from tqdm import tqdm


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_locations(missing_objects):
    locs = []
    for obj in missing_objects:
        # 对外展示使用 1-indexed，更符合论文表达 row 1, column 4
        r = obj["row"] + 1
        c = obj["col"] + 1
        locs.append(f"row {r}, column {c}")
    return "; ".join(locs)


def build_questions(record):
    image_id = record["image_id"]
    masked_id = record["masked_id"]
    missing_objects = record["missing_objects"]
    num_missing = len(missing_objects)

    loc_text = format_locations(missing_objects)
    first_row = missing_objects[0]["row"] + 1
    expected_group = missing_objects[0].get("expected_group", "object")

    qa_list = []

    # 1. Count
    qa_list.append({
        "qid": f"{masked_id}__count",
        "image_id": image_id,
        "masked_id": masked_id,
        "type": "count",
        "question": "How many items are missing in this shelf image?",
        "answer": f"There {'is' if num_missing == 1 else 'are'} {num_missing} missing item{'s' if num_missing != 1 else ''}.",
        "missing_objects": missing_objects
    })

    # 2. Locate
    qa_list.append({
        "qid": f"{masked_id}__locate",
        "image_id": image_id,
        "masked_id": masked_id,
        "type": "locate",
        "question": "Where is the missing item located?",
        "answer": f"The missing item is located at {loc_text}.",
        "missing_objects": missing_objects
    })

    # 3. Verify
    qa_list.append({
        "qid": f"{masked_id}__verify",
        "image_id": image_id,
        "masked_id": masked_id,
        "type": "verify",
        "question": f"Is there any missing item in row {first_row}?",
        "answer": f"Yes, there is a missing item in row {first_row}.",
        "missing_objects": missing_objects
    })

    # 4. Identify
    qa_list.append({
        "qid": f"{masked_id}__identify",
        "image_id": image_id,
        "masked_id": masked_id,
        "type": "identify",
        "question": "Which product group is expected at the missing position?",
        "answer": f"The missing position is expected to contain an item from the {expected_group} group.",
        "missing_objects": missing_objects
    })

    return qa_list


def process_split(split: str, cfg: dict):
    in_jsonl = Path(cfg["processed_root"]) / "masked" / f"{split}_masked.jsonl"
    out_dir = Path(cfg["processed_root"]) / "vqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{split}_qa.jsonl"

    if not in_jsonl.exists():
        raise FileNotFoundError(f"找不到输入文件：{in_jsonl}，请先运行 topological_mask.py")

    records = read_jsonl(in_jsonl)

    total_qa = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc=f"Building VQA {split}"):
            qa_list = build_questions(record)
            for qa in qa_list:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                total_qa += 1

    print("=" * 80)
    print(f"[完成] {split} VQA pairs")
    print(f"输出文件：{out_jsonl}")
    print(f"QA 数量：{total_qa}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        process_split(split, cfg)


if __name__ == "__main__":
    main()