from pathlib import Path
import argparse
import json
import random
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


def box_width(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return max(1.0, x2 - x1)


def center_x(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return (x1 + x2) / 2.0


def y_overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]

    inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    denom = max(1.0, min(ay2 - ay1, by2 - by1))
    return inter / denom


def has_good_left_right_neighbor(obj, obj_map, median_w):
    """
    主实验采用 bilateral context：
    被 mask 的目标必须同时有左邻居和右邻居。
    这与论文中的 contextual envelope / interpolation 逻辑一致。
    """
    r = obj["row"]
    c = obj["col"]

    left = obj_map.get((r, c - 1), None)
    right = obj_map.get((r, c + 1), None)

    if left is None or right is None:
        return False

    # 几何顺序必须成立
    if not (center_x(left) < center_x(obj) < center_x(right)):
        return False

    # 必须处在相近的垂直范围内，否则可能是行聚类错误
    if y_overlap_ratio(left, obj) < 0.25:
        return False
    if y_overlap_ratio(right, obj) < 0.25:
        return False

    # 左右距离不能过大，避免跨货架隔断或真实大空区
    lx_gap = obj["bbox"][0] - left["bbox"][2]
    rx_gap = right["bbox"][0] - obj["bbox"][2]

    max_gap = max(2.5 * median_w, 2.5 * box_width(obj))

    if lx_gap > max_gap:
        return False
    if rx_gap > max_gap:
        return False

    return True


def create_masked_sample(record, mask_per_image=1, rng=None):
    if rng is None:
        rng = random.Random(42)

    objects = record["objects"]
    obj_map = {(obj["row"], obj["col"]): obj for obj in objects}

    widths = [box_width(obj) for obj in objects]
    median_w = sorted(widths)[len(widths) // 2]

    candidates = []
    for obj in objects:
        if has_good_left_right_neighbor(obj, obj_map, median_w):
            candidates.append(obj)

    if len(candidates) == 0:
        return None

    selected = rng.sample(candidates, k=min(mask_per_image, len(candidates)))
    selected_ids = {obj["object_id"] for obj in selected}

    observed_objects = [
        obj for obj in objects
        if obj["object_id"] not in selected_ids
    ]

    missing_objects = []
    for obj in selected:
        missing_objects.append({
            "object_id": obj["object_id"],
            "bbox": obj["bbox"],
            "bbox_xywh": obj["bbox_xywh"],
            "bbox_norm_xywh": obj["bbox_norm_xywh"],
            "row": obj["row"],
            "col": obj["col"],
            "class_name": obj.get("class_name", "object"),
            "expected_group": obj.get("class_name", "object")
        })

    masked_record = {
        "masked_id": f"{record['image_id']}__mask_{selected[0]['object_id']}",
        "image_id": record["image_id"],
        "split": record["split"],
        "image_path": record["image_path"],
        "width": record["width"],
        "height": record["height"],
        "num_rows": record["num_rows"],
        "max_cols": record["max_cols"],
        "observed_objects": observed_objects,
        "missing_objects": missing_objects,
        "num_missing": len(missing_objects),
        "mask_context": "bilateral"
    }

    return masked_record


def process_split(split: str, cfg: dict, mask_per_image: int, seed: int):
    in_jsonl = Path(cfg["processed_root"]) / "grid_labels" / f"{split}_grid.jsonl"
    out_dir = Path(cfg["processed_root"]) / "masked"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{split}_masked.jsonl"

    if not in_jsonl.exists():
        raise FileNotFoundError(f"找不到输入文件：{in_jsonl}，请先运行 build_grid_labels.py")

    records = read_jsonl(in_jsonl)
    rng = random.Random(seed)

    kept = 0
    skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for record in tqdm(records, desc=f"Masking {split}"):
            masked = create_masked_sample(record, mask_per_image=mask_per_image, rng=rng)
            if masked is None:
                skipped += 1
                continue

            f.write(json.dumps(masked, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 80)
    print(f"[完成] {split} topological masking")
    print(f"输出文件：{out_jsonl}")
    print(f"生成 masked samples：{kept}")
    print(f"跳过图片：{skipped}")
    print(f"mask context：bilateral")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--mask-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        process_split(split, cfg, args.mask_per_image, args.seed)


if __name__ == "__main__":
    main()