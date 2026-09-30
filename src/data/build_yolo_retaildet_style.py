from pathlib import Path
import argparse
import json
import yaml
import os
import shutil
from tqdm import tqdm


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def xyxy_to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height

    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return xc, yc, w, h


def safe_name(s):
    return (
        s.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".jpg", "")
        .replace(".png", "")
    )


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def process_split(split, cfg, mode="joint"):
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    in_path = processed_root / "masked" / f"{split}_masked_img.jsonl"
    records = read_jsonl(in_path)

    yolo_root = processed_root / "yolo_retaildet_style"
    img_dir = yolo_root / "images" / split
    label_dir = yolo_root / "labels" / split
    manifest_path = yolo_root / f"{split}_manifest.jsonl"

    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    kept = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for rec in tqdm(records, desc=f"Build YOLO {split}"):
            image_path = rec.get("masked_image_path", rec["image_path"])
            image_src = Path(image_path)

            if not image_src.exists():
                continue

            name = safe_name(rec["masked_id"]) + ".jpg"
            dst_img = img_dir / name
            dst_label = label_dir / name.replace(".jpg", ".txt")

            link_or_copy(str(image_src), dst_img)

            lines = []

            width = float(rec["width"])
            height = float(rec["height"])

            if mode == "joint":
                # class 0: product
                for obj in rec["observed_objects"]:
                    xc, yc, w, h = xyxy_to_yolo(obj["bbox"], width, height)
                    if w > 0 and h > 0:
                        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                # class 1: vacancy
                for obj in rec["missing_objects"]:
                    xc, yc, w, h = xyxy_to_yolo(obj["bbox"], width, height)
                    if w > 0 and h > 0:
                        lines.append(f"1 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            else:
                # class 0: vacancy only
                for obj in rec["missing_objects"]:
                    xc, yc, w, h = xyxy_to_yolo(obj["bbox"], width, height)
                    if w > 0 and h > 0:
                        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            mf.write(json.dumps({
                "image_name": name,
                "masked_id": rec["masked_id"],
                "image_path": str(dst_img),
                "gt_missing": rec["missing_objects"],
                "width": rec["width"],
                "height": rec["height"]
            }, ensure_ascii=False) + "\n")

            kept += 1

    print(f"[完成] {split}: {kept} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, default="joint", choices=["joint", "vacancy_only"])
    args = parser.parse_args()

    cfg = load_config(args.config)

    for split in ["train", "val", "test"]:
        process_split(split, cfg, mode=args.mode)

    processed_root = Path(cfg["processed_root"])
    yolo_root = processed_root / "yolo_retaildet_style"

    if args.mode == "joint":
        names = ["product", "vacancy"]
        nc = 2
    else:
        names = ["vacancy"]
        nc = 1

    yaml_path = yolo_root / "retaildet_style.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {yolo_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names}\n")

    print("=" * 80)
    print("YOLO yaml:", yaml_path)
    print("=" * 80)


if __name__ == "__main__":
    main()