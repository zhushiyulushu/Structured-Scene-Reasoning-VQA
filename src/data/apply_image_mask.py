from pathlib import Path
import argparse
import json
import yaml
import cv2
import numpy as np
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_name(name):
    return (
        name.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".jpg", "")
        .replace(".png", "")
    )


def clip_box(box, w, h, pad=2):
    x1, y1, x2, y2 = box
    x1 = max(0, int(round(x1)) - pad)
    y1 = max(0, int(round(y1)) - pad)
    x2 = min(w - 1, int(round(x2)) + pad)
    y2 = min(h - 1, int(round(y2)) + pad)
    return x1, y1, x2, y2


def apply_mask_to_image(img, boxes, mode="inpaint", pad=2):
    h, w = img.shape[:2]

    if mode == "inpaint":
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = clip_box(box, w, h, pad=pad)
            mask[y1:y2, x1:x2] = 255

        # Telea inpainting：比黑块/灰块更不容易引入人工强提示
        out = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return out

    out = img.copy()

    for box in boxes:
        x1, y1, x2, y2 = clip_box(box, w, h, pad=pad)

        if mode == "gray":
            out[y1:y2, x1:x2] = (128, 128, 128)

        elif mode == "black":
            out[y1:y2, x1:x2] = (0, 0, 0)

        elif mode == "mean":
            # 使用周围区域均值填充，避免纯黑/纯灰太显眼
            px1 = max(0, x1 - 20)
            py1 = max(0, y1 - 20)
            px2 = min(w - 1, x2 + 20)
            py2 = min(h - 1, y2 + 20)

            patch = out[py1:py2, px1:px2]
            if patch.size > 0:
                mean_color = patch.reshape(-1, 3).mean(axis=0)
            else:
                mean_color = np.array([128, 128, 128])

            out[y1:y2, x1:x2] = mean_color.astype(np.uint8)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return out


def process_split(split, cfg, mode="inpaint", pad=2):
    processed_root = Path(cfg["processed_root"])

    in_path = processed_root / "masked" / f"{split}_masked.jsonl"
    out_jsonl = processed_root / "masked" / f"{split}_masked_img.jsonl"
    out_img_dir = processed_root / "masked_images" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{in_path}")

    records = read_jsonl(in_path)

    kept = 0
    skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in tqdm(records, desc=f"Applying image mask {split}"):
            img_path = rec["image_path"]
            img = cv2.imread(img_path)

            if img is None:
                skipped += 1
                continue

            boxes = [obj["bbox"] for obj in rec["missing_objects"]]
            masked_img = apply_mask_to_image(img, boxes, mode=mode, pad=pad)

            out_name = f"{safe_name(rec['masked_id'])}_{mode}.jpg"
            out_path = out_img_dir / out_name
            cv2.imwrite(str(out_path), masked_img)

            new_rec = dict(rec)
            new_rec["original_image_path"] = rec["image_path"]
            new_rec["masked_image_path"] = str(out_path)
            new_rec["image_path"] = str(out_path)
            new_rec["image_mask_mode"] = mode

            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 80)
    print(f"[完成] {split} image-level masking")
    print(f"输入：{in_path}")
    print(f"输出 jsonl：{out_jsonl}")
    print(f"输出图片目录：{out_img_dir}")
    print(f"生成：{kept}")
    print(f"跳过：{skipped}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--mode", type=str, default="inpaint", choices=["inpaint", "gray", "black", "mean"])
    parser.add_argument("--pad", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, cfg, mode=args.mode, pad=args.pad)


if __name__ == "__main__":
    main()