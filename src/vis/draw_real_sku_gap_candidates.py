from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_cols(df):
    cols = df.columns.tolist()

    image_col = None
    for c in ["image", "image_path", "file", "filename", "img", "img_path"]:
        if c in cols:
            image_col = c
            break

    if image_col is None:
        raise ValueError(f"Cannot infer image column from {cols}")

    candidates = [
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]

    for box_cols in candidates:
        if all(c in cols for c in box_cols):
            return image_col, box_cols

    if all(c in cols for c in ["x", "y", "w", "h"]):
        return image_col, ("x", "y", "w", "h")

    raise ValueError(f"Cannot infer box columns from {cols}")


def to_xyxy(row, box_cols):
    if box_cols == ("x", "y", "w", "h"):
        x1 = float(row["x"])
        y1 = float(row["y"])
        x2 = x1 + float(row["w"])
        y2 = y1 + float(row["h"])
    else:
        x1, y1, x2, y2 = [float(row[c]) for c in box_cols]

    return x1, y1, x2, y2


def group_rows(boxes, y_thr=0.55):
    # boxes: list of dict with cx, cy, w, h
    boxes = sorted(boxes, key=lambda b: b["cy"])

    rows = []
    for b in boxes:
        assigned = False
        for row in rows:
            med_y = np.median([x["cy"] for x in row])
            med_h = np.median([x["h"] for x in row])
            if abs(b["cy"] - med_y) <= y_thr * max(med_h, b["h"]):
                row.append(b)
                assigned = True
                break

        if not assigned:
            rows.append([b])

    rows = [sorted(r, key=lambda b: b["cx"]) for r in rows if len(r) >= 4]
    return rows


def detect_gap_candidates(rows, gap_factor=1.8, min_gap_px=20):
    candidates = []

    for ridx, row in enumerate(rows):
        widths = [b["w"] for b in row]
        median_w = np.median(widths)

        for i in range(len(row) - 1):
            left = row[i]
            right = row[i + 1]

            gap = right["x1"] - left["x2"]

            if gap <= max(min_gap_px, gap_factor * median_w):
                continue

            y1 = min(left["y1"], right["y1"])
            y2 = max(left["y2"], right["y2"])
            x1 = left["x2"]
            x2 = right["x1"]

            if x2 <= x1:
                continue

            candidates.append({
                "row": ridx,
                "between_index": i,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "gap_width": gap,
                "median_width": median_w,
                "suggestion": "same SKU as left neighbor" if left["w"] >= right["w"] else "same SKU as right neighbor",
                "left_box": left,
                "right_box": right,
            })

    return candidates


def draw_image(img_path, boxes, candidates, out_path, max_boxes=120):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # product boxes: green
    for b in boxes[:max_boxes]:
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=(0, 180, 80), width=1)

    # candidate gaps: red
    for idx, c in enumerate(candidates):
        draw.rectangle([c["x1"], c["y1"], c["x2"], c["y2"]], outline=(255, 0, 0), width=4)

        # left/right reference boxes
        lb = c["left_box"]
        rb = c["right_box"]
        draw.rectangle([lb["x1"], lb["y1"], lb["x2"], lb["y2"]], outline=(0, 120, 255), width=3)
        draw.rectangle([rb["x1"], rb["y1"], rb["x2"], rb["y2"]], outline=(255, 180, 0), width=3)

        text = f"gap {idx+1}: {c['suggestion']}"
        draw.text((c["x1"], max(0, c["y1"] - 18)), text, fill=(255, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--gap-factor", type=float, default=1.8)
    parser.add_argument("--min-gap-px", type=float, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)

    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    ann_path = processed_root / "annotations" / f"{args.split}_objects.csv"

    if not ann_path.exists():
        raise FileNotFoundError(f"Cannot find {ann_path}")

    df = pd.read_csv(ann_path)
    image_col, box_cols = infer_cols(df)

    out_dir = output_root / "figures" / "real_sku_gap_candidates" / args.split
    table_rows = []

    image_values = df[image_col].drop_duplicates().tolist()[: args.num]

    for image_name in tqdm(image_values, desc="Draw real SKU gap candidates"):
        sub = df[df[image_col] == image_name]

        boxes = []
        for _, row in sub.iterrows():
            x1, y1, x2, y2 = to_xyxy(row, box_cols)
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue

            boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
            })

        rows = group_rows(boxes)
        candidates = detect_gap_candidates(rows, gap_factor=args.gap_factor, min_gap_px=args.min_gap_px)

        # 尝试找到图片路径
        img_path = Path(image_name)
        if not img_path.exists():
            # 常见情况：image_name 只是文件名
            for key in ["sku_root", "sku110k_root", "raw_sku_root"]:
                if key in cfg:
                    root = Path(cfg[key])
                    p1 = root / "images" / args.split / image_name
                    p2 = root / "images" / image_name
                    p3 = root / image_name
                    for p in [p1, p2, p3]:
                        if p.exists():
                            img_path = p
                            break
                if img_path.exists():
                    break

        if not img_path.exists():
            # 根据 annotations 中可能保存的相对路径
            if "raw_root" in cfg:
                p = Path(cfg["raw_root"]) / image_name
                if p.exists():
                    img_path = p

        if img_path.exists() and len(candidates) > 0:
            out_path = out_dir / f"{Path(image_name).stem}_real_gap_candidates.jpg"
            draw_image(img_path, boxes, candidates, out_path)

        for c in candidates:
            table_rows.append({
                "image": image_name,
                "row": c["row"],
                "x1": round(c["x1"], 2),
                "y1": round(c["y1"], 2),
                "x2": round(c["x2"], 2),
                "y2": round(c["y2"], 2),
                "gap_width": round(c["gap_width"], 2),
                "median_width": round(c["median_width"], 2),
                "suggestion": c["suggestion"],
            })

    table = pd.DataFrame(table_rows)
    table_path = output_root / "tables" / f"real_sku_gap_candidates_{args.split}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Saved visualizations:", out_dir)
    print("Saved table:", table_path)
    print("Candidates:", len(table))
    print("=" * 80)


if __name__ == "__main__":
    main()
