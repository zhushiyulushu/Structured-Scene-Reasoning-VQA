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


def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    raise KeyError(f"Cannot find columns {candidates}. Existing: {list(df.columns)}")


def cluster_rows(boxes, y_thr=0.6):
    # boxes: list of dict with x1,y1,x2,y2,cx,cy,w,h
    boxes = sorted(boxes, key=lambda b: b["cy"])
    rows = []

    for b in boxes:
        placed = False
        for row in rows:
            median_h = np.median([x["h"] for x in row])
            row_cy = np.median([x["cy"] for x in row])
            if abs(b["cy"] - row_cy) < y_thr * median_h:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    for row in rows:
        row.sort(key=lambda b: b["cx"])

    return rows


def infer_gap_candidates(rows, gap_factor=1.8):
    gaps = []

    for r_idx, row in enumerate(rows):
        if len(row) < 3:
            continue

        widths = np.array([b["w"] for b in row])
        med_w = float(np.median(widths))

        for i in range(len(row) - 1):
            left = row[i]
            right = row[i + 1]

            gap = right["x1"] - left["x2"]

            if gap <= gap_factor * med_w:
                continue

            # 推断中间可能缺了几个货位
            num_missing = max(1, int(round(gap / max(med_w, 1))) - 1)
            slot_w = gap / (num_missing + 1)

            for k in range(num_missing):
                cx = left["x2"] + slot_w * (k + 1)
                h = np.median([left["h"], right["h"]])
                y1 = np.median([left["y1"], right["y1"]])
                y2 = np.median([left["y2"], right["y2"]])
                x1 = cx - med_w / 2
                x2 = cx + med_w / 2

                # 推荐补货来源：左右邻居中更近的一个；若两边同类/同结构，则可写 consistent neighboring SKU
                recommend = "left_neighbor" if k < num_missing / 2 else "right_neighbor"

                gaps.append({
                    "row": r_idx,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "left": left,
                    "right": right,
                    "recommend": recommend,
                    "gap_width": float(gap),
                    "median_width": float(med_w),
                })

    return gaps


def draw_one(image_path, boxes, gaps, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 商品框：浅蓝色
    for b in boxes:
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=(60, 140, 220), width=2)

    # 空缺候选：红色；推荐邻居：绿色
    for idx, g in enumerate(gaps):
        draw.rectangle([g["x1"], g["y1"], g["x2"], g["y2"]], outline=(255, 40, 40), width=5)

        if g["recommend"] == "left_neighbor":
            src = g["left"]
            label = "Replenish: left neighbor"
        else:
            src = g["right"]
            label = "Replenish: right neighbor"

        draw.rectangle([src["x1"], src["y1"], src["x2"], src["y2"]], outline=(40, 200, 80), width=4)

        tx = max(0, g["x1"])
        ty = max(0, g["y1"] - 18)
        draw.text((tx, ty), f"Gap {idx+1}", fill=(255, 40, 40))
        draw.text((max(0, src["x1"]), max(0, src["y1"] - 18)), label, fill=(40, 160, 60))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--gap-factor", type=float, default=1.8)
    parser.add_argument("--min-gaps", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_root = Path(cfg["processed_root"])
    output_root = Path(cfg["output_root"])

    # 常见对象表路径
    candidates = [
        processed_root / "grid_labels" / f"{args.split}_objects.csv",
        processed_root / "annotations" / f"{args.split}_objects.csv",
        processed_root / f"{args.split}_objects.csv",
    ]

    obj_path = None
    for p in candidates:
        if p.exists():
            obj_path = p
            break

    if obj_path is None:
        raise FileNotFoundError(f"Cannot find objects csv in: {candidates}")

    df = pd.read_csv(obj_path)

    img_col = find_col(df, ["image", "image_name", "filename", "file"])
    x1_col = find_col(df, ["x1", "xmin", "left"])
    y1_col = find_col(df, ["y1", "ymin", "top"])
    x2_col = find_col(df, ["x2", "xmax", "right"])
    y2_col = find_col(df, ["y2", "ymax", "bottom"])

    # 图像根目录
    sku_root = Path(cfg.get("sku_root", cfg.get("sku110k_root", "")))
    possible_img_roots = [
        sku_root / "images",
        sku_root / "SKU110K_fixed" / "images",
        Path(cfg.get("image_root", "")),
    ]

    out_dir = output_root / "figures" / "sku_real_gap_candidates" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0

    for image_name, gdf in tqdm(df.groupby(img_col), desc="Draw real gap candidates"):
        boxes = []

        for _, row in gdf.iterrows():
            x1, y1, x2, y2 = map(float, [row[x1_col], row[y1_col], row[x2_col], row[y2_col]])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "w": x2 - x1,
                "h": y2 - y1,
            })

        if len(boxes) < 10:
            continue

        rows = cluster_rows(boxes)
        gaps = infer_gap_candidates(rows, gap_factor=args.gap_factor)

        if len(gaps) < args.min_gaps:
            continue

        image_path = None
        for root in possible_img_roots:
            p = root / str(image_name)
            if p.exists():
                image_path = p
                break

        if image_path is None:
            continue

        out_path = out_dir / f"{generated:03d}_{Path(image_name).name}"
        draw_one(image_path, boxes, gaps, out_path)

        generated += 1
        if generated >= args.num:
            break

    print("=" * 80)
    print("Generated:", generated)
    print("Output:", out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()