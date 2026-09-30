from pathlib import Path
import argparse
import json
import yaml
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_image_size(image_path: str):
    with Image.open(image_path) as img:
        return img.size  # width, height


def assign_row_col(df_img: pd.DataFrame, image_width: int, image_height: int):
    """
    根据 bbox 中心点生成近似 row/col 标签。
    row：根据 yc 聚类
    col：每一行内部按 xc 从左到右排序
    """
    df = df_img.copy()

    df["w"] = df["x2"] - df["x1"]
    df["h"] = df["y2"] - df["y1"]
    df["xc"] = (df["x1"] + df["x2"]) / 2.0
    df["yc"] = (df["y1"] + df["y2"]) / 2.0

    median_h = float(np.median(df["h"].values))
    eps = max(median_h * 0.65, image_height * 0.012, 8.0)

    y_values = df["yc"].values.reshape(-1, 1)
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(y_values)

    df["row_cluster"] = labels

    # 按每个 cluster 的平均 y 从上到下排序，重新编号为 row=0,1,2...
    cluster_order = (
        df.groupby("row_cluster")["yc"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    row_map = {cluster_id: i for i, cluster_id in enumerate(cluster_order)}
    df["row"] = df["row_cluster"].map(row_map).astype(int)

    # 每一行内按 x 从左到右编号为 col
    df["col"] = -1
    for r in sorted(df["row"].unique()):
        idxs = df[df["row"] == r].sort_values("xc").index.tolist()
        for c, idx in enumerate(idxs):
            df.loc[idx, "col"] = c

    df["col"] = df["col"].astype(int)

    objects = []
    for obj_idx, row in enumerate(df.itertuples(index=False)):
        x1 = float(row.x1)
        y1 = float(row.y1)
        x2 = float(row.x2)
        y2 = float(row.y2)

        objects.append({
            "object_id": obj_idx,
            "bbox": [x1, y1, x2, y2],
            "bbox_xywh": [
                float((x1 + x2) / 2.0),
                float((y1 + y2) / 2.0),
                float(x2 - x1),
                float(y2 - y1)
            ],
            "bbox_norm_xywh": [
                float(((x1 + x2) / 2.0) / image_width),
                float(((y1 + y2) / 2.0) / image_height),
                float((x2 - x1) / image_width),
                float((y2 - y1) / image_height)
            ],
            "row": int(row.row),
            "col": int(row.col),
            "class_name": str(row.class_name)
        })

    return objects


def process_split(split: str, cfg: dict, min_objects: int = 20):
    in_csv = Path(cfg["processed_root"]) / "annotations" / f"{split}_objects.csv"
    out_dir = Path(cfg["processed_root"]) / "grid_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{split}_grid.jsonl"

    if not in_csv.exists():
        raise FileNotFoundError(f"找不到输入文件：{in_csv}，请先运行 convert_sku110k.py")

    df = pd.read_csv(in_csv)

    total_images = df["image_id"].nunique()
    kept = 0
    skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for image_id, df_img in tqdm(df.groupby("image_id"), total=total_images, desc=f"Building grid {split}"):
            if len(df_img) < min_objects:
                skipped += 1
                continue

            image_path = df_img["image_path"].iloc[0]
            if not Path(image_path).exists():
                skipped += 1
                continue

            try:
                image_width, image_height = get_image_size(image_path)
            except Exception:
                skipped += 1
                continue

            objects = assign_row_col(df_img, image_width, image_height)

            if len(objects) < min_objects:
                skipped += 1
                continue

            num_rows = max(obj["row"] for obj in objects) + 1
            max_cols = max(obj["col"] for obj in objects) + 1

            record = {
                "image_id": image_id,
                "split": split,
                "image_path": image_path,
                "width": image_width,
                "height": image_height,
                "num_objects": len(objects),
                "num_rows": num_rows,
                "max_cols": max_cols,
                "objects": objects
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 80)
    print(f"[完成] {split} grid labels")
    print(f"输出文件：{out_jsonl}")
    print(f"保留图片：{kept}")
    print(f"跳过图片：{skipped}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"])
    parser.add_argument("--min-objects", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, cfg, min_objects=args.min_objects)


if __name__ == "__main__":
    main()