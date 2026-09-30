from pathlib import Path
import json
import torch
from torch.utils.data import Dataset


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_index_to_bin(index, max_index, num_bins):
    """
    将图像内部的 row/col ordinal 映射到固定 canonical bin。
    这样不同图像的拓扑标签具有可比性。
    """
    if max_index <= 0:
        return 0

    ratio = index / float(max_index)
    bin_id = int(round(ratio * (num_bins - 1)))
    bin_id = max(0, min(num_bins - 1, bin_id))
    return bin_id


class GridObjectDataset(Dataset):
    """
    每个 object 是一个训练样本。

    input:
      bbox_norm_xywh: [xc, yc, w, h]

    target:
      row_bin: canonical row bin
      col_bin: canonical col bin

    注意：
      raw row/col 仍然保留在样本里，后面 VQA 和可视化用；
      NGI 训练用 canonical bins，避免跨图像 row/col 含义不一致。
    """
    def __init__(self, jsonl_path, row_bins=16, col_bins=64):
        self.records = read_jsonl(Path(jsonl_path))
        self.samples = []
        self.row_bins = row_bins
        self.col_bins = col_bins

        for rec in self.records:
            objects = rec.get("objects", None)
            if objects is None:
                objects = rec.get("observed_objects", [])

            if len(objects) == 0:
                continue

            # 每张图自己的最大 row
            max_raw_row = max(int(obj["row"]) for obj in objects)

            # 每一行自己的最大 col
            row_to_max_col = {}
            for obj in objects:
                r = int(obj["row"])
                c = int(obj["col"])
                row_to_max_col[r] = max(row_to_max_col.get(r, -1), c)

            for obj in objects:
                bbox = obj["bbox_norm_xywh"]

                raw_row = int(obj["row"])
                raw_col = int(obj["col"])

                max_raw_col = row_to_max_col[raw_row]

                row_bin = normalize_index_to_bin(raw_row, max_raw_row, row_bins)
                col_bin = normalize_index_to_bin(raw_col, max_raw_col, col_bins)

                self.samples.append({
                    "bbox": bbox,
                    "row": row_bin,
                    "col": col_bin,
                    "raw_row": raw_row,
                    "raw_col": raw_col,
                    "image_id": rec["image_id"]
                })

        self.num_rows = row_bins
        self.num_cols = col_bins

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = torch.tensor(s["bbox"], dtype=torch.float32)
        row = torch.tensor(s["row"], dtype=torch.long)
        col = torch.tensor(s["col"], dtype=torch.long)
        return x, row, col