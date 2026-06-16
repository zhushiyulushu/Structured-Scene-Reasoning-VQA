from collections import defaultdict
import numpy as np

from src.reasoning.dsi import box_area, box_iou, overlap_ratio


def bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def interpolate_box(left_box, right_box, ratio=0.5):
    lx, ly = bbox_center(left_box)
    rx, ry = bbox_center(right_box)

    lw = left_box[2] - left_box[0]
    lh = left_box[3] - left_box[1]
    rw = right_box[2] - right_box[0]
    rh = right_box[3] - right_box[1]

    cx = (1 - ratio) * lx + ratio * rx
    cy = (1 - ratio) * ly + ratio * ry
    w = (1 - ratio) * lw + ratio * rw
    h = (1 - ratio) * lh + ratio * rh

    return [
        float(cx - w / 2),
        float(cy - h / 2),
        float(cx + w / 2),
        float(cy + h / 2),
    ]


def heuristic_rule_based(record, gap_factor=1.8, sigma=0.15):
    """
    Rule-based baseline:
    在每个 GT row 内按 x 排序，用水平距离判断大空隙。
    不使用 col continuity，只看几何间距。
    """
    observed = record["observed_objects"]

    row_to_objs = defaultdict(list)
    for obj in observed:
        row_to_objs[int(obj["row"])].append(obj)

    preds = []

    for r, objs in row_to_objs.items():
        objs = sorted(objs, key=lambda x: bbox_center(x["bbox"])[0])

        widths = [obj["bbox"][2] - obj["bbox"][0] for obj in objs]
        if len(widths) < 3:
            continue

        median_w = float(np.median(widths))

        for i in range(len(objs) - 1):
            left = objs[i]
            right = objs[i + 1]

            gap = right["bbox"][0] - left["bbox"][2]

            if gap > gap_factor * median_w:
                pred_box = interpolate_box(left["bbox"], right["bbox"], ratio=0.5)

                max_overlap = 0.0
                for obs in observed:
                    max_overlap = max(max_overlap, overlap_ratio(pred_box, obs["bbox"]))

                if max_overlap < sigma:
                    # heuristic 没有可靠 col，只用相邻 col 中间值
                    pred_col = int(left["col"]) + 1
                    preds.append({
                        "row": int(r),
                        "col": pred_col,
                        "bbox": pred_box,
                        "score": 1.0 - max_overlap,
                        "source": "heuristic_rule"
                    })

    return preds


def planogram_row_alignment(record, sigma=0.15):
    """
    Simplified planogram-style baseline:
    每一行根据平均间距构造规则格子，检测缺失 cell。
    """
    observed = record["observed_objects"]

    row_to_objs = defaultdict(list)
    for obj in observed:
        row_to_objs[int(obj["row"])].append(obj)

    preds = []

    for r, objs in row_to_objs.items():
        objs = sorted(objs, key=lambda x: int(x["col"]))

        if len(objs) < 3:
            continue

        cols = [int(o["col"]) for o in objs]
        min_c, max_c = min(cols), max(cols)

        obj_map = {int(o["col"]): o for o in objs}

        for c in range(min_c, max_c + 1):
            if c in obj_map:
                continue

            # 找最近左/右邻居
            left_candidates = [cc for cc in cols if cc < c]
            right_candidates = [cc for cc in cols if cc > c]

            if not left_candidates or not right_candidates:
                continue

            lc = max(left_candidates)
            rc = min(right_candidates)

            left = obj_map[lc]
            right = obj_map[rc]

            ratio = (c - lc) / float(rc - lc)
            pred_box = interpolate_box(left["bbox"], right["bbox"], ratio=ratio)

            max_overlap = 0.0
            for obs in observed:
                max_overlap = max(max_overlap, overlap_ratio(pred_box, obs["bbox"]))

            if max_overlap < sigma:
                preds.append({
                    "row": int(r),
                    "col": int(c),
                    "bbox": pred_box,
                    "score": 1.0 - max_overlap,
                    "source": "planogram_row"
                })

    return preds