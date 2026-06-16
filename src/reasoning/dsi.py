from collections import defaultdict


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = box_area([ix1, iy1, ix2, iy2])
    union = box_area(a) + box_area(b) - inter

    if union <= 0:
        return 0.0
    return inter / union


def overlap_ratio(pred_box, obs_box):
    """
    计算 observed box 覆盖 predicted gap box 的比例。
    用于 Dynamic Topological Slack。
    """
    px1, py1, px2, py2 = pred_box
    ox1, oy1, ox2, oy2 = obs_box

    ix1 = max(px1, ox1)
    iy1 = max(py1, oy1)
    ix2 = min(px2, ox2)
    iy2 = min(py2, oy2)

    inter = box_area([ix1, iy1, ix2, iy2])
    denom = max(box_area(pred_box), 1e-6)
    return inter / denom


def bbox_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def interpolate_gap_box(left_box, right_box, ratio):
    """
    根据左右邻居插值生成 missing proposal。
    ratio: 1/(gap_count+1), 2/(gap_count+1), ...
    """
    lx, ly = bbox_center(left_box)
    rx, ry = bbox_center(right_box)

    lw = left_box[2] - left_box[0]
    lh = left_box[3] - left_box[1]
    rw = right_box[2] - right_box[0]
    rh = right_box[3] - right_box[1]

    cx = (1.0 - ratio) * lx + ratio * rx
    cy = (1.0 - ratio) * ly + ratio * ry

    w = (1.0 - ratio) * lw + ratio * rw
    h = (1.0 - ratio) * lh + ratio * rh

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return [float(x1), float(y1), float(x2), float(y2)]


def infer_missing_by_dsi(record, sigma=0.15):
    """
    Oracle DSI:
    输入 observed_objects 中已有的 GT row/col，
    扫描同一行内 col 的断裂。
    """
    observed = record["observed_objects"]

    row_to_objs = defaultdict(list)
    for obj in observed:
        row_to_objs[int(obj["row"])].append(obj)

    predictions = []

    for r, objs in row_to_objs.items():
        objs = sorted(objs, key=lambda x: int(x["col"]))

        for i in range(len(objs) - 1):
            left = objs[i]
            right = objs[i + 1]

            c_left = int(left["col"])
            c_right = int(right["col"])

            gap_count = c_right - c_left - 1

            if gap_count <= 0:
                continue

            for k in range(1, gap_count + 1):
                missing_col = c_left + k
                ratio = k / float(gap_count + 1)

                pred_box = interpolate_gap_box(
                    left["bbox"],
                    right["bbox"],
                    ratio=ratio
                )

                max_overlap = 0.0
                for obs in observed:
                    max_overlap = max(
                        max_overlap,
                        overlap_ratio(pred_box, obs["bbox"])
                    )

                if max_overlap < sigma:
                    predictions.append({
                        "row": int(r),
                        "col": int(missing_col),
                        "bbox": pred_box,
                        "score": 1.0 - max_overlap,
                        "source": "oracle_dsi"
                    })

    return predictions