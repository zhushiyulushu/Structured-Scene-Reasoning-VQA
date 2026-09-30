import json
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score

import gate1_v2_today as g


print("========== LOAD TRAIN ==========")

man = g.ego_manifest()
man = man[man["split"] == "train"].reset_index(drop=True)

cache = g.read_jsonl(
    g.OUT / "egocart_ssr_products.jsonl"
)

ngi, _, _ = g.load_ngi()

print("TRAIN_FRAMES =", len(man))
print("NGI_LOAD_PASS")


X_current = []
X_sourceaware = []
Y = []


for i, r in man.iterrows():

    key = str(r["key"])

    rec = cache[key]

    W = rec["width"]
    H = rec["height"]

    prod = [
        x for x in rec["products"]
        if x["score"] >= .30
    ]

    vac = rec["vacancies"]

    geom = g.geom_gaps(
        prod,
        .6,
        1.8,
        .15
    )

    ng = []

    try:
        ng = g.ngi_dsi(
            g.predict_ngi_boxes(
                ngi,
                prod,
                W,
                H
            ),
            W,
            H,
            .15
        )
    except Exception:
        ng = []

    structural = g.nms(
        geom + ng,
        .3
    )

    visual_with_source = [
        {
            "bbox": x["bbox"],
            "score": x["score"],
            "source": "visual",
        }
        for x in vac
    ]

    full = g.nms(
        visual_with_source + structural,
        .3
    )

    compat = float(
        g.structure_compatible(
            prod,
            W,
            H
        )
    )

    row_count = len(
        g.cluster_rows(
            prod,
            .6
        )
    )

    # 当前论文代码的 SSR feature
    f0 = (
        g.stats_boxes(full, W, H)
        + g.stats_boxes(structural, W, H)
        + [
            len(prod),
            row_count,
            compat,
        ]
    )

    # 新的 source-aware SSR feature
    # 仍然使用同一个 logistic calibration head，
    # 只是保留 Visual / Geometry / NGI 三路信息
    f1 = (
        g.stats_boxes(full, W, H)
        + g.stats_boxes(structural, W, H)
        + g.stats_boxes(vac, W, H)
        + g.stats_boxes(geom, W, H)
        + g.stats_boxes(ng, W, H)
        + [
            len(prod),
            row_count,
            compat,
        ]
    )

    X_current.append(f0)
    X_sourceaware.append(f1)

    Y.append(
        int(r["num_gt"] > 0)
    )

    if (i + 1) % 1000 == 0:
        print(
            "FEATURES",
            i + 1,
            "/",
            len(man),
            flush=True
        )


Y = np.asarray(Y, dtype=int)

variants = {
    "current": np.asarray(
        X_current,
        dtype=float
    ),
    "sourceaware": np.asarray(
        X_sourceaware,
        dtype=float
    ),
}


def evaluate_cv(X, y):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    f1s = []
    aps = []
    thrs = []

    for fold, (fit_idx, val_idx) in enumerate(
        skf.split(X, y),
        1
    ):

        mdl = g.train_frame_model(
            X[fit_idx],
            y[fit_idx]
        )

        p_fit = mdl.predict_proba(
            X[fit_idx]
        )[:, 1]

        thr, _ = g.best_thr(
            y[fit_idx],
            p_fit
        )

        p_val = mdl.predict_proba(
            X[val_idx]
        )[:, 1]

        pred = (
            p_val >= thr
        ).astype(int)

        f1 = f1_score(
            y[val_idx],
            pred
        )

        ap = average_precision_score(
            y[val_idx],
            p_val
        )

        f1s.append(float(f1))
        aps.append(float(ap))
        thrs.append(float(thr))

        print(
            "fold",
            fold,
            "F1=",
            round(float(f1), 6),
            "AUPRC=",
            round(float(ap), 6),
            "thr=",
            round(float(thr), 4),
        )

    return {
        "cv_F1_mean": float(
            np.mean(f1s)
        ),
        "cv_F1_std": float(
            np.std(f1s)
        ),
        "cv_AUPRC_mean": float(
            np.mean(aps)
        ),
        "threshold_mean": float(
            np.mean(thrs)
        ),
    }


results = {}

for name, X in variants.items():

    print()
    print(
        "==========",
        name,
        "=========="
    )

    print(
        "FEATURE_DIM =",
        X.shape[1]
    )

    results[name] = evaluate_cv(
        X,
        Y
    )

    print(
        results[name]
    )


selected = max(
    results,
    key=lambda k:
    results[k]["cv_F1_mean"]
)

out = {
    "selection_source": "EgoCart Train only",
    "test_used_for_selection": False,
    "results": results,
    "selected": selected,
}

with open(
    "/root/autodl-tmp/"
    "ego_ssr_traincv_feature_selection.json",
    "w"
) as f:
    json.dump(
        out,
        f,
        indent=2
    )

print()
print(
    "SELECTED =",
    selected
)

print(
    json.dumps(
        out,
        indent=2
    )
)

print(
    "EGO_SSR_TRAINCV_COMPLETE"
)
