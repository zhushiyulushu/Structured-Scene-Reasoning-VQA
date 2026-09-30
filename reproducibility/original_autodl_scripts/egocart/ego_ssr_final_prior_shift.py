import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import gate1_v2_today as g


OUT = Path(
    "/root/autodl-tmp/"
    "ego_ssr_final_prior_shift"
)
OUT.mkdir(
    parents=True,
    exist_ok=True
)


def build_current_ssr_features():

    man = g.ego_manifest()

    cache = g.read_jsonl(
        g.OUT /
        "egocart_ssr_products.jsonl"
    )

    ngi, _, _ = g.load_ngi()

    print("NGI_LOAD_PASS")

    X_train = []
    y_train = []

    X_test = []
    test_keys = []

    total = len(man)

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

        full = g.nms(
            [
                {
                    "bbox": x["bbox"],
                    "score": x["score"],
                    "source": "visual",
                }
                for x in vac
            ]
            + structural,
            .3
        )

        # EXACT current SSR feature.
        sf = (
            g.stats_boxes(
                full,
                W,
                H
            )
            + g.stats_boxes(
                structural,
                W,
                H
            )
            + [
                len(prod),
                len(
                    g.cluster_rows(
                        prod,
                        .6
                    )
                ),
                float(
                    g.structure_compatible(
                        prod,
                        W,
                        H
                    )
                ),
            ]
        )

        split = str(r["split"])

        if split == "train":

            X_train.append(sf)

            # Only Train labels are used here.
            y_train.append(
                int(r["num_gt"] > 0)
            )

        elif split == "test":

            # IMPORTANT:
            # Test labels are deliberately NOT read.
            X_test.append(sf)
            test_keys.append(key)

        if (i + 1) % 1000 == 0:

            print(
                "FEATURES",
                i + 1,
                "/",
                total,
                flush=True
            )

    return (
        np.asarray(
            X_train,
            dtype=float
        ),
        np.asarray(
            y_train,
            dtype=int
        ),
        np.asarray(
            X_test,
            dtype=float
        ),
        test_keys,
    )


print(
    "========== BUILD CURRENT SSR FEATURES =========="
)

Xtr, ytr, Xte, test_keys = (
    build_current_ssr_features()
)

print(
    "TRAIN =",
    Xtr.shape
)

print(
    "TEST_UNLABELED =",
    Xte.shape
)

print(
    "TRAIN_POS_RATE =",
    float(ytr.mean())
)


print()
print(
    "========== OOF TRAIN SCORES =========="
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = np.zeros(
    len(ytr),
    dtype=float
)

for fold, (fit_idx, val_idx) in enumerate(
    skf.split(Xtr, ytr),
    1
):

    mdl = g.train_frame_model(
        Xtr[fit_idx],
        ytr[fit_idx]
    )

    oof[val_idx] = (
        mdl.predict_proba(
            Xtr[val_idx]
        )[:, 1]
    )

    print(
        "OOF_FOLD",
        fold,
        "DONE",
        flush=True
    )


print()
print(
    "========== FINAL TRAIN MODEL =========="
)

mdl = g.train_frame_model(
    Xtr,
    ytr
)

pte = mdl.predict_proba(
    Xte
)[:, 1]


# ------------------------------------------------
# Unlabeled label-prior shift estimation.
#
# Under label-prior shift:
#
# E[p(x)|target]
# =
# q * E[p(x)|y=1,source]
# +
# (1-q) * E[p(x)|y=0,source]
#
# Test labels are NOT used.
# ------------------------------------------------

mu_pos = float(
    oof[ytr == 1].mean()
)

mu_neg = float(
    oof[ytr == 0].mean()
)

mu_target = float(
    pte.mean()
)

den = (
    mu_pos -
    mu_neg
)

if abs(den) < 1e-6:

    raise RuntimeError(
        "PRIOR_SHIFT_ESTIMATION_UNSTABLE"
    )

q_raw = (
    mu_target -
    mu_neg
) / den

q_hat = float(
    np.clip(
        q_raw,
        .05,
        .95
    )
)


print(
    "TRAIN_SCORE_MEAN_POS =",
    mu_pos
)

print(
    "TRAIN_SCORE_MEAN_NEG =",
    mu_neg
)

print(
    "TARGET_SCORE_MEAN_UNLABELED =",
    mu_target
)

print(
    "TARGET_PRIOR_RAW =",
    q_raw
)

print(
    "TARGET_PRIOR_FROZEN =",
    q_hat
)


# ------------------------------------------------
# Choose threshold using ONLY:
# 1. Train OOF labels/scores
# 2. estimated unlabeled target prior
#
# No target labels.
# ------------------------------------------------

best = None

for t in np.linspace(
    .001,
    .999,
    999
):

    zp = (
        oof[ytr == 1] >= t
    )

    zn = (
        oof[ytr == 0] >= t
    )

    tpr = float(
        zp.mean()
    )

    fpr = float(
        zn.mean()
    )

    tp = (
        q_hat *
        tpr
    )

    fp = (
        (1.0 - q_hat) *
        fpr
    )

    fn = (
        q_hat *
        (1.0 - tpr)
    )

    precision = (
        tp /
        max(
            tp + fp,
            1e-12
        )
    )

    recall = tpr

    f1_est = (
        2.0 *
        precision *
        recall /
        max(
            precision + recall,
            1e-12
        )
    )

    if (
        best is None
        or f1_est > best["estimated_target_F1"]
    ):

        best = {
            "threshold": float(t),
            "estimated_target_F1":
                float(f1_est),
            "estimated_TPR":
                float(tpr),
            "estimated_FPR":
                float(fpr),
            "estimated_precision":
                float(precision),
            "estimated_recall":
                float(recall),
        }


print()
print(
    "========== FROZEN ADAPTATION =========="
)

print(
    json.dumps(
        best,
        indent=2
    )
)


freeze = {
    "method":
        "SSR Full-Adapted",
    "adaptation":
        "unlabeled_label_prior_shift",
    "feature_variant":
        "current",
    "selection":
        "threshold estimated from Train OOF "
        "TPR/FPR and unlabeled Test score mean",
    "test_labels_used":
        False,
    "train_positive_rate":
        float(ytr.mean()),
    "mu_pos":
        mu_pos,
    "mu_neg":
        mu_neg,
    "mu_target_unlabeled":
        mu_target,
    "target_prior_raw":
        float(q_raw),
    "target_prior_frozen":
        q_hat,
    **best,
}


with open(
    OUT /
    "SSR_PRIOR_SHIFT_FREEZE.json",
    "w"
) as f:

    json.dump(
        freeze,
        f,
        indent=2
    )


pd.DataFrame({
    "key": test_keys,
    "score": pte,
}).to_csv(
    OUT /
    "SSR_TEST_SCORES_UNLABELED.csv",
    index=False
)


print(
    "SAVED =",
    OUT /
    "SSR_PRIOR_SHIFT_FREEZE.json"
)

print(
    "TEST_LABELS_USED = False"
)

print(
    "SSR_PRIOR_SHIFT_FREEZE_COMPLETE"
)
