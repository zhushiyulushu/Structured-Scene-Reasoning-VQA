import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

import gate1_v2_today as g


OUT = Path(
    "/root/autodl-tmp/"
    "ego_ssr_final_prior_shift"
)


freeze = json.loads(
    (
        OUT /
        "SSR_PRIOR_SHIFT_FREEZE.json"
    ).read_text()
)

assert (
    freeze["test_labels_used"]
    is False
)


scores = pd.read_csv(
    OUT /
    "SSR_TEST_SCORES_UNLABELED.csv"
)

man = g.ego_manifest()

test = (
    man[
        man["split"] == "test"
    ][
        [
            "key",
            "num_gt",
        ]
    ]
    .copy()
)

test["key"] = (
    test["key"].astype(str)
)

scores["key"] = (
    scores["key"].astype(str)
)

df = test.merge(
    scores,
    on="key",
    how="inner",
    validate="one_to_one"
)

assert (
    len(df)
    ==
    len(test)
    ==
    len(scores)
)

y = (
    df["num_gt"]
    .to_numpy()
    > 0
).astype(int)

p = (
    df["score"]
    .to_numpy(
        dtype=float
    )
)

t = float(
    freeze["threshold"]
)

met = g.metrics_binary(
    y,
    p,
    t
)

met["AUROC"] = float(
    roc_auc_score(
        y,
        p
    )
)

met["AUPRC"] = float(
    average_precision_score(
        y,
        p
    )
)

met["Threshold"] = t

met["Estimated_Target_Prior"] = (
    float(
        freeze[
            "target_prior_frozen"
        ]
    )
)

met["Observed_Test_Positive_Rate"] = (
    float(
        y.mean()
    )
)

met["Success_F1_gt_07702"] = (
    bool(
        met["F1"]
        >
        0.7702
    )
)

met["Nontrivial_BA_gt_050"] = (
    bool(
        met[
            "Balanced_Accuracy"
        ]
        >
        0.50
    )
)


print(
    "========== FINAL SSR ALL =========="
)

for k, v in met.items():

    print(
        k,
        "=",
        v
    )


with open(
    OUT /
    "SSR_PRIOR_SHIFT_TEST_RESULT.json",
    "w"
) as f:

    json.dump(
        met,
        f,
        indent=2
    )


print()
print(
    "PRIMARY_SUCCESS =",
    met[
        "Success_F1_gt_07702"
    ]
)

print(
    "NONTRIVIAL_SUCCESS =",
    (
        met[
            "Success_F1_gt_07702"
        ]
        and
        met[
            "Nontrivial_BA_gt_050"
        ]
    )
)

print(
    "SSR_PRIOR_SHIFT_FINAL_TEST_COMPLETE"
)
