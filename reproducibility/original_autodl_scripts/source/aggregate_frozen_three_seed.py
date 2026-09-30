
from pathlib import Path
import json, numpy as np
ROOT=Path("/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA")
rows=[]

# Seed42 result already validated with the same 20-epoch detector and the frozen
# hyperparameters selected from Seed42 validation.
p42=ROOT/"outputs/f1_recovery/seed42/FINAL_LISTWISE_STRUCTURE_RESCUE_SEED42.json"
j=json.load(open(p42))
rows.append({"seed":42, **{k:j["structure_test"][k] for k in ["Precision","Recall","F1"]}})

for s in [2024,3407]:
    p=ROOT/f"outputs/frozen_multiseed/seed{s}/FINAL_FROZEN_RESULT.json"
    q=json.load(open(p))
    rows.append({"seed":s, **{k:q["test"][k] for k in ["Precision","Recall","F1"]}})

print("THREE SEED RESULTS")
for r in rows: print(r)
A=np.array([[r["Precision"],r["Recall"],r["F1"]] for r in rows],dtype=float)
summary={
    "seeds":[42,2024,3407],
    "Precision_mean":float(A[:,0].mean()),"Precision_std":float(A[:,0].std()),
    "Recall_mean":float(A[:,1].mean()),"Recall_std":float(A[:,1].std()),
    "F1_mean":float(A[:,2].mean()),"F1_std":float(A[:,2].std()),
    "target_reference_f1":0.9504,
    "success_gate_mean_f1_gt_0p9504":bool(A[:,2].mean()>0.9504),
    "all_individual_f1_gt_0p9504":bool((A[:,2]>0.9504).all())
}
out=ROOT/"outputs/frozen_multiseed/THREE_SEED_SUMMARY.json"
out.parent.mkdir(parents=True,exist_ok=True)
json.dump(summary,open(out,"w"),indent=2)
print(json.dumps(summary,indent=2))
print("FROZEN_THREE_SEED_COMPLETE")
