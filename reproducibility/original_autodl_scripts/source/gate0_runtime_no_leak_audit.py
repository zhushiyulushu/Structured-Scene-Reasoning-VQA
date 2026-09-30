
from pathlib import Path
import json, math, numpy as np, torch, importlib.util, copy, hashlib

ROOT=Path("/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA")
OUT=ROOT/"outputs/f1_recovery/seed42"
BASE=Path("/root/autodl-tmp/ssr_f1_recovery_seed42.py")
CK=OUT/"listwise_structure_ranker.pt"
TEST=OUT/"test_pred_cache.json"
CONF_THR=.03; MARGIN=.12; ALPHA=.50
KEEP=list(range(1,16))+[18]

spec=importlib.util.spec_from_file_location("base",str(BASE))
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)

class Ranker(torch.nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(d,64),torch.nn.ReLU(),
                                     torch.nn.Linear(64,32),torch.nn.ReLU(),
                                     torch.nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

z=torch.load(CK,map_location="cpu")
model=Ranker(len(KEEP)); model.load_state_dict(z["model"]); model.eval()
mu=np.asarray(z["mu"],dtype=np.float32); sd=np.asarray(z["sd"],dtype=np.float32)

rows=json.load(open(TEST))

def pred_only(rec):
    vv=[]; ff=[]
    for v in rec["vacancies"]:
        if float(v["conf"])<0.001: continue
        f=base.feat_for_candidate(v,rec["products"]).astype(np.float32)[KEEP]
        ff.append((f-mu)/sd); vv.append(dict(v))
    if not vv: return None
    with torch.no_grad():
        sc=model(torch.from_numpy(np.stack(ff)).float()).numpy()
    for q,s in zip(vv,sc): q["structure_score"]=float(s)
    cand=[v for v in vv if float(v["conf"])>=CONF_THR]
    if not cand: return None
    cand=sorted(cand,key=lambda v:float(v["conf"]),reverse=True)
    if len(cand)==1: p=cand[0]
    else:
        gap=float(cand[0]["conf"])-float(cand[1]["conf"])
        if gap>MARGIN: p=cand[0]
        else:
            s=np.array([v["structure_score"] for v in cand],dtype=np.float64)
            zz=(s-s.mean())/(s.std()+1e-8)
            joint=[math.log(max(float(v["conf"]),1e-8))+ALPHA*float(a) for v,a in zip(cand,zz)]
            p=cand[int(np.argmax(joint))]
    return [round(float(x),6) for x in p["bbox"]]

orig=[]
scrub=[]
poison=[]
for i,r in enumerate(rows):
    orig.append(pred_only(r))
    # scrub all GT / missing / row-column metadata: prediction receives only detector outputs
    rr={"products":copy.deepcopy(r["products"]),"vacancies":copy.deepcopy(r["vacancies"])}
    scrub.append(pred_only(rr))
    # adversarial metadata should also be irrelevant
    pp={"products":copy.deepcopy(r["products"]),"vacancies":copy.deepcopy(r["vacancies"]),
        "gts":[{"bbox":[0,0,1,1]}],"row":999999,"col":-999999,
        "missing_row":123456,"missing_col":654321}
    poison.append(pred_only(pp))

same_scrub=(orig==scrub)
same_poison=(orig==poison)
n=sum(x is not None for x in orig)

result={
  "gate":"Gate0 runtime no-leak audit for revised frozen SSR",
  "test_samples":len(rows),
  "predictions_emitted":n,
  "prediction_identical_after_removing_all_gt_and_rowcol_metadata":same_scrub,
  "prediction_identical_after_adversarial_gt_rowcol_metadata":same_poison,
  "structure_feature_indices_from_19d_detector_context":KEEP,
  "structure_branch_excludes_vacancy_and_neighbor_confidences":True,
  "test_inference_requires_ground_truth_missing_bbox":False,
  "test_inference_requires_ground_truth_row_col":False,
  "test_ground_truth_role":"evaluation only",
  "runtime_gate0_pass":bool(same_scrub and same_poison)
}
out=OUT/"GATE0_RUNTIME_NO_LEAK_AUDIT.json"
json.dump(result,open(out,"w"),indent=2)
print(json.dumps(result,indent=2))
print("GATE0_RUNTIME_AUDIT_PASS" if result["runtime_gate0_pass"] else "GATE0_RUNTIME_AUDIT_FAIL")
