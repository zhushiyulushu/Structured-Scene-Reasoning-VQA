
from pathlib import Path
import json, math, random, importlib.util
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA")
OUT = ROOT / "outputs/f1_recovery/seed42"
BASE_SCRIPT = Path("/root/autodl-tmp/ssr_f1_recovery_seed42.py")
IOU_THR = 0.30
SEED = 42
TARGET = 0.9504
DEVICE = torch.device("cpu")  # intentionally CPU: does not occupy the 4090

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_module():
    spec = importlib.util.spec_from_file_location("ssrbase", str(BASE_SCRIPT))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def loadj(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def gtbox(g):
    return g["bbox"] if isinstance(g,dict) else g

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.,ix2-ix1); ih=max(0.,iy2-iy1); inter=iw*ih
    aa=max(0.,ax2-ax1)*max(0.,ay2-ay1); bb=max(0.,bx2-bx1)*max(0.,by2-by1)
    u=aa+bb-inter
    return inter/u if u>0 else 0.

# Base feature has 19 dims:
# 0 vacancy_conf; 1..15 geometry/context; 16/17 neighbor product conf; 18 overlap.
# To force a genuinely complementary structural branch, REMOVE all confidence values:
# keep only geometry/topology/context = 1..15 and 18.
KEEP = list(range(1,16)) + [18]

class Ranker(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): return self.net(x).squeeze(-1)

def prepare_images(rows, base, min_conf=0.001):
    images=[]
    raw=[]
    for rec in rows:
        if len(rec.get("gts",[])) != 1:
            continue
        gt=gtbox(rec["gts"][0])
        cand=[]
        for v in rec["vacancies"]:
            if float(v["conf"]) < min_conf: continue
            f=base.feat_for_candidate(v,rec["products"]).astype(np.float32)[KEEP]
            cand.append((v,f,iou(v["bbox"],gt)))
            raw.append(f)
        if not cand: continue
        positives=[j for j,z in enumerate(cand) if z[2] >= IOU_THR]
        if not positives: continue
        # supervised ranking target: among valid positives choose the one with highest IoU
        target=max(positives,key=lambda j:cand[j][2])
        images.append((cand,target))
    if not raw: raise RuntimeError("No structural candidates")
    return images, np.stack(raw)

def normalize(images,mu,sd):
    out=[]
    for cand,target in images:
        cc=[]
        for v,f,ov in cand:
            cc.append((v,(f-mu)/sd,ov))
        out.append((cc,target))
    return out

def rank_acc(model,images):
    ok=0
    model.eval()
    with torch.no_grad():
        for cand,target in images:
            x=torch.from_numpy(np.stack([z[1] for z in cand])).float().to(DEVICE)
            pred=int(torch.argmax(model(x)).item())
            ok += int(pred==target)
    return ok/max(len(images),1)

def train(train_rows,val_rows,base):
    tr,raw=prepare_images(train_rows,base)
    va,_=prepare_images(val_rows,base)
    mu=raw.mean(0).astype(np.float32); sd=(raw.std(0)+1e-6).astype(np.float32)
    tr=normalize(tr,mu,sd); va=normalize(va,mu,sd)
    model=Ranker(len(KEEP)).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best_acc=-1.; best=None
    print("LISTWISE structural train_images=%d val_images=%d dims=%d" % (len(tr),len(va),len(KEEP)),flush=True)
    for ep in range(1,26):
        random.shuffle(tr)
        model.train(); total=0.
        for cand,target in tr:
            x=torch.from_numpy(np.stack([z[1] for z in cand])).float().to(DEVICE)
            logits=model(x).unsqueeze(0)
            y=torch.tensor([target],dtype=torch.long,device=DEVICE)
            loss=nn.functional.cross_entropy(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        a=rank_acc(model,va)
        print("LISTWISE ep=%02d loss=%.5f val_rank_acc=%.5f" % (ep,total/max(len(tr),1),a),flush=True)
        if a>best_acc:
            best_acc=a
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    torch.save({"model":best,"mu":mu,"sd":sd,"keep":KEEP,"val_rank_acc":best_acc},
               OUT/"listwise_structure_ranker.pt")
    return model,mu,sd,best_acc

def attach(rows,model,mu,sd,base):
    out=[]; model.eval()
    with torch.no_grad():
        for rec in rows:
            vv=[]; feats=[]
            for v in rec["vacancies"]:
                if float(v["conf"]) < 0.001: continue
                f=base.feat_for_candidate(v,rec["products"]).astype(np.float32)[KEEP]
                feats.append((f-mu)/sd); vv.append(dict(v))
            if feats:
                x=torch.from_numpy(np.stack(feats)).float().to(DEVICE)
                sc=model(x).cpu().numpy()
                for q,s in zip(vv,sc): q["structure_score"]=float(s)
            rr=dict(rec); rr["vacancies"]=vv; out.append(rr)
    return out

def choose_conf(rec,thr):
    c=[v for v in rec["vacancies"] if float(v["conf"])>=thr]
    return max(c,key=lambda v:float(v["conf"])) if c else None

def choose_joint(rec,thr,margin,alpha):
    c=[v for v in rec["vacancies"] if float(v["conf"])>=thr]
    if not c: return None
    c=sorted(c,key=lambda v:float(v["conf"]),reverse=True)
    if len(c)==1: return c[0]
    gap=float(c[0]["conf"])-float(c[1]["conf"])
    if gap>margin: return c[0]
    s=np.array([v["structure_score"] for v in c],dtype=np.float64)
    # within-image z-normalization makes alpha comparable
    sz=(s-s.mean())/(s.std()+1e-8)
    vals=[math.log(max(float(v["conf"]),1e-8))+alpha*float(z) for v,z in zip(c,sz)]
    return c[int(np.argmax(vals))]

def evaluate(rows, mode, thr, margin=0., alpha=0.):
    TP=FP=FN=0; changed=corrected=harmed=0
    for rec in rows:
        if len(rec["gts"])!=1: raise RuntimeError("Expected one target per sample")
        gt=gtbox(rec["gts"][0])
        b=choose_conf(rec,thr)
        p=b if mode=="conf" else choose_joint(rec,thr,margin,alpha)
        bok=(b is not None and iou(b["bbox"],gt)>=IOU_THR)
        pok=(p is not None and iou(p["bbox"],gt)>=IOU_THR)
        if p is None: FN+=1
        elif pok: TP+=1
        else: FP+=1; FN+=1
        if b is not None and p is not None and b["bbox"]!=p["bbox"]:
            changed+=1
            if (not bok) and pok: corrected+=1
            if bok and (not pok): harmed+=1
    P=TP/max(TP+FP,1); R=TP/max(TP+FN,1)
    F=2*P*R/max(P+R,1e-12)
    return {"TP":TP,"FP":FP,"FN":FN,"Precision":P,"Recall":R,"F1":F,
            "changed":changed,"corrected":corrected,"harmed":harmed}

def main():
    if not BASE_SCRIPT.exists(): raise FileNotFoundError(BASE_SCRIPT)
    base=load_module()
    train_rows=loadj(OUT/"train_pred_cache.json")
    val_rows=loadj(OUT/"val_pred_cache.json")
    model,mu,sd,racc=train(train_rows,val_rows,base)
    val=attach(val_rows,model,mu,sd,base)

    conf_grid=[0.015,0.02,0.025,0.03,0.035,0.04,0.05,0.07,0.10]
    conf_results=[{"conf_thr":t,**evaluate(val,"conf",t)} for t in conf_grid]
    conf_best=max(conf_results,key=lambda x:(x["F1"],x["Precision"]))

    margin_grid=[0.005,0.01,0.02,0.03,0.05,0.08,0.12,0.20,0.35,0.60,1.0]
    alpha_grid=[0.02,0.05,0.10,0.20,0.35,0.50,0.75,1.0,1.5,2.0]
    grid=[]
    for t in conf_grid:
        for m in margin_grid:
            for a in alpha_grid:
                z=evaluate(val,"joint",t,m,a)
                grid.append({"conf_thr":t,"margin":m,"alpha":a,**z})
    best=max(grid,key=lambda x:(x["F1"],x["Precision"],x["corrected"]-x["harmed"],x["changed"],-x["alpha"]))

    print("CONF_ONLY_VAL",conf_best,flush=True)
    print("STRUCTURE_SELECTED_VAL",best,flush=True)

    gate = bool(best["changed"]>0 and best["corrected"]>best["harmed"] and best["F1"]>conf_best["F1"])
    result={
        "protocol":"confidence-free listwise structural ranker; geometry/topology features only; intervention on visually ambiguous images; all selection on source val",
        "seed":SEED,
        "removed_from_structure_branch":["vacancy_confidence","left_product_confidence","right_product_confidence"],
        "listwise_val_rank_acc":racc,
        "conf_only_val":conf_best,
        "structure_selected_val":best,
        "structural_gate_changed_corrected_and_improved_val":gate,
        "target_reference_f1":TARGET
    }

    # Only touch test if validation establishes an actual structural contribution.
    if gate:
        test_rows=loadj(OUT/"test_pred_cache.json")
        test=attach(test_rows,model,mu,sd,base)
        conf_test=evaluate(test,"conf",conf_best["conf_thr"])
        struct_test=evaluate(test,"joint",best["conf_thr"],best["margin"],best["alpha"])
        result["conf_only_test"]=conf_test
        result["structure_test"]=struct_test
        result["success_gate_structure_test_gt_0p9504"]=bool(struct_test["F1"]>TARGET)
        result["structure_improves_test_over_conf_only"]=bool(struct_test["F1"]>conf_test["F1"])
        print("CONF_ONLY_TEST",conf_test,flush=True)
        print("STRUCTURE_TEST",struct_test,flush=True)
    else:
        result["test_not_run_because_validation_structural_gate_failed"]=True

    with open(OUT/"FINAL_LISTWISE_STRUCTURE_RESCUE_SEED42.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    print("LISTWISE_STRUCTURE_RESCUE_COMPLETE")

if __name__=="__main__":
    main()
