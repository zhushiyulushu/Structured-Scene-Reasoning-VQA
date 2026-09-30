
from pathlib import Path
import json, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT=Path("/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA")
SRC=ROOT/"outputs/f1_recovery/seed42"
TRAIN=SRC/"train_pred_cache.json"
VAL=SRC/"val_pred_cache.json"
TEST=SRC/"test_pred_cache.json"
FULL_CK=SRC/"listwise_structure_ranker.pt"
FROZEN_JSON=SRC/"FINAL_LISTWISE_STRUCTURE_RESCUE_SEED42.json"
OUT=ROOT/"outputs/gate3_true_end2end_ablation_seed42"

SEED=42
TRAIN_MIN_CONF=0.001
CONF=0.03
MARGIN=0.12
ALPHA=0.50
IOU=0.30
EPOCHS=25

# Exact confidence-free structural branch used by the frozen method.
FULL_KEEP=list(range(1,16))+[18]
GEOM_KEEP=[1,2,3,4,5]
TOPO_KEEP=list(range(6,16))+[18]
NO_OVERLAP_KEEP=list(range(1,16))

EXPECTED={"TP":2650,"FP":40,"FN":212,"F1":0.9546109510086455}

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def loadj(p):
    with open(p,"r") as f: return json.load(f)

def gtbox(g):
    return g["bbox"] if isinstance(g,dict) else g

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.,ix2-ix1); ih=max(0.,iy2-iy1)
    inter=iw*ih
    aa=max(0.,ax2-ax1)*max(0.,ay2-ay1)
    bb=max(0.,bx2-bx1)*max(0.,by2-by1)
    u=aa+bb-inter
    return inter/u if u>0 else 0.

def feature19(v,products,W=640.,H=640.,prod_thr=.30):
    x1,y1,x2,y2=v["bbox"]
    cx=(x1+x2)/2.; cy=(y1+y2)/2.
    w=max(x2-x1,1.); h=max(y2-y1,1.)
    ps=[p for p in products if float(p["conf"])>=prod_thr]
    local=[]
    for p in ps:
        a,b,c,d=p["bbox"]
        pcx=(a+c)/2.; pcy=(b+d)/2.
        pw=max(c-a,1.); ph=max(d-b,1.)
        if abs(pcy-cy)<=max(h,ph)*.75:
            local.append((p,pcx,pcy,pw,ph))
    left=[q for q in local if q[1]<cx]
    right=[q for q in local if q[1]>cx]
    L=max(left,key=lambda q:q[1]) if left else None
    R=min(right,key=lambda q:q[1]) if right else None
    avgw=np.mean([q[3] for q in local]) if local else w
    avgh=np.mean([q[4] for q in local]) if local else h
    maxov=max([iou(v["bbox"],p["bbox"]) for p in ps],default=0.)
    if L is not None:
        ldx=(cx-L[1])/max(avgw,1.); ldy=abs(cy-L[2])/max(avgh,1.)
        lconf=float(L[0]["conf"])
    else:
        ldx=5.; ldy=5.; lconf=0.
    if R is not None:
        rdx=(R[1]-cx)/max(avgw,1.); rdy=abs(cy-R[2])/max(avgh,1.)
        rconf=float(R[0]["conf"])
    else:
        rdx=5.; rdy=5.; rconf=0.
    both=1. if (L is not None and R is not None) else 0.
    if L is not None and R is not None:
        mid=(L[1]+R[1])/2.
        midres=abs(cx-mid)/max(avgw,1.)
        span=(R[1]-L[1])/max(avgw,1.)
        wr=w/max((L[3]+R[3])/2.,1.)
        hr=h/max((L[4]+R[4])/2.,1.)
    else:
        midres=5.; span=0.
        wr=w/max(avgw,1.); hr=h/max(avgh,1.)
    return np.array([
        float(v["conf"]),                  # 0 excluded from learned structural branch
        cx/W,cy/H,w/W,h/H,(w*h)/(W*H),    # 1..5 global geometry
        min(len(local),50)/50.,both,       # 6..7 local topology
        min(ldx,5)/5.,min(rdx,5)/5.,      # 8..9
        min(ldy,5)/5.,min(rdy,5)/5.,      # 10..11
        min(midres,5)/5.,min(span,10)/10.,# 12..13
        min(wr,5)/5.,min(hr,5)/5.,        # 14..15
        lconf,rconf,                       # 16..17 excluded
        maxov                              # 18 overlap consistency
    ],dtype=np.float32)

class Ranker(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): return self.net(x).squeeze(-1)

def build_rank_groups(rows,keep):
    groups=[]; raw=[]
    for r in rows:
        if len(r.get("gts",[]))!=1: continue
        gt=gtbox(r["gts"][0])
        cand=[]
        for v in r.get("vacancies",[]):
            if float(v["conf"])<TRAIN_MIN_CONF: continue
            f=feature19(v,r.get("products",[]))[keep]
            ov=iou(v["bbox"],gt)
            cand.append((f,ov)); raw.append(f)
        pos=[i for i,(_,ov) in enumerate(cand) if ov>=IOU]
        if cand and pos:
            t=max(pos,key=lambda i:cand[i][1])
            groups.append((cand,t))
    if not raw: raise RuntimeError("No ranker training features.")
    return groups,np.stack(raw)

def normalize_groups(groups,mu,sd):
    return [([(f-mu)/sd for f,_ in cand],t) for cand,t in groups]

def rankacc(model,groups):
    if not groups:return 0.
    ok=0; model.eval()
    with torch.no_grad():
        for xs,t in groups:
            x=torch.from_numpy(np.stack(xs)).float()
            ok += int(int(torch.argmax(model(x)))==t)
    return ok/len(groups)

def train_ranker(train_rows,val_rows,keep):
    seed_all()
    tr,raw=build_rank_groups(train_rows,keep)
    va,_=build_rank_groups(val_rows,keep)
    mu=raw.mean(0).astype(np.float32)
    sd=(raw.std(0)+1e-6).astype(np.float32)
    tr=normalize_groups(tr,mu,sd); va=normalize_groups(va,mu,sd)
    m=Ranker(len(keep))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-1.; state=None; best_ep=-1
    for ep in range(EPOCHS):
        random.shuffle(tr); m.train()
        for xs,t in tr:
            x=torch.from_numpy(np.stack(xs)).float()
            y=torch.tensor([t],dtype=torch.long)
            logits=m(x).unsqueeze(0)
            loss=nn.functional.cross_entropy(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
        a=rankacc(m,va)
        if a>best:
            best=a; best_ep=ep+1
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    return m,mu,sd,best,best_ep

def load_frozen_full():
    ck=torch.load(FULL_CK,map_location="cpu")
    m=Ranker(len(FULL_KEEP)); m.load_state_dict(ck["model"]); m.eval()
    return m,np.asarray(ck["mu"],np.float32),np.asarray(ck["sd"],np.float32),float(ck.get("val_rank_acc",np.nan))

def zscores(a):
    a=np.asarray(a,dtype=np.float64)
    return (a-a.mean())/(a.std()+1e-8)

def learned_scores(r,cands,model,mu,sd,keep):
    X=np.stack([(feature19(v,r.get("products",[]))[keep]-mu)/sd for v in cands])
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).cpu().numpy()

def heuristic_scores(r,cands):
    ss=[]
    for v in cands:
        f=feature19(v,r.get("products",[]))
        both=f[7]; ldx,rdx=f[8],f[9]; ldy,rdy=f[10],f[11]
        mid=f[12]; span=f[13]; wr,hr=f[14],f[15]; ov=f[18]
        ss.append(
            2.0*both - 1.5*mid - 0.5*(ldy+rdy)
            - 0.35*abs(ldx-rdx) - 0.25*abs(span-0.2)
            - 0.35*abs(wr-0.2) - 0.35*abs(hr-0.2) - 1.5*ov
        )
    return np.asarray(ss,dtype=np.float64)

def choose(r,mode,model=None,mu=None,sd=None,keep=None):
    # Every main ablation uses the SAME frozen proposal confidence threshold.
    cands=[v for v in r.get("vacancies",[]) if float(v["conf"])>=CONF]
    cands=sorted(cands,key=lambda v:float(v["conf"]),reverse=True)
    if not cands:return None
    if len(cands)==1:return cands[0]

    if mode=="visual_only":
        return cands[0]

    if mode=="heuristic_gated":
        if float(cands[0]["conf"])-float(cands[1]["conf"])>MARGIN:
            return cands[0]
        s=zscores(heuristic_scores(r,cands))
        joint=[math.log(max(float(v["conf"]),1e-8))+ALPHA*float(z) for v,z in zip(cands,s)]
        return cands[int(np.argmax(joint))]

    if mode=="learned_gated":
        if float(cands[0]["conf"])-float(cands[1]["conf"])>MARGIN:
            return cands[0]
        s=zscores(learned_scores(r,cands,model,mu,sd,keep))
        joint=[math.log(max(float(v["conf"]),1e-8))+ALPHA*float(z) for v,z in zip(cands,s)]
        return cands[int(np.argmax(joint))]

    if mode=="learned_no_gate":
        s=zscores(learned_scores(r,cands,model,mu,sd,keep))
        joint=[math.log(max(float(v["conf"]),1e-8))+ALPHA*float(z) for v,z in zip(cands,s)]
        return cands[int(np.argmax(joint))]

    raise ValueError(mode)

def eval_e2e(rows,mode,model=None,mu=None,sd=None,keep=None):
    TP=FP=FN=0; N=0; emitted=0
    for r in rows:
        if len(r.get("gts",[]))!=1: continue
        N+=1; gt=gtbox(r["gts"][0])
        p=choose(r,mode,model,mu,sd,keep)
        if p is None:
            FN+=1
            continue
        emitted+=1
        if iou(p["bbox"],gt)>=IOU:
            TP+=1
        else:
            FP+=1; FN+=1
    P=TP/max(TP+FP,1); R=TP/max(TP+FN,1)
    F=2*P*R/max(P+R,1e-12)
    return {"N":N,"Predictions":emitted,"TP":TP,"FP":FP,"FN":FN,"Precision":P,"Recall":R,"F1":F}

def main():
    seed_all()
    for p in [TRAIN,VAL,TEST,FULL_CK,FROZEN_JSON]:
        if not p.exists(): raise FileNotFoundError(str(p))
    train=loadj(TRAIN); val=loadj(VAL); test=loadj(TEST)
    OUT.mkdir(parents=True,exist_ok=True)

    # Exact frozen full checkpoint.
    fm,fmu,fsd,frozen_rankacc=load_frozen_full()

    # Reproduction gate: this MUST exactly reproduce the already-frozen seed42 main result.
    full=eval_e2e(test,"learned_gated",fm,fmu,fsd,FULL_KEEP)
    print("FROZEN FULL REPRODUCTION =",full,flush=True)
    ok=(full["TP"]==EXPECTED["TP"] and full["FP"]==EXPECTED["FP"] and
        full["FN"]==EXPECTED["FN"] and abs(full["F1"]-EXPECTED["F1"])<1e-12)
    if not ok:
        raise RuntimeError(
            "ABORT: evaluation logic does not exactly reproduce frozen seed42 main result. "
            f"Expected {EXPECTED}, got {full}"
        )
    print("FROZEN_FULL_REPRODUCTION_PASS",flush=True)

    # Retrain ONLY the changed learned structural variants, using exactly the same
    # 25-epoch schedule and source-validation model-selection criterion.
    print("TRAIN w/o Topology Context (geometry-only ranker)",flush=True)
    gm,gmu,gsd,gacc,gep=train_ranker(train,val,GEOM_KEEP)
    print("  val rank acc",gacc,"epoch",gep,flush=True)

    print("TRAIN w/o Global Geometry (topology-only ranker)",flush=True)
    tm,tmu,tsd,tacc,tep=train_ranker(train,val,TOPO_KEEP)
    print("  val rank acc",tacc,"epoch",tep,flush=True)

    print("TRAIN w/o Overlap Consistency",flush=True)
    om,omu,osd,oacc,oep=train_ranker(train,val,NO_OVERLAP_KEEP)
    print("  val rank acc",oacc,"epoch",oep,flush=True)

    variants=[
        ("w/o Learnable Structural Ranking","heuristic_gated",None,None,None,None,np.nan,np.nan),
        ("w/o Topology Context","learned_gated",gm,gmu,gsd,GEOM_KEEP,gacc,gep),
        ("w/o Global Geometry","learned_gated",tm,tmu,tsd,TOPO_KEEP,tacc,tep),
        ("w/o Overlap Consistency","learned_gated",om,omu,osd,NO_OVERLAP_KEEP,oacc,oep),
        ("w/o Ambiguity-aware Gating","learned_no_gate",fm,fmu,fsd,FULL_KEEP,frozen_rankacc,np.nan),
        ("w/o Structural Branch","visual_only",None,None,None,None,np.nan,np.nan),
        ("Full SSR","learned_gated",fm,fmu,fsd,FULL_KEEP,frozen_rankacc,np.nan),
    ]

    rows=[]
    for name,mode,m,mu,sd,keep,racc,bep in variants:
        va=eval_e2e(val,mode,m,mu,sd,keep)
        te=eval_e2e(test,mode,m,mu,sd,keep)
        rows.append({
            "Variant":name,
            "Seed":SEED,
            "Detector":"same frozen seed42 20-epoch detector cache",
            "StructuralTrainEpochs":0 if m is None else (0 if name in ["w/o Ambiguity-aware Gating","Full SSR"] else EPOCHS),
            "ValRankAcc":racc,
            "BestEpoch":bep,
            "Val_Precision":va["Precision"],"Val_Recall":va["Recall"],"Val_F1":va["F1"],
            "Test_TP":te["TP"],"Test_FP":te["FP"],"Test_FN":te["FN"],
            "Test_Precision":te["Precision"],"Test_Recall":te["Recall"],"Test_F1":te["F1"],
            "F1_Drop_vs_Full":full["F1"]-te["F1"],
        })

    df=pd.DataFrame(rows)
    csvp=OUT/"gate3_true_end2end_ablation_seed42.csv"
    df.to_csv(csvp,index=False,encoding="utf-8-sig")

    protocol={
        "seed":SEED,
        "purpose":"true controlled end-to-end ablation of the frozen revised SSR pipeline",
        "dataset":"same SKU-110K train/validation/test split used by the main frozen experiment",
        "test_scope":"entire seed42 test split; no hard/stress subset",
        "primary_metric":"same end-to-end missing-region Precision/Recall/F1 as the main result",
        "iou":IOU,
        "confidence_threshold":CONF,
        "ambiguity_margin":MARGIN,
        "structure_alpha":ALPHA,
        "detector":"same already-trained 20-epoch seed42 detector outputs for every row because none of these ablations changes the detector architecture or detector training",
        "ranker_training":"every ablation that changes ranker inputs is retrained for exactly 25 epochs on the same source train cache; best epoch selected by the same source-validation top-1 rank-accuracy criterion",
        "full_model":"uses the exact frozen ranker checkpoint from the main experiment, not a newly favorable retraining",
        "reproduction_gate":{
            "expected":EXPECTED,
            "observed":full,
            "pass":True
        },
        "variant_definitions":{
            "w/o Learnable Structural Ranking":"replace the learned structural ranker with a fixed hand-crafted structural compatibility score while retaining the same proposal threshold, ambiguity gate, and visual-structural fusion",
            "w/o Topology Context":"retrain the structural ranker using only global proposal geometry features [1..5]; same gate/fusion/evaluation",
            "w/o Global Geometry":"retrain the structural ranker using only topology/local-context features [6..15,18]; same gate/fusion/evaluation",
            "w/o Overlap Consistency":"retrain the ranker after removing overlap-consistency feature 18; same gate/fusion/evaluation",
            "w/o Ambiguity-aware Gating":"use the exact full structural ranker and same visual-structural fusion, but apply reranking to every multi-candidate case instead of only visually ambiguous cases",
            "w/o Structural Branch":"remove structural ranking entirely; use frozen visual-confidence Top1 under the same proposal threshold",
            "Full SSR":"exact frozen seed42 pipeline"
        },
        "important_note":"No test subset, threshold, or evaluation rule is changed across rows. Do not tune these after observing ablation results."
    }
    with open(OUT/"GATE3_TRUE_ABLATION_PROTOCOL.json","w") as f:
        json.dump(protocol,f,indent=2)

    print("\n========== TRUE END-TO-END ABLATION ==========")
    print(df[["Variant","ValRankAcc","Test_Precision","Test_Recall","Test_F1","F1_Drop_vs_Full"]].to_string(index=False))
    print("\nGATE3_TRUE_END2END_COMPLETE")
    print("CSV =",csvp)
    print("PROTOCOL =",OUT/"GATE3_TRUE_ABLATION_PROTOCOL.json")

if __name__=="__main__":
    main()
