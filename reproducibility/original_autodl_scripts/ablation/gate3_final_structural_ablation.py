
from pathlib import Path
import json, random
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
OUT=ROOT/"outputs/gate3_final_structural_ablation_seed42"

SEED=42
TRAIN_MIN_CONF=0.001
EVAL_MIN_CONF=0.001
MIN_CANDIDATES=3
IOU_THR=0.30
EPOCHS=25

FULL=list(range(1,16))+[18]
GEOM=[1,2,3,4,5]
TOPO=list(range(6,16))+[18]
NO_BILATERAL=sorted(set(FULL)-set([7,8,9,10,11]))
NO_GAPREG=sorted(set(FULL)-set([12,13,14,15]))
NO_OVERLAP=sorted(set(FULL)-set([18]))

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_json(p):
    with open(p,"r") as f: return json.load(f)

def gtbox(g):
    return g["bbox"] if isinstance(g,dict) else g

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.,ix2-ix1); ih=max(0.,iy2-iy1); inter=iw*ih
    aa=max(0.,ax2-ax1)*max(0.,ay2-ay1); bb=max(0.,bx2-bx1)*max(0.,by2-by1)
    u=aa+bb-inter
    return inter/u if u>0 else 0.0

def feat19(v, products, W=640., H=640., prod_thr=.30):
    x1,y1,x2,y2=v["bbox"]; cx=(x1+x2)/2.; cy=(y1+y2)/2.
    w=max(x2-x1,1.); h=max(y2-y1,1.)
    ps=[p for p in products if float(p["conf"])>=prod_thr]
    local=[]
    for p in ps:
        a,b,c,d=p["bbox"]; pcx=(a+c)/2.; pcy=(b+d)/2.
        pw=max(c-a,1.); ph=max(d-b,1.)
        if abs(pcy-cy)<=max(h,ph)*.75: local.append((p,pcx,pcy,pw,ph))
    left=[q for q in local if q[1]<cx]; right=[q for q in local if q[1]>cx]
    L=max(left,key=lambda q:q[1]) if left else None
    R=min(right,key=lambda q:q[1]) if right else None
    avgw=np.mean([q[3] for q in local]) if local else w
    avgh=np.mean([q[4] for q in local]) if local else h
    maxov=max([iou(v["bbox"],p["bbox"]) for p in ps],default=0.)
    if L is not None:
        ldx=(cx-L[1])/max(avgw,1.); ldy=abs(cy-L[2])/max(avgh,1.); lconf=float(L[0]["conf"])
    else: ldx=5.; ldy=5.; lconf=0.
    if R is not None:
        rdx=(R[1]-cx)/max(avgw,1.); rdy=abs(cy-R[2])/max(avgh,1.); rconf=float(R[0]["conf"])
    else: rdx=5.; rdy=5.; rconf=0.
    both=1. if (L is not None and R is not None) else 0.
    if L is not None and R is not None:
        mid=(L[1]+R[1])/2.; midres=abs(cx-mid)/max(avgw,1.); span=(R[1]-L[1])/max(avgw,1.)
        wr=w/max((L[3]+R[3])/2.,1.); hr=h/max((L[4]+R[4])/2.,1.)
    else:
        midres=5.; span=0.; wr=w/max(avgw,1.); hr=h/max(avgh,1.)
    return np.array([
        float(v["conf"]),cx/W,cy/H,w/W,h/H,(w*h)/(W*H),
        min(len(local),50)/50.,both,
        min(ldx,5)/5.,min(rdx,5)/5.,min(ldy,5)/5.,min(rdy,5)/5.,
        min(midres,5)/5.,min(span,10)/10.,min(wr,5)/5.,min(hr,5)/5.,
        lconf,rconf,maxov
    ],dtype=np.float32)

class Ranker(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def build_train_groups(rows, keep):
    groups=[]; raw=[]
    for r in rows:
        if len(r.get("gts",[]))!=1: continue
        gt=gtbox(r["gts"][0]); cand=[]
        for v in r.get("vacancies",[]):
            if float(v["conf"])<TRAIN_MIN_CONF: continue
            f=feat19(v,r.get("products",[]))[keep]; ov=iou(v["bbox"],gt)
            cand.append((f,ov)); raw.append(f)
        pos=[i for i,(_,ov) in enumerate(cand) if ov>=IOU_THR]
        if cand and pos:
            target=max(pos,key=lambda i:cand[i][1]); groups.append((cand,target))
    if not raw: raise RuntimeError("No ranker training features.")
    return groups,np.stack(raw)

def normalize_groups(groups,mu,sd):
    return [([(f-mu)/sd for f,_ in cand],target) for cand,target in groups]

def rank_acc(model,groups):
    if not groups: return 0.
    ok=0; model.eval()
    with torch.no_grad():
        for xs,target in groups:
            x=torch.from_numpy(np.stack(xs)).float()
            ok+=int(int(torch.argmax(model(x)))==target)
    return ok/len(groups)

def train_ranker(train_rows,val_rows,keep):
    seed_all(); tr,raw=build_train_groups(train_rows,keep); va,_=build_train_groups(val_rows,keep)
    mu=raw.mean(0).astype(np.float32); sd=(raw.std(0)+1e-6).astype(np.float32)
    tr=normalize_groups(tr,mu,sd); va=normalize_groups(va,mu,sd)
    model=Ranker(len(keep)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-1.; best_state=None; best_epoch=-1
    for ep in range(EPOCHS):
        random.shuffle(tr); model.train()
        for xs,target in tr:
            x=torch.from_numpy(np.stack(xs)).float(); y=torch.tensor([target],dtype=torch.long)
            logits=model(x).unsqueeze(0); loss=nn.functional.cross_entropy(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
        a=rank_acc(model,va)
        if a>best:
            best=a; best_epoch=ep+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model,mu,sd,best,best_epoch

def load_full():
    ck=torch.load(FULL_CK,map_location="cpu")
    model=Ranker(len(FULL)); model.load_state_dict(ck["model"]); model.eval()
    return model,np.asarray(ck["mu"],dtype=np.float32),np.asarray(ck["sd"],dtype=np.float32),float(ck.get("val_rank_acc",np.nan))

def heuristic_score(f):
    both=f[7]; ldx,rdx=f[8],f[9]; ldy,rdy=f[10],f[11]
    mid=f[12]; span=f[13]; wr,hr=f[14],f[15]; ov=f[18]
    return 2.0*both-1.5*mid-0.5*(ldy+rdy)-0.35*abs(ldx-rdx)-0.25*abs(span-0.2)-0.35*abs(wr-0.2)-0.35*abs(hr-0.2)-1.5*ov

def build_stress_cases(rows):
    cases=[]
    for r in rows:
        if len(r.get("gts",[]))!=1: continue
        cands=[v for v in r.get("vacancies",[]) if float(v["conf"])>=EVAL_MIN_CONF]
        if len(cands)>=MIN_CANDIDATES: cases.append((r,cands))
    return cases

def evaluate(rows, keep=None, model=None, mu=None, sd=None, heuristic=False):
    cases=build_stress_cases(rows)
    n_all=len(cases); n_rankable=0; correct_all=0; correct_rankable=0; cand_counts=[]
    for r,cands in cases:
        gt=gtbox(r["gts"][0]); ovs=[iou(v["bbox"],gt) for v in cands]
        rankable=any(v>=IOU_THR for v in ovs); n_rankable+=int(rankable); cand_counts.append(len(cands))
        if heuristic:
            scores=[heuristic_score(feat19(v,r.get("products",[]))) for v in cands]
            pred=int(np.argmax(scores))
        else:
            X=np.stack([(feat19(v,r.get("products",[]))[keep]-mu)/sd for v in cands])
            with torch.no_grad(): scores=model(torch.from_numpy(X).float()).cpu().numpy()
            pred=int(np.argmax(scores))
        good=ovs[pred]>=IOU_THR
        correct_all+=int(good)
        if rankable: correct_rankable+=int(good)
    all_f1=correct_all/max(n_all,1); rankable_f1=correct_rankable/max(n_rankable,1)
    return {
        "Stress_N":n_all,"Rankable_N":n_rankable,"ProposalRecall":n_rankable/max(n_all,1),
        "CandidateF1_AllStress":all_f1,"StructuralF1_Rankable":rankable_f1,
        "Correct_AllStress":correct_all,"Correct_Rankable":correct_rankable,
        "MeanCandidates":float(np.mean(cand_counts)) if cand_counts else 0.,
        "MedianCandidates":float(np.median(cand_counts)) if cand_counts else 0.,
        "MaxCandidates":int(max(cand_counts)) if cand_counts else 0
    }

def main():
    seed_all()
    for p in [TRAIN,VAL,TEST,FULL_CK]:
        if not p.exists(): raise FileNotFoundError(str(p))
    train=load_json(TRAIN); val=load_json(VAL); test=load_json(TEST)
    OUT.mkdir(parents=True,exist_ok=True)
    fm,fmu,fsd,frozen_val_acc=load_full()

    variants=[{"name":"w/o Learnable Structural Ranking","mode":"heuristic"}]
    specs=[
        ("w/o Topology Context",GEOM),
        ("w/o Global Geometry",TOPO),
        ("w/o Bilateral Neighbor Context",NO_BILATERAL),
        ("w/o Gap Regularity",NO_GAPREG),
        ("w/o Overlap Consistency",NO_OVERLAP),
    ]
    for name,keep in specs:
        print(f"TRAIN {name}: {len(keep)} features",flush=True)
        model,mu,sd,best,bep=train_ranker(train,val,keep)
        print(f"  source-val rank acc={best:.6f} @ epoch {bep}",flush=True)
        variants.append({"name":name,"mode":"learned","keep":keep,"model":model,"mu":mu,"sd":sd,"val_rank_acc":best,"best_epoch":bep})
    variants.append({"name":"Full Learnable Structural Ranker","mode":"learned","keep":FULL,"model":fm,"mu":fmu,"sd":fsd,"val_rank_acc":frozen_val_acc,"best_epoch":np.nan})

    rows=[]
    for v in variants:
        if v["mode"]=="heuristic":
            va=evaluate(val,heuristic=True); te=evaluate(test,heuristic=True); vra=np.nan; bep=np.nan
        else:
            va=evaluate(val,v["keep"],v["model"],v["mu"],v["sd"],False)
            te=evaluate(test,v["keep"],v["model"],v["mu"],v["sd"],False)
            vra=v["val_rank_acc"]; bep=v["best_epoch"]
        rows.append({
            "Variant":v["name"],"ValRankAcc_Selected":vra,"BestEpoch":bep,
            "Val_Stress_N":va["Stress_N"],"Val_Rankable_N":va["Rankable_N"],"Val_StructuralF1":va["StructuralF1_Rankable"],
            "Test_Stress_N":te["Stress_N"],"Test_Rankable_N":te["Rankable_N"],"Test_ProposalRecall":te["ProposalRecall"],
            "Test_StructuralF1":te["StructuralF1_Rankable"],"Test_AllStressF1":te["CandidateF1_AllStress"],
            "Test_MeanCandidates":te["MeanCandidates"],"Test_MaxCandidates":te["MaxCandidates"],
        })

    df=pd.DataFrame(rows)
    full_f1=float(df.loc[df["Variant"]=="Full Learnable Structural Ranker","Test_StructuralF1"].iloc[0])
    df["StructuralF1_Drop_vs_Full"]=full_f1-df["Test_StructuralF1"]
    csvp=OUT/"gate3_final_structural_ablation.csv"
    df.to_csv(csvp,index=False,encoding="utf-8-sig")

    protocol={
        "seed":SEED,
        "dataset":"same SKU-110K train/validation/test split and same frozen detector-prediction caches as the main experiment",
        "purpose":"stage-specific ablation of the learned structural decision module",
        "candidate_source":"same frozen detector cache as the main experiment",
        "training_candidate_floor":TRAIN_MIN_CONF,
        "evaluation_candidate_floor":EVAL_MIN_CONF,
        "minimum_candidates_per_case":MIN_CANDIDATES,
        "case_selection_uses_gt":False,
        "rankable_condition":"GT is used only to identify whether the frozen detector included at least one correct proposal, and to score the selected proposal; GT never enters inference.",
        "primary_metric":"Test_StructuralF1 = top-1 structural candidate-selection F1 on rankable proposal-rich cases. Exactly one proposal is selected for one target, so precision=recall=F1=selection accuracy.",
        "secondary_metric":"Test_AllStressF1 includes detector-unrecoverable stress cases and is reported for completeness.",
        "iou":IOU_THR,
        "ranker_epochs":EPOCHS,
        "ranker_optimizer":"AdamW(lr=1e-3, weight_decay=1e-4)",
        "model_selection":"same source-validation top-1 rank accuracy criterion for every learned ablation",
        "full_model":"exact frozen listwise structural ranker used by the final main method",
        "important_note":"Protocol fixed before reading these results. Do not retune thresholds, candidate count, IoU, or feature groups to target a desired numeric range."
    }
    with open(OUT/"GATE3_FINAL_PROTOCOL.json","w") as f: json.dump(protocol,f,indent=2)

    print("\n========== FINAL STRUCTURAL ABLATION ==========")
    show=["Variant","ValRankAcc_Selected","Test_Stress_N","Test_Rankable_N","Test_ProposalRecall","Test_StructuralF1","StructuralF1_Drop_vs_Full","Test_AllStressF1","Test_MeanCandidates"]
    print(df[show].to_string(index=False))
    print("\nGATE3_FINAL_STRUCTURAL_ABLATION_COMPLETE")
    print("CSV =",csvp)
    print("PROTOCOL =",OUT/"GATE3_FINAL_PROTOCOL.json")

if __name__=="__main__":
    main()
