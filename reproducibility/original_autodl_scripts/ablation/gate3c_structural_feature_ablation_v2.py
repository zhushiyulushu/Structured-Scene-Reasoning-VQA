
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
OUT=ROOT/"outputs/gate3c_structural_feature_ablation_seed42_v2"

SEED=42
TRAIN_MIN_CONF=0.001
EVAL_CONF=0.03
IOU=0.30
EPOCHS=25

# Original 19-D detector-context feature indices.
FULL=list(range(1,16))+[18]
GROUPS={
    "global_geometry":[1,2,3,4,5],
    "neighbor_presence":[6,7],
    "directional_offsets":[8,9,10,11],
    "gap_regularity":[12,13,14,15],
    "overlap_consistency":[18],
}

def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def loadj(p):
    with open(p,"r") as f:
        return json.load(f)

def gtbox(g):
    return g["bbox"] if isinstance(g,dict) else g

def iou(a,b):
    ax1,ay1,ax2,ay2=a
    bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.,ix2-ix1); ih=max(0.,iy2-iy1)
    inter=iw*ih
    aa=max(0.,ax2-ax1)*max(0.,ay2-ay1)
    bb=max(0.,bx2-bx1)*max(0.,by2-by1)
    u=aa+bb-inter
    return inter/u if u>0 else 0.

def raw_feat19(v, products, W=640., H=640., prod_thr=.30):
    """Return the exact 19-D detector-context feature vector used by the frozen ranker."""
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

    if L:
        ldx=(cx-L[1])/max(avgw,1.)
        ldy=abs(cy-L[2])/max(avgh,1.)
        lconf=float(L[0]["conf"])
    else:
        ldx=5.; ldy=5.; lconf=0.

    if R:
        rdx=(R[1]-cx)/max(avgw,1.)
        rdy=abs(cy-R[2])/max(avgh,1.)
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
        wr=w/max(avgw,1.)
        hr=h/max(avgh,1.)

    return np.array([
        float(v["conf"]),            # 0 vacancy confidence (excluded from structural branch)
        cx/W, cy/H, w/W, h/H,        # 1..4
        (w*h)/(W*H),                 # 5
        min(len(local),50)/50.,      # 6
        both,                        # 7
        min(ldx,5)/5.,               # 8
        min(rdx,5)/5.,               # 9
        min(ldy,5)/5.,               # 10
        min(rdy,5)/5.,               # 11
        min(midres,5)/5.,            # 12
        min(span,10)/10.,            # 13
        min(wr,5)/5.,                # 14
        min(hr,5)/5.,                # 15
        lconf, rconf,                 # 16,17 neighbor confidences (excluded)
        maxov                         # 18
    ],dtype=np.float32)

class Ranker(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        return self.net(x).squeeze(-1)

def build_train_groups(rows, keep):
    groups=[]
    raw=[]
    for r in rows:
        if len(r.get("gts",[]))!=1:
            continue
        gt=gtbox(r["gts"][0])
        cand=[]
        for v in r.get("vacancies",[]):
            if float(v["conf"])<TRAIN_MIN_CONF:
                continue
            f19=raw_feat19(v,r.get("products",[]))
            fs=f19[keep]
            ov=iou(v["bbox"],gt)
            cand.append((fs,ov))
            raw.append(fs)

        pos=[i for i,(_,ov) in enumerate(cand) if ov>=IOU]
        if cand and pos:
            t=max(pos,key=lambda i:cand[i][1])
            groups.append((cand,t))

    if not raw:
        raise RuntimeError("No training features were built.")
    return groups,np.stack(raw)

def normalize_groups(groups,mu,sd):
    out=[]
    for cand,t in groups:
        xs=[(f-mu)/sd for f,_ in cand]
        out.append((xs,t))
    return out

def rank_acc(model,groups):
    if not groups:
        return 0.0
    model.eval()
    ok=0
    with torch.no_grad():
        for xs,t in groups:
            x=torch.from_numpy(np.stack(xs)).float()
            pred=int(torch.argmax(model(x)))
            ok += int(pred==t)
    return ok/len(groups)

def train_ranker(train_rows,val_rows,keep):
    seed_all()
    tr,raw=build_train_groups(train_rows,keep)
    va,_=build_train_groups(val_rows,keep)
    mu=raw.mean(0).astype(np.float32)
    sd=(raw.std(0)+1e-6).astype(np.float32)
    tr=normalize_groups(tr,mu,sd)
    va=normalize_groups(va,mu,sd)

    m=Ranker(len(keep))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)

    best=-1.
    best_state=None
    best_epoch=-1
    for ep in range(EPOCHS):
        random.shuffle(tr)
        m.train()
        for xs,t in tr:
            x=torch.from_numpy(np.stack(xs)).float()
            logits=m(x).unsqueeze(0)
            y=torch.tensor([t],dtype=torch.long)
            loss=nn.functional.cross_entropy(logits,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        a=rank_acc(m,va)
        if a>best:
            best=a
            best_epoch=ep+1
            best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

    m.load_state_dict(best_state)
    m.eval()
    return m,mu,sd,best,best_epoch

def load_full_ranker():
    ck=torch.load(FULL_CK,map_location="cpu")
    m=Ranker(len(FULL))
    m.load_state_dict(ck["model"])
    m.eval()
    return (
        m,
        np.asarray(ck["mu"],dtype=np.float32),
        np.asarray(ck["sd"],dtype=np.float32),
        float(ck.get("val_rank_acc",np.nan))
    )

def fixed_heuristic_score(f19):
    """Non-learned structural baseline using ONLY structure/context, never GT/confidence."""
    both=f19[7]
    ldx=f19[8]; rdx=f19[9]
    ldy=f19[10]; rdy=f19[11]
    mid=f19[12]
    span=f19[13]
    wr=f19[14]; hr=f19[15]
    ov=f19[18]

    lr_balance=abs(ldx-rdx)
    # Fixed, predeclared structural compatibility score.
    return (
        2.0*both
        -1.5*mid
        -0.5*(ldy+rdy)
        -0.35*lr_balance
        -0.25*abs(span-0.2)
        -0.35*abs(wr-0.2)
        -0.35*abs(hr-0.2)
        -1.5*ov
    )

def evaluate_variant(rows, keep=None, model=None, mu=None, sd=None, heuristic=False):
    """
    Main ablation subset: ALL detector-defined multi-candidate cases (>=2 proposals at conf>=0.03).
    This selection uses NO ground truth.
    Candidate-F1 = fraction for which the selected proposal matches the unique GT at IoU>=0.30.
    Since there is exactly one selected proposal and one GT target per image, P=R=F1=accuracy.
    Also reports conditional Rankable-F1 among cases where at least one candidate overlaps GT.
    """
    N_multi=0
    correct_all=0
    N_rankable=0
    correct_rankable=0
    candidate_recall_cases=0

    for r in rows:
        if len(r.get("gts",[]))!=1:
            continue
        gt=gtbox(r["gts"][0])

        cands=[]
        for v in r.get("vacancies",[]):
            if float(v["conf"])<EVAL_CONF:
                continue
            f19=raw_feat19(v,r.get("products",[]))
            cands.append((v,f19,iou(v["bbox"],gt)))

        if len(cands)<2:
            continue

        N_multi += 1
        rankable=any(z[2]>=IOU for z in cands)
        if rankable:
            N_rankable += 1

        if heuristic:
            scores=[fixed_heuristic_score(z[1]) for z in cands]
            pred=int(np.argmax(scores))
        else:
            X=np.stack([(z[1][keep]-mu)/sd for z in cands])
            with torch.no_grad():
                scores=model(torch.from_numpy(X).float()).cpu().numpy()
            pred=int(np.argmax(scores))

        good=cands[pred][2]>=IOU
        correct_all += int(good)
        if rankable:
            correct_rankable += int(good)

    all_f1=correct_all/max(N_multi,1)
    rankable_f1=correct_rankable/max(N_rankable,1)
    proposal_recall=N_rankable/max(N_multi,1)

    return {
        "MultiCandidate_N":N_multi,
        "Rankable_N":N_rankable,
        "ProposalRecall_on_Multi":proposal_recall,
        "CandidateF1_AllMulti":all_f1,
        "CandidateF1_Rankable":rankable_f1,
        "Correct_AllMulti":correct_all,
        "Correct_Rankable":correct_rankable,
    }

def main():
    seed_all()

    for p in [TRAIN,VAL,TEST,FULL_CK]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    train=loadj(TRAIN)
    val=loadj(VAL)
    test=loadj(TEST)
    OUT.mkdir(parents=True,exist_ok=True)

    variants=[]

    # A: no learning
    variants.append((
        "w/o Learnable Structural Ranking",
        "fixed hand-crafted structural score",
        None,None,None,None,None
    ))

    # B-F: leave-one-feature-group-out, each retrained fairly on source train and selected on source val.
    ablations=[
        ("w/o Global Geometry", sorted(set(FULL)-set(GROUPS["global_geometry"]))),
        ("w/o Neighbor Presence", sorted(set(FULL)-set(GROUPS["neighbor_presence"]))),
        ("w/o Directional Offsets", sorted(set(FULL)-set(GROUPS["directional_offsets"]))),
        ("w/o Gap Regularity", sorted(set(FULL)-set(GROUPS["gap_regularity"]))),
        ("w/o Overlap Consistency", sorted(set(FULL)-set(GROUPS["overlap_consistency"]))),
    ]

    for name,keep in ablations:
        print(f"TRAINING {name} with {len(keep)} features...",flush=True)
        m,mu,sd,best,bep=train_ranker(train,val,keep)
        print(f"  best source-val rank acc = {best:.6f} @ epoch {bep}",flush=True)
        variants.append((name,keep,m,mu,sd,best,bep))

    # G: exact frozen full structural ranker used by the final method.
    fm,fmu,fsd,fv=load_full_ranker()
    variants.append((
        "Full Learnable Structural Ranker",
        FULL,fm,fmu,fsd,fv,None
    ))

    rows=[]
    for item in variants:
        name=item[0]
        if name=="w/o Learnable Structural Ranking":
            va=evaluate_variant(val,heuristic=True)
            te=evaluate_variant(test,heuristic=True)
            val_train_rank=np.nan
            best_ep=np.nan
            feature_desc="fixed heuristic; no learned ranker"
        else:
            _,keep,m,mu,sd,val_train_rank,best_ep=item
            va=evaluate_variant(val,keep,m,mu,sd,heuristic=False)
            te=evaluate_variant(test,keep,m,mu,sd,heuristic=False)
            feature_desc=",".join(map(str,keep))

        rows.append({
            "Variant":name,
            "Features_or_rule":feature_desc,
            "TrainSelected_ValRankAcc":val_train_rank,
            "BestEpoch":best_ep,
            "Val_MultiCandidate_N":va["MultiCandidate_N"],
            "Val_Rankable_N":va["Rankable_N"],
            "Val_ProposalRecall_on_Multi":va["ProposalRecall_on_Multi"],
            "Val_CandidateF1_AllMulti":va["CandidateF1_AllMulti"],
            "Val_CandidateF1_Rankable":va["CandidateF1_Rankable"],
            "Test_MultiCandidate_N":te["MultiCandidate_N"],
            "Test_Rankable_N":te["Rankable_N"],
            "Test_ProposalRecall_on_Multi":te["ProposalRecall_on_Multi"],
            "Test_CandidateF1_AllMulti":te["CandidateF1_AllMulti"],
            "Test_CandidateF1_Rankable":te["CandidateF1_Rankable"],
        })

    df=pd.DataFrame(rows)
    csvp=OUT/"gate3c_structural_feature_ablation.csv"
    df.to_csv(csvp,index=False)

    protocol={
        "seed":SEED,
        "main_end_to_end_result_is_untouched":True,
        "purpose":"isolate the learned structural ranking stage on detector-defined multi-candidate cases instead of re-measuring the already-dominant visual detector",
        "test_subset_definition":"all test images with exactly one GT target and >=2 detector vacancy proposals at frozen confidence threshold 0.03",
        "test_subset_selection_uses_gt":False,
        "candidate_metric":"CandidateF1_AllMulti: exactly one proposal is selected for exactly one target; a selection is correct iff IoU>=0.30, so precision=recall=F1=top-1 correctness on the detector-defined multi-candidate subset",
        "rankable_metric":"CandidateF1_Rankable is secondary/diagnostic and conditions on detector proposal recall; it must not replace CandidateF1_AllMulti as the primary ablation metric",
        "gt_role":"GT is used only to score the selected proposal and to report secondary proposal-recall/rankable diagnostics; GT never enters inference",
        "frozen_eval_conf":EVAL_CONF,
        "iou":IOU,
        "epochs_for_retrained_ablations":EPOCHS,
        "feature_groups":GROUPS,
        "important_note":"Do not mix Candidate-F1 with the ~0.95 end-to-end localization F1 in one column. They answer different questions. Main table: end-to-end localization. Ablation table: structural candidate-selection ability."
    }
    with open(OUT/"GATE3C_PROTOCOL.json","w") as f:
        json.dump(protocol,f,indent=2)

    print("\n========== GATE3C RESULTS ==========")
    print(df.to_string(index=False))
    print("\nGATE3C_FIXED_COMPLETE")
    print("CSV =",csvp)
    print("PROTOCOL =",OUT/"GATE3C_PROTOCOL.json")

if __name__=="__main__":
    main()
