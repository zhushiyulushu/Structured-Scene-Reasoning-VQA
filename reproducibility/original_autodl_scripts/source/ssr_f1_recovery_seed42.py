
from pathlib import Path
import json, math, random, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA")
DATA = ROOT / "data/processed/yolo_retaildet_style"
YAML = DATA / "retaildet_style.yaml"
OUT = ROOT / "outputs/f1_recovery/seed42"
TRAIN_PROJECT = Path("/root/autodl-tmp/ssr_yolorace_exp/outputs")
TRAIN_NAME = "SSR_F1_RECOVERY_seed42_20ep"
SEED = 42
IOU_THR = 0.30
TARGET_F1 = 0.9504

OUT.mkdir(parents=True, exist_ok=True)

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def read_jsonl(p):
    rows=[]
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def box_iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.0,ix2-ix1); ih=max(0.0,iy2-iy1); inter=iw*ih
    aa=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    bb=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    u=aa+bb-inter
    return inter/u if u>0 else 0.0

def eval_preds(per_sample, score_key="conf", thr=0.3):
    TP=FP=FN=0
    for rec in per_sample:
        preds=[p for p in rec["vacancies"] if float(p.get(score_key,0.0))>=thr]
        gts=rec["gts"]
        used=set(); tp=0
        for gt in gts:
            gtbox = gt["bbox"] if isinstance(gt,dict) else gt
            best_i=None; best=0.0
            for i,p in enumerate(preds):
                if i in used: continue
                v=box_iou(p["bbox"], gtbox)
                if v>best: best=v; best_i=i
            if best_i is not None and best>=IOU_THR:
                used.add(best_i); tp+=1
        TP+=tp; FP+=len(preds)-tp; FN+=len(gts)-tp
    P=TP/max(TP+FP,1); R=TP/max(TP+FN,1); F=2*P*R/max(P+R,1e-12)
    return {"TP":TP,"FP":FP,"FN":FN,"Precision":P,"Recall":R,"F1":F}

def locate_initial_weight():
    cands=[
        Path("/root/autodl-tmp/ssr_yolorace_exp/code/YOLO-RACE/yolov8n.pt"),
        Path("/root/autodl-tmp/yolov8n.pt"),
        Path("/root/yolov8n.pt"),
        Path("yolov8n.pt"),
    ]
    for p in cands:
        if p.exists(): return str(p)
    return "yolov8n.pt"

def train_detector():
    from ultralytics import YOLO
    best = TRAIN_PROJECT / TRAIN_NAME / "weights/best.pt"
    if best.exists():
        print("DETECTOR EXISTS", best, flush=True)
        return best
    if not YAML.exists():
        raise FileNotFoundError("Missing dataset yaml: %s" % YAML)
    init=locate_initial_weight()
    print("TRAIN DETECTOR from", init, flush=True)
    model=YOLO(init)
    model.train(
        data=str(YAML),
        epochs=20,
        imgsz=640,
        batch=4,
        device=0,
        workers=4,
        project=str(TRAIN_PROJECT),
        name=TRAIN_NAME,
        seed=SEED,
        amp=False,
        exist_ok=True,
    )
    if not best.exists():
        alt=TRAIN_PROJECT/TRAIN_NAME/"weights"/"best.pt"
        if alt.exists(): best=alt
    if not best.exists():
        raise RuntimeError("Training finished but best.pt not found")
    return best

def cache_split(model, split):
    cache=OUT/f"{split}_pred_cache.json"
    if cache.exists():
        print("CACHE EXISTS", cache, flush=True)
        return json.load(open(cache,"r",encoding="utf-8"))
    manifest=DATA/f"{split}_manifest.jsonl"
    rows=read_jsonl(manifest)
    out=[]
    for j,rec in enumerate(rows,1):
        r=model.predict(source=rec["image_path"], conf=0.001, iou=0.50, imgsz=640, max_det=1000, verbose=False)[0]
        products=[]; vacancies=[]
        if r.boxes is not None:
            xy=r.boxes.xyxy.detach().cpu().numpy()
            cl=r.boxes.cls.detach().cpu().numpy()
            cf=r.boxes.conf.detach().cpu().numpy()
            for b,c,s in zip(xy,cl,cf):
                d={"bbox":[float(x) for x in b.tolist()],"conf":float(s)}
                if int(c)==0: products.append(d)
                elif int(c)==1: vacancies.append(d)
        gts=rec.get("gt_missing", rec.get("missing_objects", []))
        out.append({"masked_id":rec.get("masked_id",str(j)),
                    "image_path":rec["image_path"],
                    "products":products,"vacancies":vacancies,"gts":gts})
        if j%50==0 or j==len(rows): print(f"CACHE {split} {j}/{len(rows)}", flush=True)
    json.dump(out,open(cache,"w",encoding="utf-8"))
    return out

def tune_visual(val):
    grid=[0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70]
    rows=[]
    for t in grid:
        m=eval_preds(val,"conf",t); rows.append({"thr":t,**m})
        print("VIS VAL",t,m,flush=True)
    best=max(rows,key=lambda x:(x["F1"],x["Precision"],x["thr"]))
    return best,rows

def feat_for_candidate(v, products, W=640.0, H=640.0, prod_thr=0.30):
    x1,y1,x2,y2=v["bbox"]
    cx=(x1+x2)/2; cy=(y1+y2)/2; w=max(x2-x1,1.0); h=max(y2-y1,1.0)
    ps=[p for p in products if p["conf"]>=prod_thr]
    local=[]
    for p in ps:
        a,b,c,d=p["bbox"]; pcx=(a+c)/2; pcy=(b+d)/2; pw=max(c-a,1.0); ph=max(d-b,1.0)
        if abs(pcy-cy) <= max(h,ph)*0.75:
            local.append((p,pcx,pcy,pw,ph))
    left=[q for q in local if q[1] < cx]
    right=[q for q in local if q[1] > cx]
    L=max(left,key=lambda q:q[1]) if left else None
    R=min(right,key=lambda q:q[1]) if right else None
    avgw=np.mean([q[3] for q in local]) if local else w
    avgh=np.mean([q[4] for q in local]) if local else h
    maxov=0.0
    for p in ps:
        maxov=max(maxov,box_iou(v["bbox"],p["bbox"]))
    if L:
        ldx=(cx-L[1])/max(avgw,1.0); ldy=abs(cy-L[2])/max(avgh,1.0); lconf=L[0]["conf"]
    else: ldx=5.0; ldy=5.0; lconf=0.0
    if R:
        rdx=(R[1]-cx)/max(avgw,1.0); rdy=abs(cy-R[2])/max(avgh,1.0); rconf=R[0]["conf"]
    else: rdx=5.0; rdy=5.0; rconf=0.0
    both=1.0 if (L and R) else 0.0
    if L and R:
        mid=(L[1]+R[1])/2.0
        midres=abs(cx-mid)/max(avgw,1.0)
        span=(R[1]-L[1])/max(avgw,1.0)
        whratio=w/max((L[3]+R[3])/2.0,1.0)
        hhratio=h/max((L[4]+R[4])/2.0,1.0)
    else:
        midres=5.0; span=0.0; whratio=w/max(avgw,1.0); hhratio=h/max(avgh,1.0)
    return np.array([
        v["conf"], cx/W, cy/H, w/W, h/H, (w*h)/(W*H),
        min(len(local),50)/50.0, both,
        min(ldx,5)/5.0, min(rdx,5)/5.0,
        min(ldy,5)/5.0, min(rdy,5)/5.0,
        min(midres,5)/5.0, min(span,10)/10.0,
        min(whratio,5)/5.0, min(hhratio,5)/5.0,
        lconf, rconf, maxov
    ],dtype=np.float32)

class TopoFilter(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(0.1),
                               nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def make_candidate_set(cache, train=False):
    X=[];Y=[]
    for rec in cache:
        positives=[]; negatives=[]
        for v in rec["vacancies"]:
            if v["conf"] < 0.003: continue
            f=feat_for_candidate(v,rec["products"])
            y=int(any(box_iou(v["bbox"], g["bbox"] if isinstance(g,dict) else g)>=IOU_THR for g in rec["gts"]))
            (positives if y else negatives).append((f,y,v["conf"]))
        if train:
            X += [(f,y) for f,y,_ in positives]
            negatives=sorted(negatives,key=lambda z:z[2],reverse=True)[:max(3,3*max(1,len(positives)))]
            X += [(f,y) for f,y,_ in negatives]
        else:
            X += [(f,y) for f,y,_ in positives+negatives]
    if train:
        random.shuffle(X)
    if not X: raise RuntimeError("No topology candidates")
    A=np.stack([z[0] for z in X]); B=np.array([z[1] for z in X],dtype=np.float32)
    return A,B

def train_topofilter(train_cache, val_cache):
    ck=OUT/"topofilter.pt"
    X,Y=make_candidate_set(train_cache,True)
    XV,YV=make_candidate_set(val_cache,False)
    mu=X.mean(0); sd=X.std(0)+1e-6
    X=(X-mu)/sd; XV=(XV-mu)/sd
    print("TOPO TRAIN",X.shape,"pos",int(Y.sum()),"VAL",XV.shape,"pos",int(YV.sum()),flush=True)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TopoFilter(X.shape[1]).to(dev)
    ds=TensorDataset(torch.from_numpy(X).float(),torch.from_numpy(Y).float())
    dl=DataLoader(ds,batch_size=512,shuffle=True,num_workers=0)
    pos=max(float(Y.sum()),1.0); neg=max(float(len(Y)-Y.sum()),1.0)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos],device=dev).squeeze())
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=1e9; best_state=None
    xv=torch.from_numpy(XV).float().to(dev); yv=torch.from_numpy(YV).float().to(dev)
    for ep in range(1,16):
        model.train(); tot=0;n=0
        for a,b in dl:
            a=a.to(dev); b=b.to(dev); opt.zero_grad()
            loss=crit(model(a),b); loss.backward(); opt.step()
            tot+=float(loss.item())*len(a);n+=len(a)
        model.eval()
        with torch.no_grad(): vl=float(crit(model(xv),yv).item())
        print(f"TOPO ep={ep:02d} train={tot/max(n,1):.5f} val={vl:.5f}",flush=True)
        if vl<best:
            best=vl
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"model":best_state,"mu":mu,"sd":sd},ck)
    return model,mu,sd,dev

def apply_toposcores(cache,model,mu,sd,dev):
    model.eval()
    out=[]
    with torch.no_grad():
        for rec in cache:
            vac=[]
            feats=[]; keep=[]
            for v in rec["vacancies"]:
                if v["conf"]<0.003: continue
                feats.append(feat_for_candidate(v,rec["products"])); keep.append(dict(v))
            if feats:
                A=(np.stack(feats)-mu)/sd
                sc=torch.sigmoid(model(torch.from_numpy(A).float().to(dev))).cpu().numpy()
                for v,s in zip(keep,sc): v["topo_score"]=float(s)
            out.append({**rec,"vacancies":keep})
    return out

def tune_topo(val):
    grid=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.93,0.95,0.97]
    rows=[]
    for t in grid:
        m=eval_preds(val,"topo_score",t); rows.append({"thr":t,**m})
        print("TOPO VAL",t,m,flush=True)
    best=max(rows,key=lambda x:(x["F1"],x["Precision"],x["thr"]))
    return best,rows

def main():
    seed_all(SEED)
    bestpt=train_detector()
    from ultralytics import YOLO
    model=YOLO(str(bestpt))
    train_cache=cache_split(model,"train")
    val_cache=cache_split(model,"val")
    test_cache=cache_split(model,"test")

    visual_best,visual_grid=tune_visual(val_cache)
    visual_test=eval_preds(test_cache,"conf",visual_best["thr"])
    print("VISUAL SELECTED",visual_best,flush=True)
    print("VISUAL TEST",visual_test,flush=True)

    topo_model,mu,sd,dev=train_topofilter(train_cache,val_cache)
    val_topo=apply_toposcores(val_cache,topo_model,mu,sd,dev)
    test_topo=apply_toposcores(test_cache,topo_model,mu,sd,dev)
    topo_best,topo_grid=tune_topo(val_topo)
    topo_test=eval_preds(test_topo,"topo_score",topo_best["thr"])
    print("OURS SELECTED",topo_best,flush=True)
    print("OURS TEST",topo_test,flush=True)

    summary={
        "protocol":"fresh YOLOv8n vacancy proposals + learned local-topology candidate filtering; all thresholds selected on source val",
        "seed":SEED,
        "detector_weights":str(bestpt),
        "test_gt_used_only_after_predictions_for_evaluation":True,
        "visual_val_selected":visual_best,
        "visual_test":visual_test,
        "ours_topofilter_val_selected":topo_best,
        "ours_topofilter_test":topo_test,
        "strongest_same_protocol_reference_f1":TARGET_F1,
        "success_gate_ours_f1_gt_0p9504":bool(topo_test["F1"]>TARGET_F1),
        "topology_improves_over_visual_on_val":bool(topo_best["F1"]>visual_best["F1"]),
        "topology_improves_over_visual_on_test":bool(topo_test["F1"]>visual_test["F1"]),
        "visual_grid":visual_grid,
        "topology_grid":topo_grid,
    }
    with open(OUT/"FINAL_F1_RECOVERY_SEED42.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    print("\n"+"="*88)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print("="*88)
    print("F1_RECOVERY_SEED42_COMPLETE",flush=True)

if __name__=="__main__":
    main()
