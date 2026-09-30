
from pathlib import Path
import argparse, json, math, random, os, shutil
import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA")
DATA = ROOT / "data/processed/yolo_retaildet_style"
YAML = DATA / "retaildet_style.yaml"
TRAIN_PROJECT = Path("/root/autodl-tmp/ssr_yolorace_exp/outputs")
IOU_THR = 0.30
EPOCHS = 20

# Frozen from Seed42 validation only. Do not change for 2024/3407.
CONF_THR = 0.03
AMBIG_MARGIN = 0.12
STRUCT_ALPHA = 0.50
TARGET_F1 = 0.9504
KEEP = list(range(1,16)) + [18]  # confidence-free structure branch

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def read_jsonl(p):
    rows=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

def loadj(p):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def box_iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0.,ix2-ix1); ih=max(0.,iy2-iy1); inter=iw*ih
    aa=max(0.,ax2-ax1)*max(0.,ay2-ay1)
    bb=max(0.,bx2-bx1)*max(0.,by2-by1)
    u=aa+bb-inter
    return inter/u if u>0 else 0.

def gtbox(g): return g["bbox"] if isinstance(g,dict) else g

def locate_initial_weight():
    cands=[
        Path("/root/autodl-tmp/ssr_yolorace_exp/code/YOLO-RACE/yolov8n.pt"),
        Path("/root/autodl-tmp/yolov8n.pt"),
        Path("/root/yolov8n.pt"),
        Path("yolov8n.pt")
    ]
    for p in cands:
        if p.exists(): return str(p)
    return "yolov8n.pt"

def train_detector(seed):
    from ultralytics import YOLO
    name=f"SSR_FROZEN_seed{seed}_20ep"
    best=TRAIN_PROJECT/name/"weights/best.pt"
    if best.exists():
        print("DETECTOR EXISTS",best,flush=True); return best
    if not YAML.exists(): raise FileNotFoundError(YAML)
    model=YOLO(locate_initial_weight())
    print("TRAIN DETECTOR seed",seed,"epochs",EPOCHS,flush=True)
    model.train(data=str(YAML),epochs=EPOCHS,imgsz=640,batch=4,device=0,workers=4,
                project=str(TRAIN_PROJECT),name=name,seed=seed,amp=False,exist_ok=True)
    if not best.exists(): raise RuntimeError("best.pt missing after training")
    last=best.parent/"last.pt"
    if last.exists():
        try: last.unlink()
        except Exception: pass
    return best

def cache_split(model, split, out):
    cache=out/f"{split}_pred_cache.json"
    if cache.exists():
        print("CACHE EXISTS",cache,flush=True); return loadj(cache)
    rows=read_jsonl(DATA/f"{split}_manifest.jsonl")
    pred=[]
    for j,rec in enumerate(rows,1):
        r=model.predict(source=rec["image_path"],conf=0.001,iou=0.50,imgsz=640,
                        max_det=1000,verbose=False)[0]
        products=[]; vacancies=[]
        if r.boxes is not None:
            xy=r.boxes.xyxy.detach().cpu().numpy()
            cl=r.boxes.cls.detach().cpu().numpy()
            cf=r.boxes.conf.detach().cpu().numpy()
            for b,c,s in zip(xy,cl,cf):
                q={"bbox":[float(x) for x in b.tolist()],"conf":float(s)}
                if int(c)==0: products.append(q)
                elif int(c)==1: vacancies.append(q)
        gts=rec.get("gt_missing",rec.get("missing_objects",[]))
        pred.append({"masked_id":rec.get("masked_id",str(j)),
                     "image_path":rec["image_path"],
                     "products":products,"vacancies":vacancies,"gts":gts})
        if j%100==0 or j==len(rows):
            print("CACHE",split,f"{j}/{len(rows)}",flush=True)
    with open(cache,"w",encoding="utf-8") as f: json.dump(pred,f)
    return pred

def feat_for_candidate(v, products, W=640., H=640., prod_thr=.30):
    x1,y1,x2,y2=v["bbox"]; cx=(x1+x2)/2; cy=(y1+y2)/2
    w=max(x2-x1,1.); h=max(y2-y1,1.)
    ps=[p for p in products if p["conf"]>=prod_thr]
    local=[]
    for p in ps:
        a,b,c,d=p["bbox"]; pcx=(a+c)/2; pcy=(b+d)/2
        pw=max(c-a,1.); ph=max(d-b,1.)
        if abs(pcy-cy)<=max(h,ph)*.75: local.append((p,pcx,pcy,pw,ph))
    left=[q for q in local if q[1]<cx]; right=[q for q in local if q[1]>cx]
    L=max(left,key=lambda q:q[1]) if left else None
    R=min(right,key=lambda q:q[1]) if right else None
    avgw=np.mean([q[3] for q in local]) if local else w
    avgh=np.mean([q[4] for q in local]) if local else h
    maxov=max([box_iou(v["bbox"],p["bbox"]) for p in ps],default=0.)
    if L:
        ldx=(cx-L[1])/max(avgw,1.); ldy=abs(cy-L[2])/max(avgh,1.); lconf=L[0]["conf"]
    else: ldx=5.; ldy=5.; lconf=0.
    if R:
        rdx=(R[1]-cx)/max(avgw,1.); rdy=abs(cy-R[2])/max(avgh,1.); rconf=R[0]["conf"]
    else: rdx=5.; rdy=5.; rconf=0.
    both=1. if (L and R) else 0.
    if L and R:
        mid=(L[1]+R[1])/2.; midres=abs(cx-mid)/max(avgw,1.)
        span=(R[1]-L[1])/max(avgw,1.)
        wr=w/max((L[3]+R[3])/2.,1.); hr=h/max((L[4]+R[4])/2.,1.)
    else:
        midres=5.; span=0.; wr=w/max(avgw,1.); hr=h/max(avgh,1.)
    return np.array([
        v["conf"],cx/W,cy/H,w/W,h/H,(w*h)/(W*H),
        min(len(local),50)/50.,both,
        min(ldx,5)/5.,min(rdx,5)/5.,min(ldy,5)/5.,min(rdy,5)/5.,
        min(midres,5)/5.,min(span,10)/10.,
        min(wr,5)/5.,min(hr,5)/5.,
        lconf,rconf,maxov],dtype=np.float32)

class Ranker(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),
                               nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def prepare_images(rows):
    images=[]; raw=[]
    for rec in rows:
        if len(rec["gts"])!=1: continue
        gt=gtbox(rec["gts"][0]); cand=[]
        for v in rec["vacancies"]:
            if float(v["conf"])<0.001: continue
            f=feat_for_candidate(v,rec["products"])[KEEP]
            cand.append((v,f,box_iou(v["bbox"],gt))); raw.append(f)
        pos=[j for j,z in enumerate(cand) if z[2]>=IOU_THR]
        if not cand or not pos: continue
        target=max(pos,key=lambda j:cand[j][2]); images.append((cand,target))
    if not raw: raise RuntimeError("no ranker candidates")
    return images,np.stack(raw)

def normalize(images,mu,sd):
    return [([(v,(f-mu)/sd,ov) for v,f,ov in cand],target) for cand,target in images]

def rank_acc(model,images):
    model.eval(); ok=0
    with torch.no_grad():
        for cand,target in images:
            x=torch.from_numpy(np.stack([z[1] for z in cand])).float()
            ok += int(int(torch.argmax(model(x)).item())==target)
    return ok/max(len(images),1)

def train_ranker(train,val,out,seed):
    tr,raw=prepare_images(train); va,_=prepare_images(val)
    mu=raw.mean(0).astype(np.float32); sd=(raw.std(0)+1e-6).astype(np.float32)
    tr=normalize(tr,mu,sd); va=normalize(va,mu,sd)
    model=Ranker(len(KEEP))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best_acc=-1.; best=None
    for ep in range(1,26):
        random.shuffle(tr); model.train(); total=0.
        for cand,target in tr:
            x=torch.from_numpy(np.stack([z[1] for z in cand])).float()
            y=torch.tensor([target],dtype=torch.long)
            loss=nn.functional.cross_entropy(model(x).unsqueeze(0),y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        a=rank_acc(model,va)
        print("RANK seed=%d ep=%02d loss=%.5f val_rank_acc=%.5f" %
              (seed,ep,total/max(len(tr),1),a),flush=True)
        if a>best_acc:
            best_acc=a; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    torch.save({"model":best,"mu":mu,"sd":sd,"keep":KEEP,
                "val_rank_acc":best_acc},out/"listwise_structure_ranker.pt")
    return model,mu,sd,best_acc

def attach_scores(rows,model,mu,sd):
    out=[]; model.eval()
    with torch.no_grad():
        for rec in rows:
            vv=[]; ff=[]
            for v in rec["vacancies"]:
                if float(v["conf"])<0.001: continue
                ff.append((feat_for_candidate(v,rec["products"])[KEEP]-mu)/sd)
                vv.append(dict(v))
            if ff:
                s=model(torch.from_numpy(np.stack(ff)).float()).numpy()
                for q,z in zip(vv,s): q["structure_score"]=float(z)
            rr=dict(rec); rr["vacancies"]=vv; out.append(rr)
    return out

def choose(rec):
    cand=[v for v in rec["vacancies"] if float(v["conf"])>=CONF_THR]
    if not cand: return None
    cand=sorted(cand,key=lambda v:float(v["conf"]),reverse=True)
    if len(cand)==1: return cand[0]
    gap=float(cand[0]["conf"])-float(cand[1]["conf"])
    if gap>AMBIG_MARGIN: return cand[0]
    s=np.array([v["structure_score"] for v in cand],dtype=np.float64)
    z=(s-s.mean())/(s.std()+1e-8)
    joint=[math.log(max(float(v["conf"]),1e-8))+STRUCT_ALPHA*float(q)
           for v,q in zip(cand,z)]
    return cand[int(np.argmax(joint))]

def evaluate(rows):
    TP=FP=FN=0; changed=corrected=harmed=0
    for rec in rows:
        if len(rec["gts"])!=1: raise RuntimeError("expected one GT")
        gt=gtbox(rec["gts"][0])
        basecand=[v for v in rec["vacancies"] if float(v["conf"])>=CONF_THR]
        b=max(basecand,key=lambda v:float(v["conf"])) if basecand else None
        p=choose(rec)
        bok=(b is not None and box_iou(b["bbox"],gt)>=IOU_THR)
        pok=(p is not None and box_iou(p["bbox"],gt)>=IOU_THR)
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

def run(seed):
    seed_all(seed)
    out=ROOT/f"outputs/frozen_multiseed/seed{seed}"
    out.mkdir(parents=True,exist_ok=True)
    bestpt=train_detector(seed)
    from ultralytics import YOLO
    det=YOLO(str(bestpt))
    train=cache_split(det,"train",out)
    val=cache_split(det,"val",out)

    ranker,mu,sd,racc=train_ranker(train,val,out,seed)
    val_s=attach_scores(val,ranker,mu,sd)
    val_m=evaluate(val_s)
    print("FROZEN VAL",val_m,flush=True)

    # Formal test is predicted/evaluated only after detector, ranker and all decoder
    # hyperparameters are frozen. No test-driven tuning is performed here.
    test=cache_split(det,"test",out)
    test_s=attach_scores(test,ranker,mu,sd)
    test_m=evaluate(test_s)
    print("FROZEN TEST",test_m,flush=True)

    res={
        "seed":seed,
        "detector_epochs":EPOCHS,
        "detector_weights":str(bestpt),
        "iou_threshold":IOU_THR,
        "frozen_decoder":{"conf_thr":CONF_THR,"ambiguity_margin":AMBIG_MARGIN,
                          "structure_alpha":STRUCT_ALPHA},
        "structure_branch":"confidence-free geometry/topology listwise ranker",
        "ranker_val_accuracy":racc,
        "val":val_m,
        "test":test_m,
        "test_gt_used_for_model_or_hyperparameter_selection":False,
        "success_test_f1_gt_0p9504":bool(test_m["F1"]>TARGET_F1)
    }
    with open(out/"FINAL_FROZEN_RESULT.json","w",encoding="utf-8") as f:
        json.dump(res,f,indent=2)
    print(json.dumps(res,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    a=ap.parse_args()
    run(a.seed)

if __name__=="__main__": main()
