from pathlib import Path
import json, math, re, hashlib, collections
import numpy as np
P=Path('/root/autodl-tmp/ssr_yolorace_exp/code/SSR_release_clean/SSR-VQA')
CACHE_DIR=P/'outputs/f1_recovery/seed42'
TOP1_CONF=0.03
REGIONS=['upper left','upper center','upper right','middle left','middle center','middle right','lower left','lower center','lower right']

def lines(path):
    with open(path,encoding='utf-8') as f:
        for s in f:
            s=s.strip()
            if s: yield json.loads(s)

def load_cache(path):
    z=json.load(open(path,encoding='utf-8'))
    return z['per_sample'] if isinstance(z,dict) and 'per_sample' in z else z

def center_region(box,W,H):
    x1,y1,x2,y2=map(float,box); x=((x1+x2)/2)/max(float(W),1.0); y=((y1+y2)/2)/max(float(H),1.0)
    cx=0 if x<1/3 else 1 if x<2/3 else 2; cy=0 if y<1/3 else 1 if y<2/3 else 2
    return REGIONS[cy*3+cx]

def det_choice(rec):
    vv=[v for v in rec.get('vacancies',[]) if float(v.get('conf',0))>=TOP1_CONF]
    return max(vv,key=lambda v:float(v.get('conf',0))) if vv else None

def wrong_region(mid,truth,task):
    pool=[r for r in REGIONS if r!=truth]
    h=int(hashlib.sha256((mid+'|'+task).encode()).hexdigest()[:12],16)
    return pool[h%len(pool)]

def query_region(mid,truth,task):
    h=int(hashlib.sha256((mid+'|'+task+'|balance').encode()).hexdigest()[:8],16)
    pos=(h%2==0); return (truth if pos else wrong_region(mid,truth,task)),pos

def overlap_y(a,b):
    ay1,ay2=float(a[1]),float(a[3]); by1,by2=float(b[1]),float(b[3])
    inter=max(0.0,min(ay2,by2)-max(ay1,by1))
    return inter/max(min(max(ay2-ay1,1e-6),max(by2-by1,1e-6)),1e-6)

def side_support(vbox,boxes,vert_thr,dist_factor):
    if not boxes:return 'neither'
    vx1,vy1,vx2,vy2=map(float,vbox); vcx=(vx1+vx2)/2
    medw=float(np.median([max(float(b[2])-float(b[0]),1.0) for b in boxes])); maxgap=dist_factor*medw
    L=R=False
    for b in boxes:
        if overlap_y(vbox,b)<vert_thr: continue
        x1,y1,x2,y2=map(float,b); cx=(x1+x2)/2
        if cx<vcx and max(0.0,vx1-x2)<=maxgap:L=True
        if cx>vcx and max(0.0,x1-vx2)<=maxgap:R=True
    return 'both' if L and R else 'left' if L else 'right' if R else 'neither'

def ref_boxes(m): return [o['bbox'] for o in m.get('observed_objects',[]) if 'bbox' in o]
def pred_boxes(rec,thr): return [p['bbox'] for p in rec.get('products',[]) if float(p.get('conf',0))>=thr]
def tok(s): return re.findall(r'[a-z0-9]+',str(s).lower())
def em(a,b): return int(tok(a)==tok(b))
def tf1(a,b):
    A,B=tok(a),tok(b)
    if not A and not B:return 1.0
    ca,cb=collections.Counter(A),collections.Counter(B); m=sum((ca&cb).values()); p=m/max(len(A),1); r=m/max(len(B),1)
    return 2*p*r/max(p+r,1e-12)
def bleu1(a,b):
    A,B=tok(a),tok(b)
    if not A:return 0.0
    ca,cb=collections.Counter(A),collections.Counter(B); m=sum((ca&cb).values()); prec=m/len(A); bp=1.0 if len(A)>=len(B) else math.exp(1-len(B)/max(len(A),1))
    return bp*prec

def q(mid,t,region=None):
    banks={
    'Locate':['Where is the missing shelf position in the image?','Which image region contains the missing product position?','Locate the structural vacancy in the shelf image.','In which region of the image is the missing shelf slot?'],
    'Verify':['Is the missing shelf position in the {region} region?','Does the {region} region contain the structural vacancy?','Is the vacancy located in the {region} part of the image?','Should the missing shelf slot be assigned to the {region} region?'],
    'Count':['How many missing shelf positions are in the {region} region?','Count the structural vacancies in the {region} region.','How many missing product positions occur in the {region} part of the image?','What is the number of missing shelf slots in the {region} region?'],
    'Identify':['Which side provides nearby visible-product support for replenishment at the missing position?','Which neighboring side has local product support around the vacancy?','What side configuration surrounds the missing shelf position?','Which adjacent side supports the inferred replenishment position?']}
    i=int(hashlib.sha256((mid+'|'+t+'|q').encode()).hexdigest()[:8],16)%4
    z=banks[t][i]; return z.format(region=region) if region else z

def space(t):
    if t=='Locate':return REGIONS
    if t=='Verify':return ['yes','no']
    if t=='Count':return ['zero','one']
    return ['left','right','both','neither']

def image_path(m): return str(m.get('masked_image_path') or m.get('image_path') or m.get('original_image_path'))

def build(mid,m,rec,prod_thr,vert_thr,dist_factor):
    gt=(rec.get('gts') or m.get('missing_objects') or [None])[0]
    if gt is None:return None
    W,H=float(m['width']),float(m['height']); gtreg=center_region(gt['bbox'],W,H); ch=det_choice(rec); has=ch is not None
    preg=center_region(ch['bbox'],W,H) if has else 'unknown'; out=[]
    out.append(('Locate',q(mid,'Locate'),gtreg,preg if has else 'unknown',space('Locate')))
    qr,pos=query_region(mid,gtreg,'Verify'); out.append(('Verify',q(mid,'Verify',qr),'yes' if pos else 'no','yes' if has and preg==qr else 'no',space('Verify')))
    qr,pos=query_region(mid,gtreg,'Count'); out.append(('Count',q(mid,'Count',qr),'one' if pos else 'zero','one' if has and preg==qr else 'zero',space('Count')))
    gi=side_support(gt['bbox'],ref_boxes(m),vert_thr,dist_factor)
    pi=side_support(ch['bbox'],pred_boxes(rec,prod_thr),vert_thr,dist_factor) if has else 'neither'
    out.append(('Identify',q(mid,'Identify'),gi,pi,space('Identify')))
    return out

def summarize(rows):
    out=[]
    for t in ['Count','Verify','Locate','Identify','Overall']:
        z=rows if t=='Overall' else [r for r in rows if r['type']==t]
        out.append({'type':t,'n':len(z),'EM':sum(r['EM'] for r in z)/max(len(z),1),'TokenF1':sum(r['TokenF1'] for r in z)/max(len(z),1),'BLEU1':sum(r['BLEU1'] for r in z)/max(len(z),1)})
    return out
