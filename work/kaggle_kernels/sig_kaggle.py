# SigLIP-so400m + ConvNeXt-large: декоррелированные бэкбоны -> фичи наших пар -> компараторы -> OOF.
import os, io, time, subprocess, warnings, glob
warnings.filterwarnings("ignore")
def sh(c): print("+", c[:70], flush=True); subprocess.run(c, shell=True)
sh("pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1")
sh("pip -q install --no-deps timm 2>&1 | tail -1")
import numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
OUT="/kaggle/working"; DEV="cuda"
print("torch",torch.__version__,"GPU",torch.cuda.get_device_name(0),flush=True)
print("smoke",float((torch.randn(8,8,device=DEV)@torch.randn(8,8,device=DEV)).sum()),flush=True)
DATA=os.path.dirname([f for f in glob.glob("/kaggle/input/**/train.parquet",recursive=True) if "competitions" in f][0])
t=pq.read_table(f"{DATA}/train.parquet",columns=["image_1","image_2","is_image1_better"]).to_pydict()
y=np.array(t["is_image1_better"],dtype=np.float32)
tt_=pq.read_table(f"{DATA}/test.parquet",columns=["image_1","image_2"]).to_pydict()
NTR,NTE=len(y),len(tt_["image_1"]); idx=np.arange(NTE)
def dec(b):
    try: return Image.open(io.BytesIO(b)).convert("RGB")
    except: return Image.new("RGB",(64,64),(128,128,128))
def run_backbone(name, tag):
    import timm
    m=timm.create_model(name,pretrained=True,num_classes=0).to(DEV).eval().half()
    cfg=timm.data.resolve_data_config({},model=m)
    tf=timm.data.create_transform(**cfg)
    print(tag,"input:",cfg["input_size"],flush=True)
    @torch.no_grad()
    def feat(bl,bs=32):
        ex=ThreadPoolExecutor(max_workers=8); out=[]; t0=time.time()
        for i in range(0,len(bl),bs):
            pils=list(ex.map(dec,bl[i:i+bs]))
            px=torch.stack([tf(p) for p in pils]).to(DEV).half()
            out.append(m(px).float().cpu().numpy().astype(np.float16))
            if (i//bs)%80==0: print(f"  {i}/{len(bl)} {i/max(time.time()-t0,1):.0f}/s",flush=True)
        return np.concatenate(out)
    t0=time.time()
    f={}
    f["tr1"]=feat(t["image_1"]); f["tr2"]=feat(t["image_2"]); f["te1"]=feat(tt_["image_1"]); f["te2"]=feat(tt_["image_2"])
    print(tag,"done",round(time.time()-t0),"s dim",f["tr1"].shape[1],flush=True)
    np.savez_compressed(f"{OUT}/emb_{tag}.npz",train_img1=f["tr1"],train_img2=f["tr2"],
        test_img1=f["te1"],test_img2=f["te2"],train_y=y.astype(np.int8),test_index=idx.astype(np.int32))
    del m; torch.cuda.empty_cache(); return f
class Comp(nn.Module):
    def __init__(s,d,h=512,p=0.3):
        super().__init__()
        s.proj=nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(p),
                             nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(p))
        s.sc=nn.Linear(h,1); s.cp=nn.Sequential(nn.Linear(4*h,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(p),nn.Linear(h,1))
        s.q=nn.Parameter(torch.tensor(1.0))
    def forward(s,e1,e2):
        z1,z2=s.proj(e1),s.proj(e2)
        return (s.cp(torch.cat([z1,z2,z1-z2,z1*z2],-1))+s.q*(s.sc(z1)-s.sc(z2))).squeeze(-1)
def run_cv(f,tag,epochs=50,lr=1e-3,ls=0.05,seed=42):
    pool=np.concatenate([f["tr1"],f["tr2"]]).astype(np.float32); MU,SD=pool.mean(0),pool.std(0)+1e-6
    nz=lambda a:(a.astype(np.float32)-MU)/SD
    n1,n2,u1,u2=nz(f["tr1"]),nz(f["tr2"]),nz(f["te1"]),nz(f["te2"])
    tt2=lambda a: torch.tensor(a,dtype=torch.float32,device=DEV)
    skf=StratifiedKFold(5,shuffle=True,random_state=seed); oof=np.zeros(NTR); tp=np.zeros(NTE)
    U1,U2=tt2(u1),tt2(u2)
    for fo,(tri,vai) in enumerate(skf.split(n1,y)):
        torch.manual_seed(seed+fo); m=Comp(n1.shape[1]).to(DEV)
        op=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-2)
        sc=torch.optim.lr_scheduler.CosineAnnealingLR(op,T_max=epochs); lf=nn.BCEWithLogitsLoss()
        e1,e2,Y=tt2(n1[tri]),tt2(n2[tri]),tt2(y[tri]*(1-ls)+.5*ls); v1,v2=tt2(n1[vai]),tt2(n2[vai]); nb=len(Y); best=-1
        for ep in range(epochs):
            m.train(); perm=torch.randperm(nb,device=DEV)
            for i in range(0,nb,256):
                b=perm[i:i+256]; op.zero_grad()
                if torch.rand(1).item()<.5: out=m(e1[b],e2[b]); tg=Y[b]
                else: out=m(e2[b],e1[b]); tg=1-Y[b]
                lf(out,tg).backward(); op.step()
            sc.step(); m.eval()
            with torch.no_grad(): pv=.5*(torch.sigmoid(m(v1,v2))+(1-torch.sigmoid(m(v2,v1)))).cpu().numpy()
            a=roc_auc_score(y[vai],pv)
            if a>best:
                best=a; ov=pv
                with torch.no_grad(): tpp=.5*(torch.sigmoid(m(U1,U2))+(1-torch.sigmoid(m(U2,U1)))).cpu().numpy()
        oof[vai]=ov; tp+=tpp/5; print(f"  [{tag}] fold{fo} {best:.4f}",flush=True)
    au=roc_auc_score(y,oof); print(f"== {tag} OOF AUC = {au:.4f} ==",flush=True)
    np.savez(f"{OUT}/oof_{tag}.npz",oof=oof,test=tp,y=y.astype(np.int8),idx=idx.astype(np.int32),auc=au)
for name,tag in [("vit_so400m_patch14_siglip_384.webli","siglip"),
                 ("convnext_large.fb_in22k_ft_in1k_384","convnext")]:
    try:
        f=run_backbone(name,tag); run_cv(f,tag)
    except Exception as e:
        import traceback; traceback.print_exc(); print(tag,"FAILED",str(e)[:100],flush=True)
print("ALL DONE",flush=True)
