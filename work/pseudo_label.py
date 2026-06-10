"""Псевдо-лейблинг: уверенные предсказания лучшего ансамбля на тесте -> доучивание компаратора."""
import numpy as np, torch, torch.nn as nn
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
rk=lambda a: rankdata(a)/len(a)
full=["oof_dinoS","oof_dinoL","oof_dinoL448","oof_patchS","oof_kglobal","oof_preS",
      "oof_dinoS_s7","oof_dinoL_s7","oof_dinoS_s13","oof_dinoL448_s7","oof_patchS_s7",
      "oof_preL","oof_preL_s7"]
Z={t:np.load(f"artifacts/{t}.npz") for t in full}
y=Z[full[0]]["y"].astype(np.float32); idx=Z[full[0]]["idx"]
ens_test=np.mean([rk(Z[t]["test"]) for t in full],0)   # ранговая уверенность ансамбля
# пороги по квантилям: топ-25% -> 1, низ-25% -> 0
lo,hi=np.quantile(ens_test,[0.25,0.75])
pl_mask=(ens_test<=lo)|(ens_test>=hi)
pl_y=(ens_test>=hi).astype(np.float32)[pl_mask]
print(f"псевдо-меток: {pl_mask.sum()} из {len(ens_test)} | P(1)={pl_y.mean():.3f}")
# фичи dinoL (наши)
z=np.load("artifacts/emb_dinoL.npz")
pool=np.concatenate([z["train_img1"],z["train_img2"]]).astype(np.float32)
MU,SD=pool.mean(0),pool.std(0)+1e-6
nz=lambda a:(a.astype(np.float32)-MU)/SD
n1,n2=nz(z["train_img1"]),nz(z["train_img2"])
u1,u2=nz(z["test_img1"]),nz(z["test_img2"])
p1,p2,py=u1[pl_mask],u2[pl_mask],pl_y
NTR,NTE=len(y),len(idx)
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
tt=lambda a: torch.tensor(a,dtype=torch.float32)
P1,P2,PY=tt(p1),tt(p2),tt(py)
skf=StratifiedKFold(5,shuffle=True,random_state=42)
oof=np.zeros(NTR); tp=np.zeros(NTE); U1,U2=tt(u1),tt(u2)
ls=0.05; epochs=50
for f,(tri,vai) in enumerate(skf.split(n1,y)):
    torch.manual_seed(42+f); m=Comp(n1.shape[1])
    op=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-2)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(op,T_max=epochs); lf=nn.BCEWithLogitsLoss()
    # объединяем real train фолда + псевдо-тест (с меньшим label smoothing доверия: ls_pl=0.15)
    e1=torch.cat([tt(n1[tri]),P1]); e2=torch.cat([tt(n2[tri]),P2])
    Y=torch.cat([tt(y[tri]*(1-ls)+.5*ls), PY*(1-0.15)+0.5*0.15])
    v1,v2=tt(n1[vai]),tt(n2[vai]); nb=len(Y); best=-1
    for ep in range(epochs):
        m.train(); perm=torch.randperm(nb)
        for i in range(0,nb,256):
            b=perm[i:i+256]; op.zero_grad()
            if torch.rand(1).item()<.5: out=m(e1[b],e2[b]); tg=Y[b]
            else: out=m(e2[b],e1[b]); tg=1-Y[b]
            lf(out,tg).backward(); op.step()
        sc.step(); m.eval()
        with torch.no_grad(): pv=.5*(torch.sigmoid(m(v1,v2))+(1-torch.sigmoid(m(v2,v1)))).numpy()
        a=roc_auc_score(y[vai],pv)
        if a>best:
            best=a; ov=pv
            with torch.no_grad(): tpp=.5*(torch.sigmoid(m(U1,U2))+(1-torch.sigmoid(m(U2,U1)))).numpy()
    oof[vai]=ov; tp+=tpp/5; print(f"  fold{f} {best:.4f}",flush=True)
au=roc_auc_score(y,oof); print(f"== pseudoL OOF AUC = {au:.4f} ==")
np.savez("artifacts/oof_pseudoL.npz",oof=oof,test=tp,y=y.astype(np.int8),idx=idx.astype(np.int32),auc=au)
