"""Cheap classical image-quality metrics on resized train data; check raw AUC vs label."""
import io, numpy as np, pyarrow.parquet as pq
from PIL import Image
from sklearn.metrics import roc_auc_score

def feats(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    a = np.asarray(im, np.float32)/255.0
    g = a.mean(2)
    # sharpness: variance of Laplacian (numpy 4-neighbour)
    lap = (-4*g[1:-1,1:-1]+g[:-2,1:-1]+g[2:,1:-1]+g[1:-1,:-2]+g[1:-1,2:])
    sharp = lap.var()
    # colorfulness (Hasler-Susstrunk)
    R,G,B = a[...,0],a[...,1],a[...,2]
    rg=R-G; yb=0.5*(R+G)-B
    colorful=np.sqrt(rg.var()+yb.var())+0.3*np.sqrt(rg.mean()**2+yb.mean()**2)
    contrast=g.std()
    bright=g.mean()
    sat=(a.max(2)-a.min(2)).mean()
    edge=np.abs(np.diff(g,axis=0)).mean()+np.abs(np.diff(g,axis=1)).mean()
    return [sharp, colorful, contrast, bright, sat, edge]

t = pq.read_table("data/train_512.parquet", columns=["image_1","image_2","is_image1_better"]).to_pydict()
y = np.array(t["is_image1_better"])
N = len(y)
names=["sharp","colorful","contrast","bright","sat","edge"]
F1=np.zeros((N,6)); F2=np.zeros((N,6))
for i in range(N):
    F1[i]=feats(t["image_1"][i]); F2[i]=feats(t["image_2"][i])
    if i%2000==0: print("  ",i,"/",N,flush=True)
D = F1-F2
print("\n=== raw AUC of (metric_img1 - metric_img2) vs label ===")
for j,n in enumerate(names):
    print(f"  {n:9s} AUC={roc_auc_score(y, D[:,j]):.4f}  (1-AUC={roc_auc_score(y, -D[:,j]):.4f})")
np.savez("artifacts/classical_iqa.npz", F1=F1, F2=F2, y=y, names=names)
print("saved artifacts/classical_iqa.npz")
