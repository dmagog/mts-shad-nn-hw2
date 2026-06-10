# teta-nn-2-2026: НАДЁЖНЫЙ kernel под дедлайн. DINOv2-large global + 8 NR-IQA метрик + стакинг + сабмит.
# (тяжёлое patch-сохранение убрано — оно роняло kernel; patch проверен локально, некритичен)
import os, io, time, subprocess, warnings, glob
warnings.filterwarnings("ignore")
def sh(c): print("+", c[:70], flush=True); subprocess.run(c, shell=True)
# GPU тут P100 (sm_60): образный torch cu128 несовместим -> ставим cu121
sh("pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1")
sh("pip -q install --no-deps pyiqa timm open_clip_torch einops 2>&1 | tail -1")

import numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata

OUT = "/kaggle/working"; DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", flush=True)
if DEV == "cuda": print("CUDA smoke", float((torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum()), flush=True)
DATA = os.path.dirname(glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0]); print("DATA", DATA, flush=True)

def load(split):
    cols = ["image_1", "image_2"] + (["is_image1_better"] if split == "train" else [])
    t = pq.read_table(f"{DATA}/{split}.parquet", columns=cols).to_pydict()
    return t["image_1"], t["image_2"], (np.array(t["is_image1_better"]) if split == "train" else None)
tr1, tr2, y = load("train"); te1, te2, _ = load("test")
NTR, NTE = len(tr1), len(te1); idx = np.arange(NTE)
print("train", NTR, "test", NTE, "pos", round(float(y.mean()), 3), flush=True)
def decode(b):
    try: return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception: return Image.new("RGB", (64, 64), (128, 128, 128))
def batched(bl, bs=32):
    ex = ThreadPoolExecutor(max_workers=8)
    for i in range(0, len(bl), bs): yield list(ex.map(decode, bl[i:i + bs]))

# ---------- 1. DINOv2-large global [CLS; mean-patch] ----------
from transformers import AutoModel, AutoImageProcessor
dino = AutoModel.from_pretrained("facebook/dinov2-large").to(DEV).eval().half()
dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
@torch.no_grad()
def dino_feat(bl):
    out = []
    for pils in batched(bl, 48):
        px = dproc(images=pils, return_tensors="pt")["pixel_values"].to(DEV).half()
        o = dino(pixel_values=px)
        g = torch.cat([o.pooler_output, o.last_hidden_state[:, 1:, :].mean(1)], -1)
        out.append(g.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)
t0 = time.time()
g_tr1, g_tr2 = dino_feat(tr1), dino_feat(tr2); g_te1, g_te2 = dino_feat(te1), dino_feat(te2)
print("DINOv2 done", round(time.time() - t0), "s", g_tr1.shape, flush=True)
np.savez_compressed(f"{OUT}/emb_dinoL_global.npz", train_img1=g_tr1, train_img2=g_tr2,
                    test_img1=g_te1, test_img2=g_te2, train_y=y.astype(np.int8), test_index=idx.astype(np.int32))
del dino; torch.cuda.empty_cache()

# ---------- 2. pyiqa NR-IQA метрики ----------
import pyiqa
IQA = ["musiq", "maniqa", "topiq_nr", "clipiqa+", "arniqa", "nima", "paq2piq", "liqe_mix"]
def iqa_run(bl, metric):
    o = np.zeros(len(bl), np.float32); i = 0
    for pils in batched(bl, 16):
        ts = torch.stack([torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0) for im in pils]).to(DEV)
        with torch.no_grad(): o[i:i + len(pils)] = metric(ts).flatten().float().cpu().numpy()
        i += len(pils)
    return o
I1, I2, J1, J2, used = [], [], [], [], []
for nm in IQA:
    try:
        m = pyiqa.create_metric(nm, device=DEV); s = -1.0 if m.lower_better else 1.0; t0 = time.time()
        a, b, c, d = s * iqa_run(tr1, m), s * iqa_run(tr2, m), s * iqa_run(te1, m), s * iqa_run(te2, m)
        au = roc_auc_score(y, a - b); print(f"  IQA {nm}: AUC={max(au,1-au):.4f} ({round(time.time()-t0)}s)", flush=True)
        I1.append(a); I2.append(b); J1.append(c); J2.append(d); used.append(nm)
        del m; torch.cuda.empty_cache()
        np.savez(f"{OUT}/iqa_scores.npz", train_img1=np.stack(I1, 1), train_img2=np.stack(I2, 1),
                 test_img1=np.stack(J1, 1), test_img2=np.stack(J2, 1), names=np.array(used),
                 train_y=y.astype(np.int8), test_index=idx.astype(np.int32))  # инкрементально сохраняем
    except Exception as e:
        print(f"  IQA {nm} FAILED: {str(e)[:70]}", flush=True)
print("IQA used:", used, flush=True)

# ---------- 3. Компаратор (global) + IQA-стакинг + финал ----------
class Comp(nn.Module):
    def __init__(s, d, h=512, p=0.3):
        super().__init__()
        s.proj = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p),
                               nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
        s.sc = nn.Linear(h, 1); s.cp = nn.Sequential(nn.Linear(4 * h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))
        s.q = nn.Parameter(torch.tensor(1.0))
    def forward(s, e1, e2):
        z1, z2 = s.proj(e1), s.proj(e2)
        return (s.cp(torch.cat([z1, z2, z1 - z2, z1 * z2], -1)) + s.q * (s.sc(z1) - s.sc(z2))).squeeze(-1)
def train_global(X1, X2, T1, T2, y, epochs=70, bs=256, ls=0.05, seed=42):
    pool = np.concatenate([X1, X2]).astype(np.float32); mu, sd = pool.mean(0), pool.std(0) + 1e-6
    n1, n2, u1, u2 = (X1 - mu) / sd, (X2 - mu) / sd, (T1 - mu) / sd, (T2 - mu) / sd
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); oof = np.zeros(len(y)); tp = np.zeros(len(T1))
    tt = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV); U1, U2 = tt(u1), tt(u2)
    for f, (tri, vai) in enumerate(skf.split(X1, y)):
        torch.manual_seed(seed + f); net = Comp(X1.shape[1]).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs); lf = nn.BCEWithLogitsLoss()
        e1, e2, Y = tt(n1[tri]), tt(n2[tri]), tt(y[tri] * (1 - ls) + .5 * ls); v1, v2 = tt(n1[vai]), tt(n2[vai]); nb = len(Y); best = -1
        for ep in range(epochs):
            net.train(); perm = torch.randperm(nb, device=DEV)
            for i in range(0, nb, bs):
                b = perm[i:i + bs]; opt.zero_grad()
                if torch.rand(1).item() < .5: out = net(e1[b], e2[b]); tg = Y[b]
                else: out = net(e2[b], e1[b]); tg = 1 - Y[b]
                lf(out, tg).backward(); opt.step()
            sch.step(); net.eval()
            with torch.no_grad(): pv = .5 * (torch.sigmoid(net(v1, v2)) + (1 - torch.sigmoid(net(v2, v1)))).cpu().numpy()
            a = roc_auc_score(y[vai], pv)
            if a > best:
                best = a; ov = pv
                with torch.no_grad(): tpp = .5 * (torch.sigmoid(net(U1, U2)) + (1 - torch.sigmoid(net(U2, U1)))).cpu().numpy()
        oof[vai] = ov; tp += tpp / 5
    return oof, tp, roc_auc_score(y, oof)

res = {}
og, tg, ag = train_global(g_tr1, g_tr2, g_te1, g_te2, y); print("GLOBAL comp OOF AUC =", round(ag, 4), flush=True); res["global"] = (og, tg)
if used:
    D, Dt = np.stack(I1, 1) - np.stack(I2, 1), np.stack(J1, 1) - np.stack(J2, 1)
    skf = StratifiedKFold(5, shuffle=True, random_state=42); oi = np.zeros(NTR); ti = np.zeros(NTE)
    for tri, vai in skf.split(D, y):
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(D[tri], y[tri]); oi[vai] = lr.predict_proba(D[vai])[:, 1]; ti += lr.predict_proba(Dt)[:, 1] / 5
    print("IQA stack OOF AUC =", round(roc_auc_score(y, oi), 4), flush=True); res["iqa"] = (oi, ti)
# финальный rank-стакинг
keys = list(res); rk = lambda a: rankdata(a) / len(a)
Ro = np.stack([rk(res[k][0]) for k in keys], 1); Rt = np.stack([rk(res[k][1]) for k in keys], 1)
skf = StratifiedKFold(5, shuffle=True, random_state=42); os_ = np.zeros(NTR); ts_ = np.zeros(NTE)
for tri, vai in skf.split(Ro, y):
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Ro[tri], y[tri]); os_[vai] = lr.predict_proba(Ro[vai])[:, 1]; ts_ += lr.predict_proba(Rt)[:, 1] / 5
print("=== STACK", keys, "OOF AUC =", round(roc_auc_score(y, os_), 4), "===", flush=True)
np.savez(f"{OUT}/oof_all.npz", **{f"oof_{k}": res[k][0] for k in keys}, **{f"test_{k}": res[k][1] for k in keys}, y=y, idx=idx, oof_stack=os_, test_stack=ts_)
import csv
with open(f"{OUT}/submission.csv", "w", newline="") as fc:
    w = csv.writer(fc); w.writerow(["index", "is_image1_better"]); [w.writerow([int(i), float(p)]) for i, p in zip(idx, ts_)]
print("SAVED submission.csv oof_all.npz iqa_scores.npz emb_dinoL_global.npz\nALL DONE", flush=True)
