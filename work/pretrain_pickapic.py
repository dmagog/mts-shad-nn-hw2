"""
Предобучение компаратора на Pick-a-Pic v1 validation (реальные человеческие
предпочтения пар генераций) -> fine-tune 5-fold на наших 8710 парах.
Фичи: DINOv2-small [CLS;mean] (совместимы с локальным artifacts/emb_dinoS.npz).
"""
import os, io, glob, time, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
MPS = torch.backends.mps.is_available()
FDEV = "mps" if MPS else "cpu"      # фичи на MPS
CDEV = "cpu"                        # компаратор на CPU (маленький, быстрее)
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", "30000"))

# ---------- 1. читаем внешние пары ----------
files = sorted(glob.glob("ext_data/**/*.parquet", recursive=True))
print("parquet-файлов:", len(files), flush=True)
b1, b2, ey = [], [], []
for f in files:
    pf = pq.ParquetFile(f)
    cols = [c for c in ["jpg_0", "jpg_1", "label_0", "label_1", "are_different", "has_label"] if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        d = batch.to_pydict()
        for i in range(len(d["jpg_0"])):
            l0 = d.get("label_0", [None])[i]
            if l0 not in (0.0, 1.0):  # пропускаем ничьи
                continue
            b1.append(d["jpg_0"][i]); b2.append(d["jpg_1"][i]); ey.append(int(l0 == 1.0))
        if len(b1) >= MAX_PAIRS: break
    if len(b1) >= MAX_PAIRS: break
ey = np.array(ey, np.float32)
print(f"внешних пар: {len(b1)}  P(img1 лучше)={ey.mean():.3f}", flush=True)

# ---------- 2. DINOv2-small фичи ----------
from transformers import AutoModel, AutoImageProcessor
W = "artifacts/hf_models/dinov2-small"
proc = AutoImageProcessor.from_pretrained(W)
model = AutoModel.from_pretrained(W).to(FDEV).eval()
def dec(b):
    try: return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception: return Image.new("RGB", (64, 64), (128, 128, 128))
@torch.no_grad()
def feat(bl, bs=48):
    ex = ThreadPoolExecutor(max_workers=6); out = []
    t0 = time.time()
    for i in range(0, len(bl), bs):
        pils = list(ex.map(dec, bl[i:i + bs]))
        px = proc(images=pils, return_tensors="pt")["pixel_values"].to(FDEV)
        o = model(pixel_values=px)
        g = torch.cat([o.pooler_output, o.last_hidden_state[:, 1:, :].mean(1)], -1)
        out.append(g.float().cpu().numpy().astype(np.float16))
        if (i // bs) % 40 == 0: print(f"  {i}/{len(bl)} {i/max(time.time()-t0,1):.0f}/s", flush=True)
    return np.concatenate(out)
t0 = time.time(); E1 = feat(b1); E2 = feat(b2); del b1, b2
print("внешние фичи:", E1.shape, round(time.time() - t0), "s", flush=True)
np.savez_compressed("artifacts/ext_pickapic_dinoS.npz", E1=E1, E2=E2, y=ey.astype(np.int8))

# ---------- 3. предобучение компаратора ----------
class Comp(nn.Module):
    def __init__(s, d, h=512, p=0.3):
        super().__init__()
        s.proj = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p),
                               nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p))
        s.sc = nn.Linear(h, 1)
        s.cp = nn.Sequential(nn.Linear(4 * h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))
        s.q = nn.Parameter(torch.tensor(1.0))
    def forward(s, e1, e2):
        z1, z2 = s.proj(e1), s.proj(e2)
        return (s.cp(torch.cat([z1, z2, z1 - z2, z1 * z2], -1)) + s.q * (s.sc(z1) - s.sc(z2))).squeeze(-1)

z = np.load("artifacts/emb_dinoS.npz")
pool = np.concatenate([z["train_img1"], z["train_img2"]]).astype(np.float32)
MU, SD = pool.mean(0), pool.std(0) + 1e-6
nz = lambda a: (a.astype(np.float32) - MU) / SD
n1, n2 = nz(z["train_img1"]), nz(z["train_img2"])
u1, u2 = nz(z["test_img1"]), nz(z["test_img2"])
y = z["train_y"].astype(np.float32); idx = z["test_index"].astype(int)
NTR, NTE = len(y), len(idx)
tt = lambda a, dev=CDEV: torch.tensor(a, dtype=torch.float32, device=dev)

x1, x2, ye_t = tt(nz(E1)), tt(nz(E2)), tt(ey)
torch.manual_seed(0); net = Comp(n1.shape[1]).to(CDEV)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss()
nb = len(ey); t0 = time.time()
for ep in range(6):
    net.train(); perm = torch.randperm(nb)
    for i in range(0, nb, 512):
        b = perm[i:i + 512]; opt.zero_grad()
        if torch.rand(1).item() < .5: out = net(x1[b], x2[b]); tg = ye_t[b]
        else: out = net(x2[b], x1[b]); tg = 1 - ye_t[b]
        lf(out, tg).backward(); opt.step()
    with torch.no_grad():
        pa = roc_auc_score(ey, torch.sigmoid(net(x1, x2)).numpy())
    print(f"  pretrain ep{ep} ext-AUC={pa:.4f} ({round(time.time()-t0)}s)", flush=True)
PRE = {k: v.clone() for k, v in net.state_dict().items()}
torch.save(PRE, "artifacts/comp_pretrained_dinoS.pt")

# ---------- 4. fine-tune 5-fold на наших ----------
def run_cv(init, tag, epochs=50, lr=8e-4, ls=0.05, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(NTR); tp = np.zeros(NTE)
    U1, U2 = tt(u1), tt(u2)
    for f, (tri, vai) in enumerate(skf.split(n1, y)):
        torch.manual_seed(seed + f); m = Comp(n1.shape[1]).to(CDEV)
        if init is not None: m.load_state_dict(init)
        op = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=epochs); lf2 = nn.BCEWithLogitsLoss()
        e1, e2, Y = tt(n1[tri]), tt(n2[tri]), tt(y[tri] * (1 - ls) + .5 * ls)
        v1, v2 = tt(n1[vai]), tt(n2[vai]); nb2 = len(Y); best = -1
        for ep in range(epochs):
            m.train(); perm = torch.randperm(nb2)
            for i in range(0, nb2, 256):
                b = perm[i:i + 256]; op.zero_grad()
                if torch.rand(1).item() < .5: out = m(e1[b], e2[b]); tg = Y[b]
                else: out = m(e2[b], e1[b]); tg = 1 - Y[b]
                lf2(out, tg).backward(); op.step()
            sc.step(); m.eval()
            with torch.no_grad():
                pv = .5 * (torch.sigmoid(m(v1, v2)) + (1 - torch.sigmoid(m(v2, v1)))).numpy()
            a = roc_auc_score(y[vai], pv)
            if a > best:
                best = a; ov = pv
                with torch.no_grad():
                    tpp = .5 * (torch.sigmoid(m(U1, U2)) + (1 - torch.sigmoid(m(U2, U1)))).numpy()
        oof[vai] = ov; tp += tpp / 5
        print(f"  [{tag}] fold{f} AUC={best:.4f}", flush=True)
    au = roc_auc_score(y, oof)
    print(f"== {tag} OOF AUC = {au:.4f} ==", flush=True)
    np.savez(f"artifacts/oof_{tag}.npz", oof=oof, test=tp, y=y.astype(np.int8), idx=idx.astype(np.int32), auc=au)
    return au

run_cv(PRE, "preS")
print("ALL DONE", flush=True)
