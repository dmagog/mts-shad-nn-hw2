"""
Custom PyTorch preference comparator on frozen backbone embeddings.

Architecture (hand-written) -- a Siamese pairwise-preference network:
  - shared projector  phi: e -> z   (LN + GELU + dropout MLP)
  - scalar quality head  s(z)  -> antisymmetric ranking term  s(z1) - s(z2)
  - pairwise comparator on [z1, z2, z1-z2, z1*z2]  (captures position bias too)
  - aux head on image_1 metadata (size/format)
  logit = comparator + qscale*(s1 - s2) + aux

Validation: StratifiedKFold, metric = ROC-AUC (competition metric).
Produces OOF preds, averaged test preds, and a submission CSV (continuous probs).

Usage:
  python scripts/train_comparator.py --emb artifacts/emb_dinov2L.npz \
        [--emb artifacts/emb_clipL.npz] --tag dinoclip --folds 5 --epochs 80
"""
import argparse, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# tiny model -> CPU is faster than MPS here (avoids per-kernel launch overhead); CUDA if present
import os as _os
DEV = "cuda" if torch.cuda.is_available() else _os.environ.get("COMP_DEV", "cpu")


def load_embs(paths):
    """Concatenate embeddings from multiple backbones along feature dim."""
    tr1, tr2, te1, te2 = [], [], [], []
    y = aux_tr = aux_te = idx = None
    for p in paths:
        z = np.load(p)
        # per-model standardization fit on train (img1+img2 pooled)
        pool = np.concatenate([z["train_img1"], z["train_img2"]]).astype(np.float32)
        mu, sd = pool.mean(0), pool.std(0) + 1e-6
        norm = lambda a: ((a.astype(np.float32) - mu) / sd)
        tr1.append(norm(z["train_img1"])); tr2.append(norm(z["train_img2"]))
        te1.append(norm(z["test_img1"]));  te2.append(norm(z["test_img2"]))
        if y is None:
            y = z["train_y"].astype(np.float32)
            idx = z["test_index"].astype(int)
            aux_tr = z["train_aux"].astype(np.float32)
            aux_te = z["test_aux"].astype(np.float32)
    # standardize aux on train
    amu, asd = aux_tr.mean(0), aux_tr.std(0) + 1e-6
    aux_tr = (aux_tr - amu) / asd; aux_te = (aux_te - amu) / asd
    return (np.concatenate(tr1, 1), np.concatenate(tr2, 1),
            np.concatenate(te1, 1), np.concatenate(te2, 1),
            y, aux_tr, aux_te, idx)


class PreferenceNet(nn.Module):
    def __init__(self, d_emb, d_aux, hidden=512, p=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_emb, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p),
        )
        self.score = nn.Linear(hidden, 1)
        self.comp = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )
        self.auxh = nn.Sequential(nn.Linear(d_aux, 32), nn.GELU(), nn.Linear(32, 1))
        self.qscale = nn.Parameter(torch.tensor(1.0))

    def forward(self, e1, e2, aux):
        z1, z2 = self.proj(e1), self.proj(e2)
        s = self.qscale * (self.score(z1) - self.score(z2))
        c = self.comp(torch.cat([z1, z2, z1 - z2, z1 * z2], dim=-1))
        return (c + s + self.auxh(aux)).squeeze(-1)


def train_fold(Xtr, ytr, Xva, Xte, d_emb, d_aux, epochs, lr, wd, bs, seed):
    torch.manual_seed(seed)
    net = PreferenceNet(d_emb, d_aux).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    e1, e2, aux = (torch.tensor(Xtr[k]).to(DEV) for k in ("e1", "e2", "aux"))
    yt = torch.tensor(ytr).to(DEV)
    n = len(yt)
    best_va, best_state, best_te = -1, None, None
    v1, v2, va = (torch.tensor(Xva[k]).to(DEV) for k in ("e1", "e2", "aux"))
    t1, t2, ta = (torch.tensor(Xte[k]).to(DEV) for k in ("e1", "e2", "aux"))
    yva = Xva["y"]
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            out = net(e1[b], e2[b], aux[b])
            loss = lossf(out, yt[b])
            loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(v1, v2, va)).cpu().numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best_va:
            best_va = auc
            with torch.no_grad():
                best_te = torch.sigmoid(net(t1, t2, ta)).cpu().numpy()
            best_oof = pv
    return best_va, best_oof, best_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", action="append", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tr1, tr2, te1, te2, y, aux_tr, aux_te, idx = load_embs(a.emb)
    d_emb, d_aux = tr1.shape[1], aux_tr.shape[1]
    print(f"device={DEV}  d_emb={d_emb} d_aux={d_aux}  n_train={len(y)} n_test={len(idx)}  pos={y.mean():.3f}")

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    oof = np.zeros(len(y)); test_pred = np.zeros(len(idx)); aucs = []
    for f, (tri, vai) in enumerate(skf.split(tr1, y)):
        Xtr = dict(e1=tr1[tri], e2=tr2[tri], aux=aux_tr[tri])
        Xva = dict(e1=tr1[vai], e2=tr2[vai], aux=aux_tr[vai], y=y[vai])
        Xte = dict(e1=te1, e2=te2, aux=aux_te)
        va, ov, tp = train_fold(Xtr, y[tri], Xva, Xte, d_emb, d_aux,
                                a.epochs, a.lr, a.wd, a.bs, a.seed + f)
        oof[vai] = ov; test_pred += tp / a.folds; aucs.append(va)
        print(f"  fold{f} AUC={va:.4f}")
    oof_auc = roc_auc_score(y, oof)
    print(f"== OOF AUC = {oof_auc:.4f}  (folds {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}) ==")

    np.savez(f"artifacts/oof_{a.tag}.npz", oof=oof, y=y, test=test_pred, idx=idx, auc=oof_auc)
    import csv
    with open(f"artifacts/sub_{a.tag}.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv); w.writerow(["index", "is_image1_better"])
        for i, p in zip(idx, test_pred): w.writerow([int(i), float(p)])
    print(f"SAVED artifacts/sub_{a.tag}.csv  artifacts/oof_{a.tag}.npz")


if __name__ == "__main__":
    main()
