"""
Patch-level антисимметричный компаратор (главный приём против потолка 0.64).
Идея (консенсус ресёрча): глобальный CLS-вектор в паре одного сюжета почти совпадает,
но КАЧЕСТВО различается ЛОКАЛЬНО. Патчи img1/img2 выровнены (один сюжет) → сравниваем
их по-регионно и агрегируем attention-пулингом.

  z = proj(patch)                        # per-patch проекция
  s_local = qscale*(score(z1)-score(z2)) + comp([z1,z2,z1-z2,z1*z2])   # локальное сравнение
  logit   = Σ_i attn_i * s_local_i  + aux(meta)                        # взвешенная агрегация
Антисимметрия + TTA-swap; RankNet-подобный BCE с label smoothing (метки шумные).

Usage: python scripts/train_patch_comparator.py --emb artifacts/patch_dinoS.npz --tag patchS
"""
import argparse, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"  # tiny model -> CPU fine


class PatchComparator(nn.Module):
    def __init__(self, d, d_aux, hidden=256, p=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p))
        self.score = nn.Linear(hidden, 1)                              # per-patch качество
        self.comp = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(hidden, 1))  # локальный компаратор
        self.attn = nn.Linear(hidden, 1)                               # важность патча
        self.auxh = nn.Sequential(nn.Linear(d_aux, 32), nn.GELU(), nn.Linear(32, 1))
        self.qscale = nn.Parameter(torch.tensor(1.0))

    def forward(self, P1, P2, aux):
        z1, z2 = self.proj(P1), self.proj(P2)                          # [B, G, h]
        s = self.qscale * (self.score(z1) - self.score(z2))            # [B, G, 1] антисимм.
        c = self.comp(torch.cat([z1, z2, z1 - z2, z1 * z2], -1))       # [B, G, 1]
        local = s + c                                                  # [B, G, 1]
        w = torch.softmax((self.attn(z1) + self.attn(z2)).squeeze(-1), dim=1)  # [B, G]
        agg = (local.squeeze(-1) * w).sum(1)                           # [B]
        return agg + self.auxh(aux).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--tag", default="patch")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--ls", type=float, default=0.05, help="label smoothing (шумные метки)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    z = np.load(a.emb)
    P1 = z["train_img1"].astype(np.float32); P2 = z["train_img2"].astype(np.float32)
    T1 = z["test_img1"].astype(np.float32);  T2 = z["test_img2"].astype(np.float32)
    y = z["train_y"].astype(np.float32); idx = z["test_index"].astype(int)
    # per-канальная стандартизация по всем патчам train
    pool = np.concatenate([P1, P2]).reshape(-1, P1.shape[-1])
    mu, sd = pool.mean(0), pool.std(0) + 1e-6
    norm = lambda A: (A - mu) / sd
    P1, P2, T1, T2 = norm(P1), norm(P2), norm(T1), norm(T2)
    amu = z["train_aux"].mean(0); asd = z["train_aux"].std(0) + 1e-6
    aux_tr = (z["train_aux"] - amu) / asd; aux_te = (z["test_aux"] - amu) / asd
    d, da = P1.shape[-1], aux_tr.shape[1]
    print(f"DEV={DEV} patches={P1.shape[1]} dim={d} n_tr={len(y)} pos={y.mean():.3f}")

    def tt(x): return torch.tensor(x, device=DEV)
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    oof = np.zeros(len(y)); test_pred = np.zeros(len(idx)); aucs = []
    Tt1, Tt2, Ta = tt(T1), tt(T2), tt(aux_te)
    for f, (tri, vai) in enumerate(skf.split(P1, y)):
        torch.manual_seed(a.seed + f)
        net = PatchComparator(d, da).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        lf = nn.BCEWithLogitsLoss()
        e1, e2, au, yy = tt(P1[tri]), tt(P2[tri]), tt(aux_tr[tri]), tt(y[tri])
        v1, v2, va = tt(P1[vai]), tt(P2[vai]), tt(aux_tr[vai])
        ys = yy * (1 - a.ls) + 0.5 * a.ls          # label smoothing
        n = len(yy); best = -1
        for ep in range(a.epochs):
            net.train(); perm = torch.randperm(n, device=DEV)
            for i in range(0, n, a.bs):
                b = perm[i:i + a.bs]
                opt.zero_grad()
                # symmetric-swap augmentation: half batch swapped with flipped target
                if torch.rand(1).item() < 0.5:
                    out = net(e1[b], e2[b], au[b]); tgt = ys[b]
                else:
                    out = net(e2[b], e1[b], au[b]); tgt = 1 - ys[b]
                loss = lf(out, tgt); loss.backward(); opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():  # TTA-swap на валидации
                pv = 0.5 * (torch.sigmoid(net(v1, v2, va)) + (1 - torch.sigmoid(net(v2, v1, va)))).cpu().numpy()
            auc = roc_auc_score(y[vai], pv)
            if auc > best:
                best = auc; ov = pv
                with torch.no_grad():
                    tp = 0.5 * (torch.sigmoid(net(Tt1, Tt2, Ta)) + (1 - torch.sigmoid(net(Tt2, Tt1, Ta)))).cpu().numpy()
        oof[vai] = ov; test_pred += tp / a.folds; aucs.append(best)
        print(f"  fold{f} AUC={best:.4f}")
    oof_auc = roc_auc_score(y, oof)
    print(f"== OOF AUC = {oof_auc:.4f} ({np.mean(aucs):.4f}+/-{np.std(aucs):.4f}) ==")
    np.savez(f"artifacts/oof_{a.tag}.npz", oof=oof, y=y, test=test_pred, idx=idx, auc=oof_auc)
    import csv
    with open(f"artifacts/sub_{a.tag}.csv", "w", newline="") as fc:
        w = csv.writer(fc); w.writerow(["index", "is_image1_better"])
        for i, p in zip(idx, test_pred): w.writerow([int(i), float(p)])
    print(f"SAVED artifacts/oof_{a.tag}.npz  artifacts/sub_{a.tag}.csv")


if __name__ == "__main__":
    main()
