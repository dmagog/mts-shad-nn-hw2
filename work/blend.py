"""
Blend / stack OOF predictions from multiple models and emit a final submission.
Decisions are made on OOF ROC-AUC (the competition metric).

Inputs: artifacts/oof_<tag>.npz files, each with keys: oof, y, test, idx.
(ft_<tag>.npz from fine-tuning has val_pred/val_idx/val_y/test_pred -> we expand
 val preds into a full-length OOF vector via the saved val_idx.)

Methods compared:
  - each single model
  - simple average (rank-averaged, AUC-friendly)
  - non-negative weight search (coordinate ascent on OOF AUC)
  - logistic-regression stack on OOF preds

Usage: python scripts/blend.py oof_dinoL oof_clipL ft_ftdino ... [--tag final]
"""
import sys, glob, argparse, numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def load_one(tag, n_train_hint=None):
    """Return (oof_full, y_full, test_pred, idx) for a tag, handling ft_ files."""
    path = f"artifacts/{tag}.npz" if not tag.endswith(".npz") else tag
    z = np.load(path)
    if "oof" in z:  # comparator output: full OOF
        return z["oof"].astype(float), z["y"].astype(float), z["test"].astype(float), z["idx"].astype(int)
    # fine-tune output: only a validation subset -> build full OOF with NaN elsewhere
    n = n_train_hint
    oof = np.full(n, np.nan)
    oof[z["val_idx"].astype(int)] = z["val_pred"].astype(float)
    y = np.full(n, np.nan)
    y[z["val_idx"].astype(int)] = z["val_y"].astype(float)
    return oof, y, z["test_pred"].astype(float), None


def rank01(x):
    m = ~np.isnan(x)
    out = np.full_like(x, np.nan, dtype=float)
    out[m] = (rankdata(x[m]) - 1) / max(1, (m.sum() - 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()

    # establish full y from any comparator file
    base = None
    for t in a.tags:
        z = np.load(f"artifacts/{t}.npz")
        if "oof" in z:
            base = (z["y"].astype(float), z["idx"].astype(int)); break
    if base is None:
        raise SystemExit("need at least one comparator oof_*.npz for full labels/idx")
    y_full, idx = base
    n = len(y_full)

    oofs, tests, names = [], [], []
    for t in a.tags:
        oof, yt, test, ix = load_one(t, n)
        if len(test) != len(idx):
            print(f"skip {t}: test len {len(test)} != {len(idx)}"); continue
        oofs.append(oof); tests.append(test); names.append(t)
        m = ~np.isnan(oof)
        print(f"{t:20s} OOF AUC = {roc_auc_score(y_full[m], oof[m]):.4f}  (cover {m.mean():.2f})")

    # full-coverage models only for blending search (fine-tune partial OOF used via its own val)
    full = [i for i in range(len(oofs)) if not np.isnan(oofs[i]).any()]
    if len(full) >= 2:
        R = np.stack([rank01(oofs[i]) for i in full], 1)  # rank-normalized OOF
        Rte = np.stack([rankdata(tests[i]) / len(tests[i]) for i in full], 1)
        yv = y_full
        # simple average
        auc_avg = roc_auc_score(yv, R.mean(1))
        print(f"\nsimple rank-avg ({len(full)} models) OOF AUC = {auc_avg:.4f}")
        # coordinate-ascent non-negative weights
        w = np.ones(len(full)) / len(full)
        for _ in range(200):
            for j in range(len(full)):
                best_wj, best_auc = w[j], roc_auc_score(yv, R @ w)
                for cand in np.linspace(0, 1, 21):
                    w2 = w.copy(); w2[j] = cand
                    if w2.sum() == 0: continue
                    au = roc_auc_score(yv, R @ (w2 / w2.sum()))
                    if au > best_auc: best_auc, best_wj = au, cand
                w[j] = best_wj
            w = w / w.sum()
        auc_w = roc_auc_score(yv, R @ w)
        print(f"weighted blend OOF AUC = {auc_w:.4f}  weights={dict(zip([names[i] for i in full], np.round(w,3)))}")
        # logistic stack
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(R, yv)
        auc_lr = roc_auc_score(yv, lr.predict_proba(R)[:, 1])
        print(f"logreg stack OOF AUC = {auc_lr:.4f}")

        method, score = max([("avg", auc_avg), ("wts", auc_w), ("lr", auc_lr)], key=lambda x: x[1])
        if method == "avg": test_final = Rte.mean(1)
        elif method == "wts": test_final = Rte @ w
        else: test_final = lr.predict_proba(Rte)[:, 1]
        print(f"\n== BEST blend: {method}  OOF AUC = {score:.4f} ==")
    else:
        test_final = rankdata(tests[0]) / len(tests[0]); print("\nsingle model -> submission")

    import csv
    out = f"artifacts/sub_{a.tag}.csv"
    with open(out, "w", newline="") as f:
        w_ = csv.writer(f); w_.writerow(["index", "is_image1_better"])
        for i, p in zip(idx, test_final): w_.writerow([int(i), float(p)])
    print(f"SAVED {out}")


if __name__ == "__main__":
    main()
