"""
End-to-end fine-tuned Siamese preference network (the headline custom architecture).
Runs on the REMOTE GPU (RTX 2070, 8GB). Loads a pretrained vision backbone from a
local dir (HF CDN stalls on the remote) and fine-tunes it together with a
hand-written antisymmetric comparison head, directly optimizing the pairwise task.

  SiamesePreferenceNet:
    backbone (shared)  ->  per-image feature  e = [CLS ; mean(patch)]
    head: proj -> z ; scalar quality s(z) ; comparator on [z1,z2,z1-z2,z1*z2]
    logit = comp + qscale*(s1 - s2)

Validation: stratified hold-out, metric = ROC-AUC. Saves OOF/val + test probs.
8GB-friendly: fp16 autocast, optional freezing of early backbone blocks, grad accum.
"""
import argparse, io, os, time, numpy as np, torch, torch.nn as nn
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PairDS(Dataset):
    def __init__(self, path, res, rows=None, train=False):
        self.t = pq.read_table(path, columns=["image_1", "image_2"] +
                               (["is_image1_better"] if "train" in path else ["index"]))
        self.b1 = self.t.column("image_1").to_pylist()
        self.b2 = self.t.column("image_2").to_pylist()
        self.has_y = "is_image1_better" in self.t.column_names
        self.y = self.t.column("is_image1_better").to_pylist() if self.has_y else \
                 self.t.column("index").to_pylist()
        self.rows = rows if rows is not None else list(range(len(self.b1)))
        self.res, self.train = res, train

    def __len__(self): return len(self.rows)

    def _img(self, b):
        im = Image.open(io.BytesIO(b)).convert("RGB").resize((self.res, self.res), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)

    def __getitem__(self, i):
        r = self.rows[i]
        x1, x2 = self._img(self.b1[r]), self._img(self.b2[r])
        if self.train and torch.rand(1).item() < 0.5:  # consistent hflip (preference-invariant)
            x1, x2 = torch.flip(x1, [2]), torch.flip(x2, [2])
        return x1, x2, float(self.y[r])


class SiamesePreferenceNet(nn.Module):
    def __init__(self, backbone, d_emb, hidden=512, p=0.3):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(d_emb, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p))
        self.score = nn.Linear(hidden, 1)
        self.comp = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, 1))
        self.qscale = nn.Parameter(torch.tensor(1.0))

    def encode(self, x):
        out = self.backbone(pixel_values=x)
        return torch.cat([out.pooler_output, out.last_hidden_state[:, 1:, :].mean(1)], -1)

    def forward(self, x1, x2):
        z1, z2 = self.proj(self.encode(x1)), self.proj(self.encode(x2))
        s = self.qscale * (self.score(z1) - self.score(z2))
        c = self.comp(torch.cat([z1, z2, z1 - z2, z1 * z2], -1))
        return (c + s).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="hf_models/dinov2-large")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_bb", type=float, default=1e-5)
    ap.add_argument("--freeze", type=int, default=16, help="freeze first N backbone layers")
    ap.add_argument("--grad_ckpt", type=int, default=1, help="gradient checkpointing (8GB safe)")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 safest on Windows)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--tag", default="ftdino")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    from transformers import AutoModel
    bb = AutoModel.from_pretrained(a.weights)
    d_emb = bb.config.hidden_size * 2
    # freeze embeddings + first N encoder layers
    for p in bb.embeddings.parameters(): p.requires_grad = False
    if hasattr(bb.encoder, "layer"):
        for L in bb.encoder.layer[:a.freeze]:
            for p in L.parameters(): p.requires_grad = False
    if a.grad_ckpt and hasattr(bb, "gradient_checkpointing_enable"):
        bb.gradient_checkpointing_enable()
    net = SiamesePreferenceNet(bb, d_emb).to(DEV)
    net.register_buffer("imean", IMAGENET_MEAN.to(DEV)); net.register_buffer("istd", IMAGENET_STD.to(DEV))

    # stratified hold-out
    y_all = np.array(pq.read_table(f"{a.data_dir}/train_512.parquet",
                                   columns=["is_image1_better"]).column(0).to_pylist())
    tr_idx, va_idx = train_test_split(np.arange(len(y_all)), test_size=a.val_frac,
                                      stratify=y_all, random_state=a.seed)
    dtr = DataLoader(PairDS(f"{a.data_dir}/train_512.parquet", a.res, tr_idx.tolist(), True),
                     batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    dva = DataLoader(PairDS(f"{a.data_dir}/train_512.parquet", a.res, va_idx.tolist()),
                     batch_size=a.bs, num_workers=a.workers, pin_memory=True)
    dte = DataLoader(PairDS(f"{a.data_dir}/test_512.parquet", a.res),
                     batch_size=a.bs, num_workers=a.workers, pin_memory=True)

    head_p = [p for n, p in net.named_parameters() if not n.startswith("backbone") and p.requires_grad]
    bb_p = [p for n, p in net.named_parameters() if n.startswith("backbone") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head_p, "lr": a.lr_head},
                             {"params": bb_p, "lr": a.lr_bb}], weight_decay=1e-2)
    steps = a.epochs * (len(dtr) // a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_head, a.lr_bb],
                                                total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    def norm(x): return (x.to(DEV).float() - net.imean) / net.istd

    @torch.no_grad()
    def infer(dl, want_y):
        net.eval(); ps, ys = [], []
        for x1, x2, yb in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(norm(x1), norm(x2))
            ps.append(torch.sigmoid(out).float().cpu().numpy())
            if want_y: ys.append(yb.numpy())
        return (np.concatenate(ps), np.concatenate(ys) if want_y else None)

    best = -1
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); opt.zero_grad()
        for it, (x1, x2, yb) in enumerate(dtr):
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(norm(x1), norm(x2))
                loss = lossf(out, yb.to(DEV)) / a.accum
            scaler.scale(loss).backward()
            if (it + 1) % a.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        pv, yv = infer(dva, True); auc = roc_auc_score(yv, pv)
        print(f"epoch {ep} val_AUC={auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if auc > best:
            best = auc
            pt, _ = infer(dte, False)
            np.savez(f"artifacts/ft_{a.tag}.npz", val_pred=pv, val_idx=va_idx, val_y=yv,
                     test_pred=pt, auc=auc)
    print(f"== BEST val AUC = {best:.4f} -> artifacts/ft_{a.tag}.npz ==", flush=True)


if __name__ == "__main__":
    main()
