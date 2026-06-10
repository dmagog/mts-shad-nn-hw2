"""
Извлечь patch-токены DINOv2 (не CLS!) для patch-уровневого компаратора.
Патчи img1 и img2 семантически ВЫРОВНЕНЫ (один сюжет), поэтому локальная
разность P1_i - P2_i несёт сигнал о качестве (в отличие от разности CLS,
которая схлопывается). Пулим сетку патчей до GxG для управляемого размера.

Выход: {split}_img1/img2 float16 [N, G*G, D], + y/index/aux.
"""
import io, os, sys, time, numpy as np, torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if torch.cuda.is_available(): DEV = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): DEV = "mps"
else: DEV = "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "artifacts/hf_models/dinov2-small"
G = int(sys.argv[2]) if len(sys.argv) > 2 else 8      # выходная сетка GxG
OUT = sys.argv[3] if len(sys.argv) > 3 else "artifacts/patch_dinoS.npz"
BS = 48

from transformers import AutoModel, AutoImageProcessor
proc = AutoImageProcessor.from_pretrained(WEIGHTS)
model = AutoModel.from_pretrained(WEIGHTS).to(DEV).eval().to(DT)
D = model.config.hidden_size
print(f"device={DEV} dim={D} grid={G}x{G}", flush=True)

def decode(b):
    try: return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception: return Image.new("RGB", (64, 64), (128, 128, 128))

@torch.no_grad()
def patches(pil_list):
    px = proc(images=pil_list, return_tensors="pt")["pixel_values"].to(DEV).to(DT)
    out = model(pixel_values=px).last_hidden_state[:, 1:, :]        # [B, P, D] (без CLS)
    B, P, d = out.shape; s = int(round(P ** 0.5))
    grid = out.transpose(1, 2).reshape(B, d, s, s)                  # [B, D, s, s]
    grid = F.adaptive_avg_pool2d(grid.float(), (G, G))             # [B, D, G, G]
    return grid.reshape(B, d, G * G).transpose(1, 2)               # [B, G*G, D]

def run(path, has_label):
    pf = pq.ParquetFile(path); n = pf.metadata.num_rows
    cols = ["image_1", "image_2", "img1_w", "img1_h", "img1_fmt", "img1_nbytes"] + \
           (["is_image1_better"] if has_label else ["index"])
    e1, e2, aux, lab, idx = [], [], [], [], []
    ex = ThreadPoolExecutor(max_workers=6); done, t0 = 0, time.time()
    for batch in pf.iter_batches(batch_size=BS, columns=cols):
        d = batch.to_pydict()
        im1 = list(ex.map(decode, d["image_1"])); im2 = list(ex.map(decode, d["image_2"]))
        e1.append(patches(im1).cpu().numpy().astype(np.float16))
        e2.append(patches(im2).cpu().numpy().astype(np.float16))
        w = np.array(d["img1_w"], np.float32); h = np.array(d["img1_h"], np.float32)
        isj = np.array([1.0 if f == "JPEG" else 0.0 for f in d["img1_fmt"]], np.float32)
        nb = np.log1p(np.array(d["img1_nbytes"], np.float32))
        aux.append(np.stack([w, h, isj, nb], 1))
        if has_label: lab.extend(int(v) for v in d["is_image1_better"])
        else: idx.extend(int(v) for v in d["index"])
        done += len(im1); print(f"  {done}/{n} {done/(time.time()-t0):.0f}/s", flush=True)
    ex.shutdown()
    o = dict(img1=np.concatenate(e1), img2=np.concatenate(e2), aux=np.concatenate(aux))
    if has_label: o["y"] = np.array(lab, np.int8)
    else: o["index"] = np.array(idx, np.int32)
    return o

print("== TRAIN ==", flush=True); tr = run("data/train_512.parquet", True)
print("== TEST ==", flush=True); te = run("data/test_512.parquet", False)
np.savez_compressed(OUT,
    train_img1=tr["img1"], train_img2=tr["img2"], train_aux=tr["aux"], train_y=tr["y"],
    test_img1=te["img1"], test_img2=te["img2"], test_aux=te["aux"], test_index=te["index"])
print(f"SAVED {OUT} shape={tr['img1'].shape}", flush=True)
