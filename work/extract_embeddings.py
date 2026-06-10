"""
Extract frozen pretrained-backbone embeddings for every image in train/test.
Runs on the REMOTE GPU box (RTX 2070). Loads weights from a local dir
(HF CDN stalls on the remote, so weights were scp'd from the Mac).

Outputs one .npz per backbone with:
  {split}_img1, {split}_img2  : float16 [N, D] embeddings
  train_y                     : int8   [N]  labels (is_image1_better)
  test_index                  : int32  [N]
  {split}_aux                 : float32 [N, 4]  (img1_w, img1_h, img1_is_jpeg, log1p(img1_nbytes))

Usage:
  python extract_embeddings.py --model dinov2 --weights hf_models/dinov2-large \
      --data_dir data --out artifacts/emb_dinov2L.npz --res 224 --bs 64
"""
import argparse, io, os, time
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if torch.cuda.is_available():
    DEV = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEV = "mps"
else:
    DEV = "cpu"
HALF = (DEV == "cuda")  # fp16 only on CUDA; MPS/CPU use fp32 (half is flaky on MPS)
DT = torch.float16 if HALF else torch.float32


def decode(b):
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return Image.new("RGB", (64, 64), (128, 128, 128))


def build_model(kind, weights, res):
    if kind == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(weights)
        if res:
            proc.size = {"shortest_edge": res}
            proc.crop_size = {"height": res, "width": res}
        model = AutoModel.from_pretrained(weights).to(DEV).eval().to(DT)

        @torch.no_grad()
        def feat(pil_list):
            px = proc(images=pil_list, return_tensors="pt")["pixel_values"].to(DEV).to(DT)
            out = model(pixel_values=px)
            cls = out.pooler_output                      # [B, H]
            patch = out.last_hidden_state[:, 1:, :].mean(1)  # [B, H]
            return torch.cat([cls, patch], dim=-1)
        return feat, proc

    if kind == "clip":
        from transformers import CLIPModel, CLIPImageProcessor
        proc = CLIPImageProcessor.from_pretrained(weights)
        if res:
            proc.size = {"shortest_edge": res}
            proc.crop_size = {"height": res, "width": res}
        model = CLIPModel.from_pretrained(weights).to(DEV).eval().to(DT)

        @torch.no_grad()
        def feat(pil_list):
            px = proc(images=pil_list, return_tensors="pt")["pixel_values"].to(DEV).to(DT)
            # canonical CLIP image embedding: vision pooler -> visual_projection (version-robust)
            vout = model.vision_model(pixel_values=px)
            emb = model.visual_projection(vout.pooler_output)   # [B, 768] projected
            emb = torch.nn.functional.normalize(emb, dim=-1)
            return emb
        return feat, proc

    raise ValueError(kind)


def run_split(path, feat, bs, has_label):
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    cols = ["image_1", "image_2", "img1_w", "img1_h", "img1_fmt", "img1_nbytes"]
    cols += ["is_image1_better"] if has_label else ["index"]
    e1, e2, aux, lab, idx = [], [], [], [], []
    done, t0 = 0, time.time()
    ex = ThreadPoolExecutor(max_workers=6)
    for batch in pf.iter_batches(batch_size=bs, columns=cols):
        d = batch.to_pydict()
        im1 = list(ex.map(decode, d["image_1"]))
        im2 = list(ex.map(decode, d["image_2"]))
        e1.append(feat(im1).float().cpu().numpy().astype(np.float16))
        e2.append(feat(im2).float().cpu().numpy().astype(np.float16))
        w = np.array(d["img1_w"], np.float32); h = np.array(d["img1_h"], np.float32)
        isj = np.array([1.0 if f == "JPEG" else 0.0 for f in d["img1_fmt"]], np.float32)
        nb = np.log1p(np.array(d["img1_nbytes"], np.float32))
        aux.append(np.stack([w, h, isj, nb], axis=1))
        if has_label: lab.extend(int(v) for v in d["is_image1_better"])
        else: idx.extend(int(v) for v in d["index"])
        done += len(im1)
        print(f"  {done}/{n}  {done/(time.time()-t0):.1f} img-rows/s", flush=True)
    ex.shutdown()
    out = dict(img1=np.concatenate(e1), img2=np.concatenate(e2), aux=np.concatenate(aux))
    if has_label: out["y"] = np.array(lab, np.int8)
    else: out["index"] = np.array(idx, np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dinov2", "clip"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    print(f"device={DEV} model={a.model} res={a.res} bs={a.bs}", flush=True)
    feat, _ = build_model(a.model, a.weights, a.res)

    print("== TRAIN ==", flush=True)
    tr = run_split(f"{a.data_dir}/train_512.parquet", feat, a.bs, True)
    print("== TEST ==", flush=True)
    te = run_split(f"{a.data_dir}/test_512.parquet", feat, a.bs, False)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        train_img1=tr["img1"], train_img2=tr["img2"], train_aux=tr["aux"], train_y=tr["y"],
        test_img1=te["img1"], test_img2=te["img2"], test_aux=te["aux"], test_index=te["index"],
    )
    print(f"SAVED {a.out}  dim={tr['img1'].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
