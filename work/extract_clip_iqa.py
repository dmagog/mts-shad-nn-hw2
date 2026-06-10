"""
CLIP-IQA: zero-shot quality scoring via CLIP text-image similarity.
For each image, score it against antonym prompt pairs that target generative
failure modes (sharp vs blurry, realistic vs distorted, clean vs artifacts, ...).
Per pair: P(positive) = softmax(logit_scale * [sim_pos, sim_neg])[0].

Output (same layout as extract_embeddings, so it feeds the comparator as --emb):
  train_img1/img2, test_img1/img2 : float16 [N, n_pairs]
  train_y, test_index, *_aux
Runs on the REMOTE GPU. Needs CLIP tokenizer files in the weights dir.
"""
import argparse, io, os, time, numpy as np, torch
import pyarrow.parquet as pq
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
DEV = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
       and torch.backends.mps.is_available() else "cpu")
HALF = DEV == "cuda"; DT = torch.float16 if HALF else torch.float32

PAIRS = [
    ("a high quality image", "a low quality image"),
    ("a sharp, detailed image", "a blurry, low-detail image"),
    ("a realistic photo", "a distorted, unrealistic image"),
    ("a beautiful, aesthetically pleasing image", "an ugly, unpleasant image"),
    ("a well-composed image", "a poorly composed image"),
    ("an image with correct anatomy and proportions", "an image with deformed, broken anatomy"),
    ("a clean image without artifacts", "an image with visual artifacts and glitches"),
    ("a coherent, sensible image", "a nonsensical, garbled image"),
    ("a photorealistic image", "a fake-looking, artificial image"),
    ("correctly rendered, readable text", "garbled, misspelled text"),
]


def decode(b):
    try: return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception: return Image.new("RGB", (64, 64), (128, 128, 128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="hf_models/clip-large")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="artifacts/emb_clipiqa.npz")
    ap.add_argument("--bs", type=int, default=96)
    a = ap.parse_args()
    from transformers import CLIPModel, CLIPImageProcessor, AutoTokenizer
    proc = CLIPImageProcessor.from_pretrained(a.weights)
    tok = AutoTokenizer.from_pretrained(a.weights)
    model = CLIPModel.from_pretrained(a.weights).to(DEV).eval().to(DT)
    logit_scale = model.logit_scale.exp().detach()
    print(f"device={DEV} pairs={len(PAIRS)}", flush=True)

    # text embeddings (pos then neg, interleaved) -> [2P, D] normalized
    prompts = [p for pair in PAIRS for p in pair]
    with torch.no_grad():
        t = tok(prompts, padding=True, return_tensors="pt").to(DEV)
        tout = model.text_model(**t)
        temb = model.text_projection(tout.pooler_output)
        temb = torch.nn.functional.normalize(temb, dim=-1)   # [2P, D]

    @torch.no_grad()
    def score(pil_list):
        px = proc(images=pil_list, return_tensors="pt")["pixel_values"].to(DEV).to(DT)
        vout = model.vision_model(pixel_values=px)
        iemb = model.visual_projection(vout.pooler_output)
        iemb = torch.nn.functional.normalize(iemb, dim=-1)        # [B, D]
        sim = logit_scale * (iemb @ temb.t())                     # [B, 2P]
        sim = sim.view(sim.shape[0], len(PAIRS), 2)               # [B, P, 2]
        p_pos = torch.softmax(sim.float(), dim=-1)[:, :, 0]       # [B, P]
        return p_pos

    def run_split(path, has_label):
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        cols = ["image_1", "image_2", "img1_w", "img1_h", "img1_fmt", "img1_nbytes"]
        cols += ["is_image1_better"] if has_label else ["index"]
        e1, e2, aux, lab, idx = [], [], [], [], []
        ex = ThreadPoolExecutor(max_workers=6); done, t0 = 0, time.time()
        for batch in pf.iter_batches(batch_size=a.bs, columns=cols):
            d = batch.to_pydict()
            im1 = list(ex.map(decode, d["image_1"])); im2 = list(ex.map(decode, d["image_2"]))
            e1.append(score(im1).cpu().numpy().astype(np.float16))
            e2.append(score(im2).cpu().numpy().astype(np.float16))
            w = np.array(d["img1_w"], np.float32); h = np.array(d["img1_h"], np.float32)
            isj = np.array([1.0 if f == "JPEG" else 0.0 for f in d["img1_fmt"]], np.float32)
            nb = np.log1p(np.array(d["img1_nbytes"], np.float32))
            aux.append(np.stack([w, h, isj, nb], 1))
            if has_label: lab.extend(int(v) for v in d["is_image1_better"])
            else: idx.extend(int(v) for v in d["index"])
            done += len(im1); print(f"  {done}/{n} {done/(time.time()-t0):.0f}/s", flush=True)
        ex.shutdown()
        out = dict(img1=np.concatenate(e1), img2=np.concatenate(e2), aux=np.concatenate(aux))
        if has_label: out["y"] = np.array(lab, np.int8)
        else: out["index"] = np.array(idx, np.int32)
        return out

    print("== TRAIN ==", flush=True); tr = run_split(f"{a.data_dir}/train_512.parquet", True)
    print("== TEST ==", flush=True); te = run_split(f"{a.data_dir}/test_512.parquet", False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out,
        train_img1=tr["img1"], train_img2=tr["img2"], train_aux=tr["aux"], train_y=tr["y"],
        test_img1=te["img1"], test_img2=te["img2"], test_aux=te["aux"], test_index=te["index"])
    print(f"SAVED {a.out} dim={tr['img1'].shape[1]}", flush=True)
    # quick OOF-free sanity: mean P(positive) diff sign-agreement with label
    d = (tr["img1"].astype(np.float32) - tr["img2"].astype(np.float32)).mean(1)
    from sklearn.metrics import roc_auc_score
    print("raw mean-IQA-diff AUC:", round(roc_auc_score(tr["y"], d), 4), flush=True)


if __name__ == "__main__":
    main()
