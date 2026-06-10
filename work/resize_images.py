"""
Resize competition images to a compact, transfer-friendly parquet.

- Downscale so the longest side = 512 (LANCZOS), preserve aspect, never upscale.
- Re-encode as JPEG q92.
- Preserve image_1 original metadata (w, h, format, nbytes) as aux features
  (image_2 is always WEBP 1000x1000 -> no info, skipped).

Rationale: all planned backbones consume <=518px, so 512 source is near-lossless
for them while cutting ~6GB -> ~2GB for the slow scp to the remote GPU box.
Pure CPU work (threads) -> does not tax the Mac GPU.
"""
import io, sys, time
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SRC_DIR = "data"
DST_DIR = "data"
MAXSIDE = 512
QUALITY = 92
BATCH = 256
WORKERS = 6


def resize_one(b: bytes):
    im = Image.open(io.BytesIO(b))
    fmt = im.format
    im = im.convert("RGB")
    w, h = im.size
    s = MAXSIDE / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=QUALITY)
    return out.getvalue(), w, h, fmt, len(b)


def process(split: str, has_label: bool):
    src = f"{SRC_DIR}/{split}.parquet"
    dst = f"{DST_DIR}/{split}_512.parquet"
    pf = pq.ParquetFile(src)
    n_total = pf.metadata.num_rows
    cols = ["image_1", "image_2"] + (["is_image1_better"] if has_label else [])

    fields = [
        ("image_1", pa.binary()), ("image_2", pa.binary()),
        ("img1_w", pa.int32()), ("img1_h", pa.int32()),
        ("img1_fmt", pa.string()), ("img1_nbytes", pa.int32()),
    ]
    if has_label:
        fields.insert(2, ("is_image1_better", pa.int8()))
    else:
        fields.insert(2, ("index", pa.int32()))
    schema = pa.schema(fields)

    writer = pq.ParquetWriter(dst, schema, compression="zstd")
    done = 0
    idx = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
            d = batch.to_pydict()
            L = len(d["image_1"])
            r1 = list(ex.map(resize_one, d["image_1"]))
            r2 = list(ex.map(resize_one, d["image_2"]))
            arrs = {
                "image_1": [x[0] for x in r1],
                "image_2": [x[0] for x in r2],
                "img1_w":  [x[1] for x in r1],
                "img1_h":  [x[2] for x in r1],
                "img1_fmt":[x[3] for x in r1],
                "img1_nbytes":[x[4] for x in r1],
            }
            if has_label:
                arrs["is_image1_better"] = [int(v) for v in d["is_image1_better"]]
            else:
                arrs["index"] = list(range(idx, idx + L))
            idx += L
            writer.write_table(pa.table({k: arrs[k] for k, _ in fields}, schema=schema))
            done += L
            print(f"[{split}] {done}/{n_total}  {done/(time.time()-t0):.0f} img-rows/s", flush=True)
    writer.close()
    import os
    sz = os.path.getsize(dst) / 1e6
    print(f"[{split}] DONE -> {dst}  ({sz:.0f} MB)", flush=True)


if __name__ == "__main__":
    process("train", True)
    process("test", False)
    print("ALL DONE", flush=True)
