"""
Download pretrained vision weights ON THE MAC (fast internet) into a clean,
flat local dir, so we can scp them to the remote GPU box (where the HF CDN
stalls to ~0 B/s). Load on the remote with from_pretrained(local_path).

Usage: python scripts/fetch_weights.py dinov2-large [clip-large] [siglip]
"""
import sys, os, shutil
from huggingface_hub import snapshot_download

# Xet high-performance transfer is the fast path (~100+ MB/s); the legacy HTTP
# path is rate-limited to ~85 KB/s for unauthenticated users.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

DST = "artifacts/hf_models"
os.makedirs(DST, exist_ok=True)

REPOS = {
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-base":  "facebook/dinov2-base",
    "clip-large":   "openai/clip-vit-large-patch14",
    "siglip":       "google/siglip-so400m-patch14-384",
    "siglip2":      "google/siglip2-so400m-patch14-384",
}

want = sys.argv[1:] or ["dinov2-large"]
for key in want:
    repo = REPOS[key]
    local = os.path.join(DST, key)
    print(f"=== downloading {repo} (Xet -> default cache, then copy out)", flush=True)
    # Download into the DEFAULT hf cache (Xet's native fast path; local_dir
    # reconstruction stalls for large files), then copy resolved files to a flat dir.
    snap = snapshot_download(
        repo_id=repo,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model",
                        "tokenizer*", "vocab*", "merges*"],
    )
    os.makedirs(local, exist_ok=True)
    for f in os.listdir(snap):
        src = os.path.join(snap, f)
        if os.path.isfile(src):  # follows symlink to the blob, copies real content
            shutil.copy(src, os.path.join(local, f))
    sz = sum(os.path.getsize(os.path.join(local, f)) for f in os.listdir(local)) / 1e6
    print(f"  done {key}: {sz:.0f} MB -> {local}", flush=True)
print("FETCH DONE", flush=True)
