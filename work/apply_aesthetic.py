"""
Применить LAION improved-aesthetic-predictor (MLP на CLIP ViT-L/14 эмбеддингах)
к уже посчитанным CLIP-эмбеддингам -> эстетический скор каждой картинки.
Сохраняет как emb-файл (img1/img2 = [N,1] скоры) для компаратора + печатает raw AUC.
"""
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

class Aesthetic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768,1024), nn.Dropout(0.2),
            nn.Linear(1024,128), nn.Dropout(0.2),
            nn.Linear(128,64), nn.Dropout(0.1),
            nn.Linear(64,16),
            nn.Linear(16,1))
    def forward(self,x): return self.layers(x)

m = Aesthetic(); m.load_state_dict(torch.load("artifacts/aesthetic/sa_l14.pth", map_location="cpu", weights_only=True)); m.eval()

z = np.load("artifacts/emb_clipL.npz")
def score(a):
    x = torch.tensor(a.astype(np.float32))
    x = x / x.norm(dim=-1, keepdim=True)          # L2-norm (предиктор ждёт нормализованные)
    with torch.no_grad(): return m(x).numpy().astype(np.float16)  # [N,1]

out = {}
for split in ["train","test"]:
    out[f"{split}_img1"] = score(z[f"{split}_img1"])
    out[f"{split}_img2"] = score(z[f"{split}_img2"])
    out[f"{split}_aux"] = z[f"{split}_aux"]
out["train_y"] = z["train_y"]; out["test_index"] = z["test_index"]
np.savez_compressed("artifacts/emb_aesthetic.npz", **out)

y = z["train_y"]; d = (out["train_img1"].astype(np.float32) - out["train_img2"].astype(np.float32)).ravel()
print("эстетика img1: mean=%.2f std=%.2f" % (out["train_img1"].mean(), out["train_img1"].std()))
print("raw AUC (aesthetic_img1 - aesthetic_img2):", round(roc_auc_score(y, d), 4))
print("raw AUC (обратный знак):", round(roc_auc_score(y, -d), 4))
print("saved artifacts/emb_aesthetic.npz")
