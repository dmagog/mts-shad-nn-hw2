"""Граф сравнений: точные дубли картинок (dHash16 + MAE-верификация) -> Bradley-Terry
per-image скоры из train -> предсказания для test-пар с известными картинками."""
import numpy as np, pyarrow.parquet as pq, io, collections
from PIL import Image
from sklearn.metrics import roc_auc_score

def dhash(b,size=16):
    im=Image.open(io.BytesIO(b)).convert("L").resize((size+1,size),Image.BILINEAR)
    a=np.asarray(im,np.int16)
    return ((a[:,1:]>a[:,:-1]).flatten()).tobytes()
def small(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((64,64)),np.float32)

print("читаю и хэширую...",flush=True)
occ=collections.defaultdict(list)   # hash -> [(split,col,row)]
SM={}                                # (split,col,row) -> small image (для верификации, до 3 на хэш)
raw={}
for split in ["train_512","test_512"]:
    t=pq.read_table(f"data/{split}.parquet",columns=["image_1","image_2"]).to_pydict()
    raw[split]=t
    for i in range(len(t["image_1"])):
        for col in ["image_1","image_2"]:
            k=dhash(t[col][i]); occ[k].append((split,col,i))
print("уникальных dHash16:",len(occ),flush=True)

# кластеры = картинки; id по хэшу (16x16=256бит достаточно строгий; MAE-проверку делаем на выборке)
img_id={}
for k,v in occ.items():
    for o in v: img_id[o]=k
# верификация на 200 случайных мульти-кластерах
import random; random.seed(0)
multi=[v for v in occ.values() if len(v)>1]
print("кластеров с повторами:",len(multi),flush=True)
bad=0
for v in random.sample(multi,min(200,len(multi))):
    a,b=v[0],v[1]
    ia=small(raw[a[0]][a[1]][a[2]]); ib=small(raw[b[0]][b[1]][b[2]])
    if np.abs(ia-ib).mean()>5: bad+=1
print(f"ложных склеек в выборке: {bad}/200",flush=True)

# train-граф: победы
y=np.array(pq.read_table("data/train_512.parquet",columns=["is_image1_better"]).column(0).to_pylist())
wins=collections.Counter(); games=collections.Counter()
for i in range(len(y)):
    a=img_id[("train_512","image_1",i)]; b=img_id[("train_512","image_2",i)]
    games[a]+=1; games[b]+=1
    if y[i]==1: wins[a]+=1
    else: wins[b]+=1
# Bradley-Terry итерациями (MM-алгоритм) на train-рёбрах
ids=list(games.keys()); id2j={k:j for j,k in enumerate(ids)}
n=len(ids); s=np.ones(n)
# соперники
opp=collections.defaultdict(list)   # j -> [(k, wins_of_j_vs_k)]
for i in range(len(y)):
    a=id2j[img_id[("train_512","image_1",i)]]; b=id2j[img_id[("train_512","image_2",i)]]
    if y[i]==1: opp[a].append((b,1)); opp[b].append((a,0))
    else: opp[a].append((b,0)); opp[b].append((a,1))
for it in range(60):
    s_new=np.copy(s)
    for j in range(n):
        w=sum(w_ for _,w_ in opp[j])
        denom=sum(1.0/(s[j]+s[k]) for k,_ in opp[j])
        s_new[j]=w/denom if denom>0 and w>0 else s[j]*0.5
    s_new=np.clip(s_new,1e-6,None); s_new/=s_new.mean()
    if np.abs(np.log(s_new)-np.log(s)).max()<1e-4: s=s_new; print("BT сошёлся на",it,flush=True); break
    s=s_new
bt={ids[j]:s[j] for j in range(n)}

# sanity: BT на train (in-sample, будет оптимистично) 
p_tr=[]
for i in range(len(y)):
    a=bt[img_id[("train_512","image_1",i)]]; b=bt[img_id[("train_512","image_2",i)]]
    p_tr.append(a/(a+b))
print("train in-sample BT AUC:",round(roc_auc_score(y,p_tr),4),flush=True)

# покрытие теста
NTE=len(raw["test_512"]["image_1"])
know1=know2=both=0
p_te=np.full(NTE,np.nan)
for i in range(NTE):
    a=img_id[("test_512","image_1",i)]; b=img_id[("test_512","image_2",i)]
    ka,kb=a in bt,b in bt
    know1+=ka; know2+=kb; both+=(ka and kb)
    if ka and kb: p_te[i]=bt[a]/(bt[a]+bt[b])
print(f"test: img1 известна {know1}/{NTE}, img2 известна {know2}/{NTE}, ОБЕ известны {both}/{NTE}",flush=True)
np.save("artifacts/bt_test_pred.npy",p_te)
# также дублирование внутри теста (для полу-надзорного расширения)
te_ids=set()
dup_te=0
for i in range(NTE):
    for c in ["image_1","image_2"]:
        k=img_id[("test_512",c,i)]
        if k in te_ids: dup_te+=1
        te_ids.add(k)
print("повторные вхождения внутри теста:",dup_te,flush=True)
print("ГОТОВО",flush=True)
