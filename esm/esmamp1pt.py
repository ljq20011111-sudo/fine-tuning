import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import esm
from sklearn.decomposition import PCA

# =======================
# 1. 用绝对路径（你的本地路径）
# =======================
FASTA_P = r"D:/Apython/xinxinsus-EBAMP-5dde21d/data/amp_p.fasta"
FASTA_N = r"D:/Apython/xinxinsus-EBAMP-5dde21d/data/amp_n.fasta"

SAVE_DIR_P = r"D:/Apython/xinxinsus-EBAMP-5dde21d/data/amp_p/"
SAVE_DIR_N = r"D:/Apython/xinxinsus-EBAMP-5dde21d/data/amp_n/"

os.makedirs(SAVE_DIR_P, exist_ok=True)
os.makedirs(SAVE_DIR_N, exist_ok=True)

# =======================
# 2. 使用极速 ESM2-t6-8M（适合 CPU）
# =======================
print("Loading ESM2-t6-8M (fast CPU version)...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
EMB_LAYER = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

model = model.to(DEVICE)
model.eval()

# =======================
# 3. 序列 → ESM embedding
# =======================
def get_esm_embedding(seq, save_path):
    if os.path.exists(save_path):
        return torch.load(save_path)

    data = [("id", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(DEVICE)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[EMB_LAYER])

    token_reps = results["representations"][EMB_LAYER][0, 1:len(seq)+1]
    mean_rep = token_reps.mean(0).cpu()

    torch.save({"mean_representations": {EMB_LAYER: mean_rep}}, save_path)
    return {"mean_representations": {EMB_LAYER: mean_rep}}

# =======================
# 4. 加载 fasta + 自动 embedding
# =======================
def load_fasta_and_embed(fasta_path, save_dir, label):
    ys, Xs, cate = [], [], []

    print(f"\nProcessing {fasta_path} ...")
    for header, seq in esm.data.read_fasta(fasta_path):
        value = header.split("|")[-1]
        save_path = f"{save_dir}/{value}.pt"

        emb = get_esm_embedding(seq, save_path)

        ys.append(float(value))
        Xs.append(emb["mean_representations"][EMB_LAYER].numpy())
        cate.append(label)

    return ys, Xs, cate


ys_p, Xs_p, cate_p = load_fasta_and_embed(FASTA_P, SAVE_DIR_P, 1)
ys_n, Xs_n, cate_n = load_fasta_and_embed(FASTA_N, SAVE_DIR_N, 0)

# 合并
ys = np.array(ys_p + ys_n).reshape(-1, 1)
Xs = np.vstack(Xs_p + Xs_n)
cate = np.array(cate_p + cate_n).reshape(-1, 1)

print("\nTotal samples:", len(ys))

# =======================
# 5. PCA 降维
# =======================
pca = PCA(n_components=0.95)
Xs_pca = pca.fit_transform(Xs)

# =======================
# 6. 计算特征重要性
# =======================
k1 = pca.components_.T
weight = (np.dot(k1, pca.explained_variance_ratio_)) / np.sum(pca.explained_variance_ratio_)
weight = weight / np.sum(weight)

idx_sorted = sorted(enumerate(weight), key=lambda x: x[1], reverse=True)
feature_idx = [i[0] for i in idx_sorted]

with open("./esm_amp_idxes.txt", "w") as f:
    for i in feature_idx:
        f.write(str(i) + "\n")

# =======================
# 7. 输出训练数据
# =======================
Xs_sel = Xs[:, feature_idx]
surface = np.hstack((ys, Xs_sel, cate))

df = pd.DataFrame(surface)
df.to_excel("./esm_amp_train.xlsx", index=False)

print("\nDONE! Saved esm_amp_train.xlsx and esm_amp_idxes.txt")
