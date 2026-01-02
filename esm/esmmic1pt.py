import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import esm
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 载入 ESM2（轻量CPU友好版）
# ---------------------------------------------------------
print("Loading ESM2-t6-8M (fast CPU version)...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Running on:", device)

# ---------------------------------------------------------
# 2. embedding 函数
# ---------------------------------------------------------
def embed_sequence(seq):
    batch_labels, batch_strs, batch_tokens = batch_converter([("seq", seq)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_embeddings = results["representations"][6][0, 1:-1]
    return token_embeddings.mean(0).cpu()

# ---------------------------------------------------------
# 3. 读取 fasta + 抽取 embedding
# ---------------------------------------------------------
def load_fasta_and_embed(fasta_path, label):
    ys = []
    Xs = []
    cate = []

    print(f"\nProcessing {fasta_path} ...")
    for header, seq in tqdm(list(esm.data.read_fasta(fasta_path))):
        seq_id = header.split("|")[-1]
        ys.append(seq_id)

        emb = embed_sequence(seq)
        Xs.append(emb.numpy())
        cate.append(label)

    return np.array(ys), np.array(Xs), np.array(cate)

# ---------------------------------------------------------
# 4. 修正后的 fasta 路径（你的路径）
# ---------------------------------------------------------
FASTA_P = "D:/Apython/xinxinsus-EBAMP-5dde21d/data/mic_p.fasta"
FASTA_N = "D:/Apython/xinxinsus-EBAMP-5dde21d/data/mic_n.fasta"

ys_p, Xs_p, cate_p = load_fasta_and_embed(FASTA_P, 1)
ys_n, Xs_n, cate_n = load_fasta_and_embed(FASTA_N, 0)

# 拼接
ys = np.concatenate([ys_p, ys_n]).reshape(-1, 1)
Xs = np.concatenate([Xs_p, Xs_n])
cate = np.concatenate([cate_p, cate_n]).reshape(-1, 1)

print("\nEmbedding extraction finished.")
print("Xs shape =", Xs.shape)

# ---------------------------------------------------------
# 5. PCA 排序
# ---------------------------------------------------------
print("\nPerforming PCA for feature ranking...")

pca = PCA(n_components=0.95)
Xs_pca = pca.fit_transform(Xs)

k1_spss = pca.components_.T
weight = (np.dot(k1_spss, pca.explained_variance_ratio_)) / np.sum(pca.explained_variance_ratio_)
weighted_weight = weight / np.sum(weight)
max_location = sorted(enumerate(weighted_weight), key=lambda y: y[1], reverse=True)

# 保存 idx
with open("./esm_mic_idxes.txt", "w") as f:
    idxes = []
    for i in range(len(Xs_pca[0])):
        idx = max_location[i]
        idxes.append(idx[0])
        f.write(str(idx[0]) + "\n")

print("Saved feature indices → esm_mic_idxes.txt")

# ---------------------------------------------------------
# 6. 写出训练集（新版 pandas 要用 with 语法）
# ---------------------------------------------------------
tmp = Xs[:, idxes]
surface = np.hstack((ys, tmp, cate))

df = pd.DataFrame(surface)

print("\nSaving esm_mic_train.xlsx ...")

with pd.ExcelWriter("./esm_mic_train.xlsx") as writer:
    df.to_excel(writer, "res", float_format="%.3f", index=False)

print("\n🎉 esm_mic_train.xlsx 已生成！")
print("✔ 完成 MIC_1 全流程（无需 pt）")
