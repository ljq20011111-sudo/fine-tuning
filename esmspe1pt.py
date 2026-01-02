import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import esm


############################################
# 1. 加载 ESM2（CPU 快速版）
############################################
print("Loading ESM2-t6-8M (fast CPU version)...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()
print("Running on: CPU\n")

batch_converter = alphabet.get_batch_converter()


############################################
# 2. 序列读取 + ESM 特征提取函数
############################################
def load_fasta_and_embed(fasta_path, category):
    ys = []
    Xs = []
    cate = []

    records = list(esm.data.read_fasta(fasta_path))
    print(f"Processing {fasta_path} ...")

    for header, seq in tqdm(records):
        seq_id = header.split("|")[-1]
        ys.append(seq_id)

        # ESM2 embedding
        batch = [(seq_id, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        token_reps = results["representations"][6]

        # 均值池化
        embedding = token_reps[0, 1:-1].mean(0).numpy()
        Xs.append(embedding)
        cate.append(category)

    return np.array(ys), np.array(Xs), np.array(cate)


############################################
# 3. 读取 positive + negative 两个文件
############################################
FASTA_P = r"D:\Apython\xinxinsus-EBAMP-5dde21d\data\spectrum_p.fasta"
FASTA_N = r"D:\Apython\xinxinsus-EBAMP-5dde21d\data\spectrum_n.fasta"

ys_p, Xs_p, cate_p = load_fasta_and_embed(FASTA_P, 1)
ys_n, Xs_n, cate_n = load_fasta_and_embed(FASTA_N, 0)

# 合并
ys = np.concatenate([ys_p, ys_n])
Xs = np.concatenate([Xs_p, Xs_n])
cate = np.concatenate([cate_p, cate_n])

print("\nEmbedding extraction finished.")
print("Xs shape =", Xs.shape)


############################################
# 4. PCA 计算特征重要性 & 排名
############################################
print("\nPerforming PCA for feature ranking...")

pca = PCA(n_components=0.95)
Xs_pca = pca.fit_transform(Xs)

k1_spss = pca.components_.T
weight = (np.dot(k1_spss, pca.explained_variance_ratio_)) / np.sum(pca.explained_variance_ratio_)
weighted_weight = weight / np.sum(weight)

max_location = sorted(enumerate(weighted_weight), key=lambda x: x[1], reverse=True)

idxes = [idx for idx, w in max_location]

# 保存特征下标
with open("esm_spectrum_idxes.txt", "w") as f:
    for idx in idxes:
        f.write(str(idx) + "\n")

print("Saved feature indices → esm_spectrum_idxes.txt")


############################################
# 5. 生成训练集 Excel
############################################
print("\nSaving esm_spectrum_train.xlsx ...")

tmp = Xs[:, idxes]      # 根据 PCA 排序取特征
surface = np.hstack([ys.reshape(-1, 1), tmp, cate.reshape(-1, 1)])

df = pd.DataFrame(surface)
df.to_excel("esm_spectrum_train.xlsx", sheet_name="res", float_format="%.3f", index=False)

print("Done → esm_spectrum_train.xlsx 已生成")
