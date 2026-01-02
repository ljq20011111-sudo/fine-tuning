import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import esm
from sklearn.decomposition import PCA


###########################################
# 加载 ESM2 模型（CPU）
###########################################
print("Loading ESM2-t6-8M (fast CPU version)...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

batch_converter = alphabet.get_batch_converter()
print("Running on: CPU\n")


###########################################
# 读取 fasta 并提取 embedding
###########################################
def embed_fasta(fasta_path):
    ys = []
    Xs = []
    cate = []

    print(f"Processing {fasta_path} ...")
    records = list(esm.data.read_fasta(fasta_path))

    for header, seq in tqdm(records):
        seq_id = header.split("|")[-1]  # 与原版保持一致
        ys.append(seq_id)

        # 组织 batch
        batch = [(seq_id, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        # 提取 ESM 表征
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])  # ESM2 小模型用 layer=6

        token_reps = results["representations"][6]
        embedding = token_reps[0, 1:-1].mean(0).numpy()  # 平均池化

        Xs.append(embedding)
        cate.append(-1)  # 测试集 cate 固定为 -1（与 EBAMP 原版一致）

    return np.array(ys), np.array(Xs), np.array(cate)


###########################################
# 路径：predict_50000.fasta
###########################################
FASTA_PATH = r"D:\Apython\xinxinsus-EBAMP-5dde21d\data\predict_50000.fasta"

ys, Xs, cate = embed_fasta(FASTA_PATH)

print("\nEmbedding extraction finished.")
print("Xs shape =", Xs.shape)


###########################################
# 加载 toxicity 的特征排序 idxes
###########################################
print("\nLoading esm_toxicity_idxes.txt ...")
idxes = []

with open("esm_toxicity_idxes.txt", "r") as f:
    for line in f:
        idxes.append(int(line.strip()))

print(f"Loaded {len(idxes)} indices.")


###########################################
# 根据 idxes 选择 PCA 排名后的特征
###########################################
tmp = Xs[:, idxes]


###########################################
# 组合成与原版一致的数据格式
###########################################
surface = np.hstack((ys.reshape(-1, 1), tmp, cate.reshape(-1, 1)))
data = pd.DataFrame(surface)


###########################################
# 存为 Excel
###########################################
print("Saving esm_toxicity_test.xlsx ...")
data.to_excel("esm_toxicity_test.xlsx", sheet_name="res",
              float_format="%.3f", index=False)

print("Done → esm_toxicity_test.xlsx 已生成")
