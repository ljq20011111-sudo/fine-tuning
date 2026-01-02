import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
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
# 2. 读取 fasta + embedding
############################################
def embed_fasta(fasta_path):
    ys = []
    Xs = []
    cate = []

    records = list(esm.data.read_fasta(fasta_path))
    print(f"Processing {fasta_path} ...")

    for header, seq in tqdm(records):
        seq_id = header.split("|")[-1]
        ys.append(seq_id)

        batch = [(seq_id, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])

        token_reps = results["representations"][6]
        embedding = token_reps[0, 1:-1].mean(0).numpy()

        Xs.append(embedding)
        cate.append(-1)       # 测试集标签固定为 -1

    return np.array(ys), np.array(Xs), np.array(cate)


############################################
# 3. 路径（你提供的）
############################################
FASTA_PATH = r"D:\Apython\xinxinsus-EBAMP-5dde21d\data\predict_50000.fasta"

ys, Xs, cate = embed_fasta(FASTA_PATH)

print("\nEmbedding extraction finished.")
print("Xs shape =", Xs.shape)


############################################
# 4. 加载 spectrum_1 生成的特征排序
############################################
idxes = []
with open("esm_spectrum_idxes.txt", "r") as f:
    for line in f:
        idxes.append(int(line.strip()))

idxes = np.array(idxes)


############################################
# 5. 特征筛选 + 生成测试集 Excel
############################################
print("\nSaving esm_spectrum_test.xlsx ...")

tmp = Xs[:, idxes]    # 选择排序后的特征

surface = np.hstack([ys.reshape(-1, 1), tmp, cate.reshape(-1, 1)])

df = pd.DataFrame(surface)
df.to_excel("esm_spectrum_test.xlsx", sheet_name="res", float_format="%.3f", index=False)

print("\nDone → esm_spectrum_test.xlsx 已生成")
