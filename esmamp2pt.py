import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import esm

# ===============================
# 路径（请根据你的本地设置）
# ===============================
FASTA_PATH = r"D:/Apython/xinxinsus-EBAMP-5dde21d/data/predict_50000.fasta"
SAVE_DIR = r"D:/Apython/xinxinsus-EBAMP-5dde21d/data/predict/"
IDX_FILE = r"./esm_amp_idxes.txt"

os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# 加载 ESM2-t6-8M（快速 CPU 模型）
# ===============================
print("Loading ESM2-t6-8M (fast CPU version)...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
EMB_LAYER = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)

model = model.to(DEVICE)
model.eval()

# ===============================
# ESM embedding 函数（与 amp_1.py 相同）
# ===============================
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

# ===============================
# 读取 esm_amp_idxes.txt（重要性排序）
# ===============================
idxes = []
with open(IDX_FILE, "r") as f:
    for line in f:
        idxes.append(int(line.strip()))

print(f"Loaded {len(idxes)} selected feature indices.")

# ===============================
# 处理 predict_50000.fasta
# ===============================
ys = []
Xs = []
cate = []

print(f"\nProcessing {FASTA_PATH} ...")

for header, seq in esm.data.read_fasta(FASTA_PATH):
    y = float(header.split("|")[-1])
    ys.append(y)

    save_path = f"{SAVE_DIR}/{y}.pt"
    emb = get_esm_embedding(seq, save_path)
    Xs.append(emb["mean_representations"][EMB_LAYER].numpy())

    cate.append(-1)  # inference 任务，不需要分类标签

Xs = np.vstack(Xs)
ys = np.array(ys).reshape(-1, 1)
cate = np.array(cate).reshape(-1, 1)

# ===============================
# 按 importance index 选取特征
# ===============================
Xs_sel = Xs[:, idxes]

# 构建输出矩阵
output = np.hstack((ys, Xs_sel, cate))

df = pd.DataFrame(output)
df.to_excel("./esm_amp_test.xlsx", index=False)

print("\nDONE! Saved esm_amp_test.xlsx")
