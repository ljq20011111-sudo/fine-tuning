import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import esm

# ---------------------------------------------------------
# 1. 加载 ESM 模型
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
        r = model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_embeddings = r["representations"][6][0, 1:-1]
    return token_embeddings.mean(0).cpu()


# ---------------------------------------------------------
# 3. 读取 fasta 并生成 embedding
# ---------------------------------------------------------
FASTA_PATH = "D:/Apython/xinxinsus-EBAMP-5dde21d/data/predict_50000.fasta"

print(f"\nProcessing {FASTA_PATH} ...")

ys = []
Xs = []
cate = []

for header, seq in tqdm(list(esm.data.read_fasta(FASTA_PATH))):
    seq_id = header.split("|")[-1]
    ys.append(seq_id)

    emb = embed_sequence(seq)
    Xs.append(emb.numpy())

    cate.append(-1)   # test 集没有标签


Xs = np.array(Xs)
ys = np.array(ys).reshape((-1, 1))
cate = np.array(cate).reshape((-1, 1))

print("Embedding finished. Xs shape =", Xs.shape)


# ---------------------------------------------------------
# 4. 读取 MIC_1 中的特征排序 idx
# ---------------------------------------------------------
print("\nLoading esm_mic_idxes.txt ...")

with open("esm_mic_idxes.txt", "r") as f:
    idxes = [int(line.strip()) for line in f.readlines()]

# 做特征选择 —— 和训练一致
tmp = Xs[:, idxes]


# ---------------------------------------------------------
# 5. 保存最终的 test excel
# ---------------------------------------------------------
print("\nSaving esm_mic_test.xlsx ...")

surface = np.hstack((ys, tmp, cate))
df = pd.DataFrame(surface)

with pd.ExcelWriter("esm_mic_test.xlsx") as writer:
    df.to_excel(writer, "res", float_format="%.3f", index=False)

print("\n🎉 esm_mic_test.xlsx 已生成！")
print("✔ 完成 MIC_2 全流程（无 pt）")
