import pandas as pd
import os

base = r"D:\Apython\xinxinsus-EBAMP-5dde21d\code\esm\esmpt"

# ==== 真实文件路径 ====
phys_file = os.path.join(base, "physicochemical_scores.xlsx")
amp_file  = os.path.join(base, "amp_scores.xlsx")
spec_file = os.path.join(base, "spectrum_scores.xlsx")
tox_file  = os.path.join(base, "toxicity_scores.xlsx")
mic_file  = os.path.join(base, "mic_scores.xlsx")

print("Loading files...\n")

df_phys = pd.read_excel(phys_file)
df_amp  = pd.read_excel(amp_file)
df_spec = pd.read_excel(spec_file)
df_tox  = pd.read_excel(tox_file)
df_mic  = pd.read_excel(mic_file)

# ===============================
# 统一列名
# ===============================

df_phys.columns = ["seq", "pI", "charge", "hm"]
df_amp.columns  = ["seq", "score_amp", "pred_amp"]
df_spec.columns = ["seq", "pred_spec", "score_spec"]
df_tox.columns  = ["seq", "pred_tox", "score_tox"]
df_mic.columns  = ["seq", "score_mic", "pred_mic"]

print("Column normalization done.\n")

# ===============================
# 按序合并
# ===============================
df = df_phys.merge(df_amp,  on="seq") \
            .merge(df_spec, on="seq") \
            .merge(df_tox,  on="seq") \
            .merge(df_mic,  on="seq")

print(f"Total merged sequences: {len(df)}")

# ===============================
# Step 1: 理化性质
# ===============================
df1 = df[(df["pI"] >= 0) & (df["charge"] > 0) & (df["hm"] >= 0)]
print(f"After physicochemical filter: {len(df1)}")

# ===============================
# Step 2: AMP > 0.5
# ===============================
df2 = df1[df1["score_amp"] > 0.5]
print(f"After AMP filter: {len(df2)}")

# ===============================
# Step 3: Broad-spectrum > 0.5
# ===============================
df3 = df2[df2["score_spec"] > 0.5]
print(f"After spectrum filter: {len(df3)}")

# ===============================
# Step 4: Non-toxicity > 0.5
# ===============================
df4 = df3[df3["score_tox"] > 0.5]
print(f"After toxicity filter: {len(df4)}")

# ===============================
# Step 5: MIC > 0.5
# ===============================
df5 = df4[df4["score_mic"] > 0.5]
print(f"After MIC filter: {len(df5)}")

# ===============================
# 保存最终结果
# ===============================
out_file = os.path.join(base, "ebamp_final_sequences.xlsx")
df5.to_excel(out_file, index=False)

print("\n🎉 EBAMP filtering completed!")
print(f"👉 Final sequences saved to: {out_file}")
print(f"👉 Total remaining: {len(df5)}")
