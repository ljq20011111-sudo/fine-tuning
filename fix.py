import os
import pandas as pd

# 目标文件名
target_files = [
    "physicochemical_scores.xlsx",
    "amp_scores.xlsx",
    "spectrum_scores.xlsx",
    "toxicity_scores.xlsx",
    "mic_scores.xlsx",
]

# ★ 修改为你的 EBAMP 根目录 ★
ROOT = r"D:\Apython\xinxinsus-EBAMP-5dde21d"

found_files = {}

print("Searching for score files...\n")

# 递归搜索
for root, dirs, files in os.walk(ROOT):
    for f in files:
        if f in target_files:
            found_files[f] = os.path.join(root, f)

# 打印找到的文件
for f in target_files:
    if f in found_files:
        print(f"FOUND: {found_files[f]}")
    else:
        print(f"MISSING: {f}")

print("\nStarting fixing...\n")

# 修复 seq 列
def fix_seq_column(file_path):
    df = pd.read_excel(file_path)

    # 找 seq 列
    seq_col = None
    for col in df.columns:
        if str(col).lower().startswith("seq"):
            seq_col = col
            break

    if seq_col is None:
        print(f"❌ No seq column in {file_path}")
        return

    print(f"Fixing seq column in: {file_path}")

    # 转字符串
    df[seq_col] = df[seq_col].astype(str)

    df[seq_col] = (
        df[seq_col]
        .str.replace("seq_", "", regex=False)
        .str.replace(".0", "", regex=False)
        .astype(str)
    )

    df.to_excel(file_path, index=False)
    print("Saved:", file_path)


for f in found_files.values():
    fix_seq_column(f)

print("\nAll done!")
