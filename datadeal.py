import pandas as pd

# 1️⃣ 读取原始 CSV 文件
df = pd.read_csv("grampa.csv")

# 2️⃣ 提取 sequence 列（假设列名正好是 'sequence'）
sequences = df["sequence"].dropna()  # 去除空值
#删除重复序列
sequences = sequences.drop_duplicates()

# 3️⃣ 保存到新文件，每行一个序列
output_path = "grampa.txt"
sequences.to_csv(output_path, index=False, header=False)

print(f"✅ 已将 {len(sequences)} 条序列保存到 {output_path}")
