import matplotlib.pyplot as plt

# === 1️⃣ 读取序列文件 ===
file_path = "tataldata_unique.txt"  # 修改为你的txt文件路径
with open(file_path, "r", encoding="utf-8") as f:
    sequences = [line.strip() for line in f if line.strip()]  # 去掉空行

# === 2️⃣ 计算序列长度 ===
lengths = [len(seq) for seq in sequences]

# === 3️⃣ 基本统计信息 ===
total = len(lengths)
long_seqs = sum(l > 100 for l in lengths)  # 超过100的数量
ratio = long_seqs / total * 100

print(f"总序列数: {total}")
print(f"最短序列长度: {min(lengths)}")
print(f"最长序列长度: {max(lengths)}")
print(f"平均序列长度: {sum(lengths)/total:.2f}")
print(f"长度超过100的序列数量: {long_seqs} ({ratio:.2f}%)")

# === 4️⃣ 绘制直方图 ===
plt.figure(figsize=(8, 5))
plt.hist(lengths, bins=30, edgecolor='black', color='skyblue')
plt.title("序列长度分布直方图", fontsize=14)
plt.xlabel("序列长度", fontsize=12)
plt.ylabel("序列数量", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
