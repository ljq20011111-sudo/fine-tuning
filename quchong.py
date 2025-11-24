# 文件路径
input_file = "tataldata.txt"
output_file = "tataldata_unique.txt"

# 读取文件
with open(input_file, 'r', encoding='utf-8') as f:
    sequences = [line.strip() for line in f if line.strip()]

# 去重
unique_sequences = list(set(sequences))
print(f"去重后共有 {len(unique_sequences)} 条序列。")

# 保存去重后的文件
with open(output_file, 'w', encoding='utf-8') as f:
    for seq in unique_sequences:
        f.write(seq + "\n")

# 求最大长度
max_length = max(len(seq) for seq in unique_sequences)
print(f"去重后序列的最大长度为: {max_length}")
