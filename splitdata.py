import random

# === 1️⃣ 读取数据 ===
with open('tataldata_unique.txt', 'r', encoding='utf-8') as file:
    data = [line for line in file if line.strip()]  # 去除空行

# === 2️⃣ 设置随机种子，保证划分一致 ===
random.seed(42)

# === 3️⃣ 打乱数据 ===
random.shuffle(data)

# === 4️⃣ 计算划分索引 ===
total_data = len(data)
train_size = int(0.8 * total_data)
val_size = int(0.1 * total_data)

# === 5️⃣ 划分数据 ===
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# === 6️⃣ 保存文件 ===
with open('train.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_data)
with open('val.txt', 'w', encoding='utf-8') as f:
    f.writelines(val_data)
with open('test.txt', 'w', encoding='utf-8') as f:
    f.writelines(test_data)

# === 7️⃣ 输出统计信息 ===
print("✅ 数据划分完成！")
print(f"总数据量: {total_data}")
print(f"训练集: {len(train_data)} ({len(train_data)/total_data*100:.2f}%)")
print(f"验证集: {len(val_data)} ({len(val_data)/total_data*100:.2f}%)")
print(f"测试集: {len(test_data)} ({len(test_data)/total_data*100:.2f}%)")
