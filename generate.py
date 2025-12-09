import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# ====== 配置 ======
model_path = "/root/autodl-tmp/fine-tuning/protGPT2_finetuned/checkpoint-24393"
num_sequences = 50
max_new_tokens = 50
temperature = 1.0
top_k = 50
top_p = 0.95
output_csv = "generated_peptides.csv"

# ====== 加载模型 ======
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    dtype=torch.float16   # 替代 torch_dtype
)
model.eval()

# pad_token 设置（ProtGPT2 常见做法）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# ====== 随机氨基酸列表 ======
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ====== 生成函数（含 attention_mask） ======
def generate_peptide():
    # 随机挑选一个氨基酸作为起始 token
    seed_aa = random.choice(AMINO_ACIDS)

    # tokenizer 编码
    inputs = tokenizer(seed_aa, return_tensors="pt", padding=False)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # 生成
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,  # 生成足够的 token
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    seq = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 移除 seed_aa
    if seq.startswith(seed_aa):
        seq = seq[1:]

    # 移除换行符
    seq = seq.replace("\n", "").replace("\r", "").strip()

    #  随机长度限制：1~50 个字符
    max_len = random.randint(1, 50)
    seq = seq[:max_len]

    return seq



# ====== 批量生成 ======
sequences = []
for i in range(num_sequences):
    seq = generate_peptide()
    sequences.append(seq)
    print(f"{i+1}/{num_sequences}: {seq}")

# 去重
sequences = list(set(sequences))

# 保存到 CSV
df = pd.DataFrame({"peptide_sequence": sequences})
df.to_csv(output_csv, index=False)
print(f"✅ 已生成 {len(sequences)} 条抗菌肽序列，保存到 {output_csv}")
