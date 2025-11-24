import torch
from transformers import AutoModelForCausalLM

model_path = "/root/autodl-tmp/fine-tuning/protGPT2_finetuned"

model = AutoModelForCausalLM.from_pretrained(model_path)

print(model)
