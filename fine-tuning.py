import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback, TrainerCallback
)
from datasets import Dataset


class EvalEveryNEpochCallback(TrainerCallback):
    def __init__(self, n=3):
        self.n = n

    def on_epoch_end(self, args, state, control, **kwargs):
        # 每 n 个 epoch 进行评估
        if (state.epoch % self.n) == 0:
            control.should_evaluate = True
        else:
            control.should_evaluate = False
        return control


'''
class ImprovedEarlyStopping(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 只获取 eval_loss 作为早停指标
        metric_value = metrics.get("eval_loss")

        if metric_value is None:
            print("⚠️ 评估结果中没有找到 eval_loss，早停检查跳过")
            return

        # 调用父类方法检查指标是否改善
        self.check_metric_value(args, state, control, metric_value)

        # 打印早停计数器状态
        if self.early_stopping_patience_counter > 0:
            print(f"⚠️ 早停计数器: {self.early_stopping_patience_counter}/{self.early_stopping_patience}")

        # 达到耐心次数则停止训练
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
'''

# 训练结果分析函数
def analyze_training_results(trainer, output_dir):
    """分析训练结果，训练曲线保存 eval_loss 和 learning_rate，最终只打印 eval_loss"""
    history = trainer.state.log_history
    df_history = pd.DataFrame(history)

    columns_to_save = [col for col in ["step", "epoch", "eval_loss", "learning_rate"] if col in df_history.columns]
    df_history[columns_to_save].to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    final_eval_loss = trainer.evaluate().get("eval_loss")
    print("\n📊 最终评估结果:")
    print(f"  eval_loss: {final_eval_loss:.4f}")


if __name__ == "__main__":
    # === 1️⃣ 环境变量设置 ===
    os.environ["HF_HOME"] = "/root/autodl-tmp/fine-tuning/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/fine-tuning/huggingface"
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

    output_dir = "/root/autodl-tmp/fine-tuning/protGPT2_finetuned"
    log_dir = "/root/autodl-tmp/fine-tuning/protGPT2_logs"


    # === 2️⃣ 加载训练和验证集 ===
    def load_dataset(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return pd.DataFrame({"text": lines})

    print("📂 正在加载训练和验证数据...")
    df_train = load_dataset("train.txt")
    df_val = load_dataset("val.txt")

    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)


    # === 3️⃣ 加载模型与分词器 ===
    print("🌐 正在加载 ProtGPT2 模型与分词器...")
    model_name = "nferruz/ProtGPT2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/root/autodl-tmp/fine-tuning/huggingface"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="/root/autodl-tmp/fine-tuning/huggingface"
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 256
    model.config.pad_token_id = tokenizer.eos_token_id


    # === 4️⃣ Tokenize 数据 ===
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding=False
        )

    print("🔡 正在进行分词处理...")
    tokenized_train = dataset_train.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = dataset_val.map(tokenize_function, batched=True, remove_columns=["text"])

    # === 5️⃣ Data Collator ===
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # === 6️⃣ 显存优化 ===
    model.gradient_checkpointing_enable()
    model.config.use_cache = False


    # === 7️⃣ 训练参数 ===
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,  # 或 2e-5
        lr_scheduler_type="cosine",  # 余弦衰减更平滑
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        logging_dir=log_dir,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[
            EvalEveryNEpochCallback(n=3)
        ]
    )
    # === 🔟 开始训练 ===
    print("🚀 开始微调 ProtGPT2 模型...\n")

    # 指定要恢复的检查点
    specific_checkpoint = "/root/autodl-tmp/fine-tuning/protGPT2_finetuned/checkpoint-16089"
    if os.path.exists(specific_checkpoint):
        print(f"🔁 从指定检查点 {specific_checkpoint} 恢复训练...\n")
        trainer.train(resume_from_checkpoint=specific_checkpoint)
    else:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = os.path.join(output_dir, sorted(checkpoints)[-1])
            print(f"🔁 检测到断点 {latest_checkpoint} ，从上次保存点恢复训练...\n")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            trainer.train()

    # === 1️⃣1️⃣ 保存最终模型 ===
    print("\n💾 保存模型与分词器中...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # === 1️⃣2️⃣ 分析训练结果 ===
    print("\n📈 正在分析训练结果...")
    training_history = analyze_training_results(trainer, output_dir)

    print(f"\n✅ 微调完成，模型已保存至: {output_dir}")
    print(f"📊 TensorBoard 日志目录: {log_dir}")
    print(f"📄 训练历史已保存至: {os.path.join(output_dir, 'training_history.csv')}")
    print(f"💡 查看训练曲线：tensorboard --logdir=\"{log_dir}\" --port=6006")
