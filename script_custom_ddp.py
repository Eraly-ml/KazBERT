#!/usr/bin/env python
import os
import torch
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Настройки окружения для предотвращения ошибок CUDA и DDP
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# Логирование
log_filename = f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        logger.info(f"Запущено в режиме DDP. LOCAL_RANK = {local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Запущено в одиночном режиме на устройстве: {device}")

    # Используем кастомный токенизатор
    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"  # Укажи свой путь
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased",
        ignore_mismatched_sizes=True
    )
    model.to(device)

    dataset = load_dataset("json", data_files="/kaggle/input/kaz-rus-eng-wiki/train_pretrain.json")

    def tokenize_function(examples):
        return tokenizer(examples["masked_sentence"], truncation=True, max_length=128, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        logging_steps=100,
        evaluation_strategy="epoch",  # Заменено с "steps" на "epoch"
        save_steps=500,
        fp16=True,
        local_rank=local_rank,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"].select(range(1000)),  # Добавлен eval_dataset
        tokenizer=tokenizer,
    )

    logger.info("Начало обучения модели")
    train_result = trainer.train()
    logger.info("Обучение завершено")

    # Завершаем DDP
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Генерация графиков бенчмарков
    plot_benchmarks(train_result)

def plot_benchmarks(train_result):
    sns.set(style="whitegrid")
    
    # График изменения loss по шагам
    steps = list(range(len(train_result.training_loss_history)))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_result.training_loss_history, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    plt.savefig("training_loss.png")

    logger.info("График потерь сохранен как training_loss.png")

if __name__ == "__main__":
    main()
