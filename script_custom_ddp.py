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
    TrainingArguments
)
from datasets import load_dataset

# Оптимизация многопоточного использования CPU
os.environ["OMP_NUM_THREADS"] = "8"  # Используем больше потоков для загрузки данных

# Отключение повторной регистрации CUDA-функций
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
torch.backends.cudnn.benchmark = True  # Улучшает производительность
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 ускоряет матмуль на Ampere (T4)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Оптимизация fp16
torch.backends.cuda.preferred_linalg_library("cublas")  # Оптимизация матриц

# Настройки окружения для DDP
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# Логирование
log_filename = f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    logger.info(f"Запущено в режиме DDP. LOCAL_RANK = {local_rank}")

    # Используем кастомный токенизатор
    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"  # Укажи свой путь
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased",
        ignore_mismatched_sizes=True
    )
    
    # 🔥 Включаем torch.compile для ускорения
    model = torch.compile(model, mode="max-autotune")
    model.to(device)

    # Загружаем датасет
    dataset = load_dataset("json", data_files="/kaggle/input/kaz-rus-eng-wiki/train_pretrain.json")

    def tokenize_function(examples):
        inputs = tokenizer(
            examples["masked_sentence"], 
            truncation=True, max_length=128, padding="max_length"
        )
        inputs["labels"] = torch.tensor(inputs["input_ids"])  # Masked LM loss
        return inputs

    # Токенизируем датасет
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, 
        remove_columns=dataset["train"].column_names
    )

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,  # 🔥 Увеличен batch size (T4 справится)
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,  # 🔥 Уменьшает VRAM usage
        learning_rate=5e-5,
        logging_steps=100,
        save_strategy="epoch",  # Сохраняем модель только по окончании каждой эпохи
        bf16=True,  # 🔥 bf16 лучше на T4, чем fp16
        gradient_checkpointing=True,  # 🔥 Снижает VRAM за счет пересчета градиентов
        dataloader_num_workers=8,  # Больше потоков загрузки
        report_to="none",
        evaluation_strategy="no",  # Отключаем валидацию
        ddp_find_unused_parameters=False,  # 🔥 Оптимизация DDP
    )

    # Создаем Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,  # Пока используется, хотя устаревает
    )

    logger.info("Начало обучения модели")
    train_result = trainer.train()
    logger.info("Обучение завершено")

    # Завершаем DDP
    torch.distributed.destroy_process_group()

    # Генерация графиков бенчмарков
    plot_benchmarks(trainer.state.log_history)

def plot_benchmarks(log_history):
    sns.set(style="whitegrid")
    
    losses = [log["loss"] for log in log_history if "loss" in log]
    steps = list(range(len(losses)))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    plt.savefig("training_loss.png")

    logger.info("График потерь сохранен как training_loss.png")

if __name__ == "__main__":
    main()
