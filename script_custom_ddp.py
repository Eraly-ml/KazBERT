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
    AutoModelForCausalLM
)
from datasets import load_dataset

# Оптимизация многопоточного использования CPU
os.environ["OMP_NUM_THREADS"] = "4"  

# Отключение повторной регистрации CUDA-функций
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# Настройки окружения для предотвращения ошибок CUDA и DDP
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# Логирование
log_filename = f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Определяем режим работы (DDP или одиночный)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Запущено в режиме DDP. LOCAL_RANK = {local_rank}")
    else:
        local_rank = -1  # Если не DDP, local_rank должен быть -1
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

    # Загружаем датасет
    dataset = load_dataset("json", data_files="/kaggle/input/kaz-rus-eng-wiki/train_pretrain.json")

    def tokenize_function(examples):
        # Токенизируем "masked_sentence"
        inputs = tokenizer(
            examples["masked_sentence"], 
            truncation=True, max_length=128, padding="max_length"
        )
        # Если в примерах присутствуют "labels", токенизируем их,
        # иначе используем input_ids как метки для вычисления loss
        if "labels" in examples:
            labels = tokenizer(
                examples["labels"], 
                truncation=True, max_length=128, padding="max_length"
            )["input_ids"]
            inputs["labels"] = torch.tensor(labels)
        else:
            inputs["labels"] = torch.tensor(inputs["input_ids"])
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        logging_steps=100,
        save_strategy="epoch",  # Сохраняем модель только по окончании каждой эпохи
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        evaluation_strategy="no",  # Отключаем валидацию
        **({"local_rank": local_rank} if local_rank != -1 else {}),  # Передаем local_rank только если используется DDP
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

    # Завершаем DDP, если он был инициализирован
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Генерация графиков бенчмарков
    plot_benchmarks(trainer.state.log_history)

    # Загрузка модели с поддержкой генерации (если нужно)
    model_name = "bert-base-multilingual-cased"
    causal_model = AutoModelForCausalLM.from_pretrained(model_name)
    causal_tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Модель успешно загружена!")

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
