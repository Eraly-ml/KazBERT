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
    AutoModelForCausalLM  # если нужна модель для генерации текста (например, для дальнейшего сравнения)
)
from datasets import load_dataset
import multiprocessing

# Оптимизация многопоточного использования CPU
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count() // 2)  # Используем половину ядер

# Настройки CUDA и XLA (при необходимости)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Настройки окружения для корректной работы DDP и токенизатора
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

# Логирование
log_filename = f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Настройка распределённого обучения, если используется DDP
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Запущено в режиме DDP. LOCAL_RANK = {local_rank}")
    else:
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Запущено в одиночном режиме на устройстве: {device}")

    # Загрузка токенизатора из указанного пути
    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Загрузка модели для задачи Masked Language Modeling
    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased",
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Загрузка датасета из JSON файла
    dataset = load_dataset("json", data_files="/kaggle/input/datafortrainmodelkazbert/train_pretrain_with_labels.json")

    # Удаление дубликатов по полю 'masked_sentence'
    unique_sentences = set()
    dataset["train"] = dataset["train"].filter(lambda example: not (example["masked_sentence"] in unique_sentences or unique_sentences.add(example["masked_sentence"])))

    # Разделение на обучающую (80%) и валидационную (20%) выборки
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    logger.info(f"Размер обучающего датасета: {len(dataset['train'])}")
    logger.info(f"Размер валидационного датасета: {len(dataset['test'])}")

    # Функция токенизации: входом является замаскированное предложение, метками – оригинальное предложение
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["masked_sentence"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        # Преобразуем оригинальное предложение в последовательность токенов для меток
        labels = tokenizer(
            examples["labels"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )["input_ids"]
        inputs["labels"] = labels
        return inputs

    # Применяем токенизацию ко всему датасету (удаляются исходные столбцы)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Настройка аргументов тренировки
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=100,
        save_strategy="steps",
        save_steps=len(tokenized_dataset["train"]) // (2 * 16),
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=500,
        **({"local_rank": local_rank} if local_rank != -1 else {}),
    )

    # Инициализация тренера с распределёнными данными (если имеются) и валидационным датасетом
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    logger.info("Начало обучения модели")
    trainer.train()
    logger.info("Обучение завершено")

    # Если используется DDP, закрываем процессную группу
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Визуализация тренировки: график потерь
    plot_benchmarks(trainer.state.log_history)

    # Пример загрузки модели для генерации (Causal LM) для дальнейших экспериментов
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
