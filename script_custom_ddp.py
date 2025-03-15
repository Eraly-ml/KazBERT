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
import multiprocessing
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count() // 2)  # Половина ядер

# Отключение повторной регистрации CUDA-функций
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

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

    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased",
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))  # Обновляем размер токенов
    model.to(device)

    dataset = load_dataset("json", data_files="/kaggle/input/datafortrainmodelkazbert/train_pretrain_with_labels.json")
    unique_sentences = set()
    dataset["train"] = dataset["train"].filter(lambda example: not (example["masked_sentence"] in unique_sentences or unique_sentences.add(example["masked_sentence"])))
    train_size = int(0.5 * len(dataset["train"]))
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
    logger.info(f"Размер нового датасета: {len(dataset['train'])}")

    def tokenize_function(examples):
        inputs = tokenizer(
            examples["masked_sentence"], 
            truncation=True, max_length=128, padding="max_length"
        )
        labels = tokenizer(
            examples["masked_sentence"],  # labels должны соответствовать входу
            truncation=True, max_length=128, padding="max_length"
        )["input_ids"]
        inputs["labels"] = labels  # labels передаются правильно
        return inputs

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, 
        remove_columns=dataset["train"].column_names
    )

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
        evaluation_strategy="no",
        **({"local_rank": local_rank} if local_rank != -1 else {}),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    logger.info("Начало обучения модели")
    trainer.train()
    logger.info("Обучение завершено")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    plot_benchmarks(trainer.state.log_history)
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
