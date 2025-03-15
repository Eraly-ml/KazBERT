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
import evaluate
import multiprocessing

# Используем половину ядер CPU
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count() // 2)

# Настройка XLA и CUDA (ошибки регистрации можно игнорировать)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Настройки для корректной работы DDP и токенизатора
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Подавляем лишние логи TensorFlow

# Логирование
log_filename = f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загружаем метрику accuracy с помощью evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Получаем предсказанные индексы по последнему измерению
    predictions = logits.argmax(axis=-1)
    # Маска для исключения паддинга (если pad_token_id задан)
    mask = labels != tokenizer.pad_token_id if tokenizer.pad_token_id is not None else None
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    result = accuracy_metric.compute(predictions=predictions, references=labels)
    return result

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

    # Загрузка токенизатора из указанного пути
    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Загрузка модели для Masked Language Modeling
    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased",
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Загрузка датасета из JSON файла
    dataset = load_dataset("json", data_files="/kaggle/input/datafortrainmodelkazbert/train_pretrain_with_labels.json")
    
    # Фильтрация дубликатов по полю 'masked_sentence'
    unique_sentences = set()
    dataset["train"] = dataset["train"].filter(
        lambda example: not (example["masked_sentence"] in unique_sentences or unique_sentences.add(example["masked_sentence"]))
    )
    
    # Разделение датасета: 80% для обучения, 20% для валидации
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    logger.info(f"Размер обучающего датасета: {len(dataset['train'])}")
    logger.info(f"Размер валидационного датасета: {len(dataset['test'])}")

    # Функция токенизации
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["masked_sentence"],
            truncation=True, max_length=128, padding="max_length"
        )
        labels = tokenizer(
            examples["labels"],
            truncation=True, max_length=128, padding="max_length"
        )["input_ids"]
        inputs["labels"] = labels
        return inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Аргументы тренировки
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
        **({"local_rank": local_rank} if local_rank != -1 else {})
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Передаем функцию вычисления метрик
    )

    logger.info("Начало обучения модели")
    trainer.train()
    logger.info("Обучение завершено")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    plot_benchmarks(trainer.state.log_history)

    # Пример загрузки модели для генерации (Causal LM)
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
    plt.xlabel("Шаги")
    plt.ylabel("Потери (Loss)")
    plt.title("График изменения потерь во времени")
    plt.legend()
    plt.savefig("training_loss.png")
    logger.info("График потерь сохранен как training_loss.png")

if __name__ == "__main__":
    main()
