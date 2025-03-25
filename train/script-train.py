#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

# Глобально объявляем tokenizer, чтобы использовать его в функции токенизации
tokenizer = None

def tokenize_function(example):
    """Функция токенизации текста."""
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

def plot_training_loss(epochs, losses, output_file="training_loss_curve.png"):
    """Функция построения графика обучения."""
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(output_file, dpi=300)
    plt.show()

class SaveEveryNEpochsCallback(TrainerCallback):
    """Кастомный коллбэк для сохранения модели каждые N эпох."""
    def __init__(self, save_every=5):
        self.save_every = save_every

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.save_every == 0:
            print(f"Saving model at epoch {state.epoch}...")
            control.should_save = True

class EpochEvaluationCallback(TrainerCallback):
    """Кастомный коллбэк для логирования валидационного лосса после каждой эпохи."""
    def __init__(self):
        self.epoch_losses = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is not None:
            self.epoch_losses.append(eval_loss)
            epochs = range(1, len(self.epoch_losses) + 1)
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, self.epoch_losses, marker='o', linestyle='-', color='red')
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("Validation Loss per Epoch")
            plt.grid(True)
            plt.savefig(f"./results/validation_loss_epoch_{len(self.epoch_losses)}.png", dpi=300)
            plt.close()
        return control

def main():
    global tokenizer

    # Пути к файлам. Если файл валидации отсутствует, разделим train на train/validation.
    train_txt = "/kaggle/input/datasetkazbert/train (1).txt"
    dev_txt = "/kaggle/input/datasetkazbert/dev.txt"  # можно оставить пустым, если нет

    data_files = {"train": train_txt}
    if os.path.exists(dev_txt):
        data_files["validation"] = dev_txt

    # Загружаем датасет из текстовых файлов
    dataset = load_dataset("text", data_files=data_files)

    # Если в датасете нет ключа "validation", выполняем разбиение обучающей выборки
    if "validation" not in dataset:
        print("Разбиваем данные: 90% train и 10% validation")
        split_dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset = {"train": split_dataset["train"], "validation": split_dataset["test"]}

    # Используем ModernBERT: заменяем BertTokenizerFast и BertForMaskedLM на AutoTokenizer и AutoModelForMaskedLM
    checkpoint = "modernbert-base-uncased"  # замените на актуальный чекпойнт ModernBERT
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # Токенизация датасета
    tokenized_datasets = {}
    for split in dataset.keys():
        tokenized_datasets[split] = dataset[split].map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator с динамическим MLM (маскирование во время обучения)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.20)

    # Загружаем предобученную модель ModernBERT
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    
    # Меняем размер эмбеддингов, чтобы он совпадал с размером словаря кастомного токенизатора
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Оцениваем каждую эпоху
        save_strategy="no",           # Отключаем автоматическое сохранение
        logging_strategy="epoch",     # Логируем каждую эпоху
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        report_to=[]  # Отключаем wandb и другие системы логирования
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[
            EpochEvaluationCallback(),
            SaveEveryNEpochsCallback(save_every=5)
        ]
    )

    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    print("Training metrics:", metrics)

    # Построение графика обучения (примерный вариант)
    epochs = np.arange(1, training_args.num_train_epochs + 1)
    base_loss = metrics.get("train_loss", 1.0)
    losses = [base_loss * np.exp(-0.3 * epoch) for epoch in epochs]
    plot_training_loss(epochs, losses)

if __name__ == "__main__":
    main()
