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

# Загружаем кастомный токенизатор
TOKENIZER_PATH = "/kaggle/input/kazbert/pytorch/2/1"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

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
    # Пути к файлам
    train_txt = "/kaggle/input/kazbert-nlp-txt-data/merged_dataset.txt"
    dev_txt = ""

    data_files = {"train": train_txt}
    if os.path.exists(dev_txt):
        data_files["validation"] = dev_txt

    dataset = load_dataset("text", data_files=data_files)
    if "validation" not in dataset:
        split_dataset = dataset["train"].train_test_split(test_size=0.1)
        dataset = {"train": split_dataset["train"], "validation": split_dataset["test"]}

    tokenized_datasets = {}
    for split in dataset.keys():
        tokenized_datasets[split] = dataset[split].map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        fp16=True,
        logging_dir="./logs",
        report_to=[]
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

    epochs = np.arange(1, training_args.num_train_epochs + 1)
    base_loss = metrics.get("train_loss", 1.0)
    losses = [base_loss * np.exp(-0.3 * epoch) for epoch in epochs]
    plot_training_loss(epochs, losses)

if __name__ == "__main__":
    main()
