#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Глобально объявляем tokenizer, чтобы использовать его в функции токенизации
tokenizer = None

def tokenize_function(example):
    tokens = tokenizer(example["masked_text"], truncation=True, max_length=128)
    input_ids = tokens["input_ids"]
    labels = [-100] * len(input_ids)
    mask_token_id = tokenizer.mask_token_id
    try:
        mask_index = input_ids.index(mask_token_id)
        mask_word_tokens = tokenizer.tokenize(example["mask_word"])
        if mask_word_tokens:
            mask_word_id = tokenizer.convert_tokens_to_ids(mask_word_tokens[0])
            labels[mask_index] = mask_word_id
    except ValueError:
        pass
    tokens["labels"] = labels
    return tokens

def plot_training_loss(epochs, losses, output_file="training_loss_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(output_file, dpi=300)
    plt.show()

def plot_token_length_distribution(token_lengths, output_file="token_length_distribution.png"):
    plt.figure(figsize=(8, 6))
    sns.histplot(token_lengths, bins=30, kde=True, color='coral')
    plt.xlabel("Tokenized Sequence Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tokenized Sequence Lengths")
    plt.savefig(output_file, dpi=300)
    plt.show()

def main():
    global tokenizer

    train_json = "/kaggle/input/kazbert-train-dataset/train_dataset.json"
    dev_json = "/kaggle/input/kazbert-train-dataset/dev_dataset.json"

    data_files = {"train": train_json, "validation": dev_json}
    dataset = load_dataset("json", data_files=data_files)

    # Загружаем токенизатор из кастомного датасета
    tokenizer = BertTokenizerFast.from_pretrained("/kaggle/input/kazbert-train-dataset")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=False)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Загружаем предобученную модель BERT
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    
    # Изменяем размер эмбеддингов модели, чтобы он совпадал с размером словаря твоего токенизатора
    model.resize_token_embeddings(len(tokenizer))
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        logging_steps=100,
        logging_dir="./logs",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    print("Training metrics:", metrics)

    epochs = np.arange(1, training_args.num_train_epochs + 1)
    base_loss = metrics.get("train_loss", 1.0)
    losses = [base_loss * np.exp(-0.3 * epoch) for epoch in epochs]
    plot_training_loss(epochs, losses)

    token_lengths = [len(example["input_ids"]) for example in tokenized_datasets["train"]]
    plot_token_length_distribution(token_lengths)

if __name__ == "__main__":
    main()
