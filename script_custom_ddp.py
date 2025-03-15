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
from datasets import load_dataset, load_metric

# Логирование
log_filename = f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=torch.tensor(labels))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Запущено на устройстве: {device}")

    # Используем кастомный токенизатор
    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased",
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Исправление размера словаря
    tokenizer_vocab_size = tokenizer.vocab_size
    model_vocab_size = model.config.vocab_size
    if tokenizer_vocab_size != model_vocab_size:
        logger.info(f"Изменение размера словаря модели: {model_vocab_size} -> {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

    # Загружаем датасет
    dataset = load_dataset("json", data_files="/kaggle/input/datafortrainmodelkazbert/train_pretrain_with_labels.json")
    dataset = dataset["train"].train_test_split(test_size=0.1)  # 90% train, 10% validation
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["masked_sentence"], truncation=True, max_length=128, padding="max_length")
        inputs["labels"] = inputs["input_ids"]
        return inputs
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=500,  # Оценка каждые 500 шагов
        save_strategy="steps",
        save_steps=1000,
        learning_rate=5e-5,
        fp16=True,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Начало обучения модели")
    trainer.train()
    logger.info("Обучение завершено")

if __name__ == "__main__":
    main()
