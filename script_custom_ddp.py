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
import numpy as np

# Оптимизация многопоточного использования CPU
import multiprocessing
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count() // 2)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
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
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Проверка и установка паддинг-токена
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    
    model = AutoModelForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased",
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Синхронизация max_length с max_position_embeddings
    max_length = min(tokenizer.model_max_length, model.config.max_position_embeddings)
    tokenizer.model_max_length = max_length
    print(f"Определенная max_length: {max_length}")

    dataset = load_dataset("json", data_files="/kaggle/input/kaz-rus-eng-wiki/train_pretrain.json")
    unique_sentences = set()
    dataset["train"] = dataset["train"].filter(lambda x: not (x["masked_sentence"] in unique_sentences or unique_sentences.add(x["masked_sentence"])))
    
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    
    def tokenize_function(examples):
        inputs = tokenizer(examples["masked_sentence"], truncation=True, max_length=max_length, padding="max_length")
        inputs["labels"] = inputs["input_ids"]
        return inputs
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=100,
        save_strategy="steps",
        save_steps=len(train_dataset) // (2 * 16),
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        evaluation_strategy="steps",  # Включаем валидацию
        eval_steps=len(train_dataset) // (2 * 16),  # Валидация каждые 0.5 эпохи
        **({"local_rank": local_rank} if local_rank != -1 else {}),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
