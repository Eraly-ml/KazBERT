#!/usr/bin/env python
import os
import torch
import logging
import multiprocessing
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Оптимизация многопоточного использования CPU
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count() // 2)

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
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        os.environ["RANK"] = str(local_rank)
    else:
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}, Local Rank: {local_rank}")
    
    tokenizer_path = "/kaggle/input/kaz-eng-rus/pytorch/default/1"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    assert tokenizer.vocab_size == 30000, f"Ожидался словарь 30000, но получено {tokenizer.vocab_size}"
    
    model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    dataset = load_dataset("json", data_files="/kaggle/input/datafortrainmodelkazbert/train_pretrain_with_labels.json")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    def tokenize_function(examples):
        return tokenizer(examples["masked_sentence"], truncation=True, max_length=128, padding="max_length")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        ddp_find_unused_parameters=False if local_rank != -1 else True,
        local_rank=local_rank,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    )
    
    trainer.train()
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
