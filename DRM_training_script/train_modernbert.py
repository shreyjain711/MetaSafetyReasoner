#!/usr/bin/env python3
"""
ModernBERT Model Fine-tuning Script for Sequence Classification
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import os
# import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from datasets import Dataset as HFDataset

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

set_seed(42)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_text_and_label(sample: Dict[str, Any]) -> tuple:
    """
    Extract the chunk text and label from a sample.

    For encoder models, we only need the chunk text (not the full instruction).
    We'll extract it from the prompt.
    """
    prompt = sample['prompt']
    label = sample['completion']
    return prompt, label


def prepare_dataset(file_path: str, verbose=True):
    """Load and prepare dataset."""
    raw_data = load_jsonl(file_path)

    texts = []
    labels = []

    for sample in raw_data:
        text, label = extract_text_and_label(sample)
        texts.append(text)
        labels.append(label)

    if verbose:
        print(f"\n Successfully loaded {len(texts)} samples from {file_path}")
        print(f"  Label distribution:")
        label_counts = pd.Series(labels).value_counts()
        for label, count in label_counts.items():
            print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")

    return texts, labels


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )[2]

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1
    }


def main():
    # Initialize distributed training
    # if 'RANK' in os.environ or 'LOCAL_RANK' in os.environ:
    #     import torch.distributed as dist
    #     dist.init_process_group(backend='nccl')
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()
    #     print(f"Initialized distributed training: rank {rank}/{world_size}")
    # else:
    #     rank = 0
    #     world_size = 1
    
    # Configuration
    TRAIN_FILE = './finetuning_data/train_no_think_sentences.jsonl'
    VAL_FILE = './finetuning_data/val_no_think_sentences.jsonl'
    TEST_FILE = './finetuning_data/test_no_think_sentences.jsonl'
    
    # MODEL_NAME = "answerdotai/ModernBERT-large"
    MODEL_NAME = "jhu-clsp/ettin-encoder-1b"


    BATCH_SIZE = 16  # Per GPU - can try 24 or 32 if memory allows
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    WARMUP_RATIO = 0.15
    WEIGHT_DECAY = 0.01
    OUTPUT_DIR = "./saferbench_ettin_encoder_1b_finetuned_epoch10_no_thinking_sentences"
    
    # Optimization flags
    USE_FLASH_ATTENTION = True
    USE_BF16 = True  # Better than FP16 for L40s

    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # is_main_process = local_rank == 0

    # if is_main_process:
    #     print("="*60)
    #     print("GPU INFORMATION")
    #     print("="*60)
    #     print(f"PyTorch version: {torch.__version__}")
    #     print(f"CUDA available: {torch.cuda.is_available()}")
    #     if torch.cuda.is_available():
    #         print(f"CUDA version: {torch.version.cuda}")
    #         print(f"Number of GPUs: {torch.cuda.device_count()}")
    #         for i in range(torch.cuda.device_count()):
    #             print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    #             print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    #     print("="*60 + "\n")
    
    # Check GPU availability
    print("="*60)
    print("GPU INFORMATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print("="*60 + "\n")
    
    # Load datasets
    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_texts, train_labels = prepare_dataset(TRAIN_FILE, verbose=True)
    val_texts, val_labels = prepare_dataset(VAL_FILE, verbose=True)
    test_texts, test_labels = prepare_dataset(TEST_FILE, verbose=True)
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Train: {len(train_texts)} samples")
    print(f"Val:   {len(val_texts)} samples")
    print(f"Test:  {len(test_texts)} samples")
    print(f"{'='*60}\n")
    
    # Create label mapping
    label_list = sorted(list(set(train_labels + val_labels + test_labels)))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    num_labels = len(label_list)
    
    print("Label Mapping:")
    for label, idx in label2id.items():
        print(f"  {idx}: {label}")
    print(f"\nNumber of classes: {num_labels}\n")
    
    # Convert labels to IDs
    train_label_ids = [label2id[label] for label in train_labels]
    val_label_ids = [label2id[label] for label in val_labels]
    test_label_ids = [label2id[label] for label in test_labels]
    
    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Enable FlashAttention 2 if available
    model_kwargs = {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
        "trust_remote_code": True,
        "ignore_mismatched_sizes": True,
        "torch_dtype": torch.bfloat16,
    }

    # Try to enable FlashAttention 2
    if USE_FLASH_ATTENTION:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Attempting to use FlashAttention 2")
        except Exception as e:
            print(f"Warning: FlashAttention 2 not available: {e}")
            print("Install with: pip install flash-attn --no-build-isolation")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        **model_kwargs
    )

    print(f"Successfully loaded model: {MODEL_NAME}")
    print(f"Successfully added classification head with {num_labels} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # Create HuggingFace datasets
    def create_hf_dataset(texts, labels):
        """Create HuggingFace dataset from texts and labels."""
        return HFDataset.from_dict({
            'text': texts,
            'label': labels
        })
    
    def tokenize_function(examples):
        """Tokenize a batch of examples."""
        return tokenizer(
            examples['text'],
            truncation=True,
            # padding='max_length',
            max_length=tokenizer.model_max_length,
        )
    
    MAX_LENGTH = tokenizer.model_max_length
    print(f"Using tokenizer's max length: {MAX_LENGTH}\n")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True  # This will pad to the longest sequence in each batch
        )
    
    print("Creating datasets...")
    train_hf = create_hf_dataset(train_texts, train_label_ids)
    # del train_texts, train_label_ids

    val_hf = create_hf_dataset(val_texts, val_label_ids)
    # del val_texts, val_label_ids
    
    test_hf = create_hf_dataset(test_texts, test_label_ids)
    # del test_texts, test_label_ids

    print("Tokenizing datasets...")
    train_dataset = train_hf.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=['text'],
        desc="Tokenizing train set",
        cache_file_name="./ettin_cache/train_no_thinking_sentences_tokenized.arrow",
        # num_proc=1
    )
    train_dataset = train_dataset.rename_column('label', 'labels')
    
    val_dataset = val_hf.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=['text'],
        desc="Tokenizing val set",
        cache_file_name="./ettin_cache/val_no_thinking_sentences_tokenized.arrow",
        # num_proc=1

    )
    val_dataset = val_dataset.rename_column('label', 'labels')
    
    test_dataset = test_hf.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=['text'],
        desc="Tokenizing test set",
        cache_file_name="./ettin_cache/test_no_thinking_sentences_tokenized.arrow",
        # num_proc=1
    )
    test_dataset = test_dataset.rename_column('label', 'labels')
    
    print("✓ Tokenization complete\n")

    # def tokenize_function(examples):
    #     """Tokenize a batch of examples."""
    #     return tokenizer(
    #         examples['text'],
    #         truncation=True,
    #         padding='max_length',
    #         max_length=tokenizer.model_max_length,
    #     )

    # MAX_LENGTH = tokenizer.model_max_length
    # print(f"Using tokenizer's max length: {MAX_LENGTH}\n")

    # print("Creating datasets...")
    # train_hf = create_hf_dataset(train_texts, train_label_ids)
    # val_hf = create_hf_dataset(val_texts, val_label_ids)
    # test_hf = create_hf_dataset(test_texts, test_label_ids)

    # import os
    # from accelerate import Accelerator

    # # Only process 0 should do the tokenization to avoid race conditions
    # accelerator = Accelerator()

    # # Use a shared cache directory that all processes can access
    # # cache_dir = "./tokenized_cache"
    # cache_dir = "./cache"
    # os.makedirs(cache_dir, exist_ok=True)

    # if accelerator.is_main_process:
    #     print("Main process: Tokenizing and caching datasets...")
        
    #     train_dataset = train_hf.map(
    #         tokenize_function,
    #         batched=True,
    #         batch_size=100,
    #         remove_columns=['text'],
    #         desc="Tokenizing train set",
    #         cache_file_name=f"{cache_dir}/train_tokenized.arrow"
    #     )
    #     train_dataset = train_dataset.rename_column('label', 'labels')
        
    #     val_dataset = val_hf.map(
    #         tokenize_function,
    #         batched=True,
    #         batch_size=100,
    #         remove_columns=['text'],
    #         desc="Tokenizing val set",
    #         cache_file_name=f"{cache_dir}/val_tokenized.arrow"
    #     )
    #     val_dataset = val_dataset.rename_column('label', 'labels')
        
    #     test_dataset = test_hf.map(
    #         tokenize_function,
    #         batched=True,
    #         batch_size=100,
    #         remove_columns=['text'],
    #         desc="Tokenizing test set",
    #         cache_file_name=f"{cache_dir}/test_tokenized.arrow"
    #     )
    #     test_dataset = test_dataset.rename_column('label', 'labels')
        
    #     print("Tokenization complete (main process)\n")

    # # Wait for main process to finish tokenization
    # accelerator.wait_for_everyone()

    # # All processes load the cached datasets
    # if not accelerator.is_main_process:
    #     print("Secondary process: Loading cached tokenized datasets...")
    #     train_dataset = train_hf.map(
    #         tokenize_function,
    #         batched=True,
    #         batch_size=1000,
    #         remove_columns=['text'],
    #         cache_file_name=f"{cache_dir}/train_tokenized.arrow",
    #         num_proc=1
    #     )
    #     train_dataset = train_dataset.rename_column('label', 'labels')
        
    #     val_dataset = val_hf.map(
    #         tokenize_function,
    #         batched=True,
    #         batch_size=1000,
    #         remove_columns=['text'],
    #         cache_file_name=f"{cache_dir}/val_tokenized.arrow",
    #         num_proc=1
    #     )
    #     val_dataset = val_dataset.rename_column('label', 'labels')
        
    #     test_dataset = test_hf.map(
    #         tokenize_function,
    #         batched=True,
    #         batch_size=1000,
    #         remove_columns=['text'],
    #         cache_file_name=f"{cache_dir}/test_tokenized.arrow",
    #         num_proc=1
    #     )
    #     test_dataset = test_dataset.rename_column('label', 'labels')

    # print(f"✓ Datasets ready (Process {accelerator.process_index})\n")

    # # Ensure process group is initialized for TrainingArguments
    # # This is needed even for single-GPU training with accelerate
    # if not torch.distributed.is_initialized():
    #     try:
    #         # Try to get rank from environment (set by accelerate)
    #         rank = int(os.environ.get('LOCAL_RANK', 0))
    #         world_size = int(os.environ.get('WORLD_SIZE', 1))
            
    #         if world_size > 1:
    #             # Multi-GPU: initialize properly
    #             torch.distributed.init_process_group(backend='nccl')
    #         else:
    #             # Single GPU: minimal initialization
    #             os.environ.setdefault('MASTER_ADDR', 'localhost')
    #             os.environ.setdefault('MASTER_PORT', '12355')
    #             torch.distributed.init_process_group(
    #                 backend='nccl' if torch.cuda.is_available() else 'gloo',
    #                 rank=0,
    #                 world_size=1
    #             )
    #     except Exception as e:
    #         print(f"Warning: Could not initialize process group: {e}")
            # Continue anyway - might work without it

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # ============================================================
        # TRAINING HYPERPARAMETERS
        # ============================================================
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Can be larger for eval
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type = "cosine",
        
        # ============================================================
        # OPTIMIZATION SETTINGS
        # ============================================================
        bf16=USE_BF16, 
        fp16=False,
        
        # Gradient accumulation
        gradient_accumulation_steps=2, 
        max_grad_norm=1.0,
        
        tf32=True,
        
        # ============================================================
        # DISTRIBUTED TRAINING SETTINGS
        # ============================================================
        # DDP settings (will be used when launching with torchrun)
        # ddp_find_unused_parameters=False,  # Faster
        # ddp_backend="nccl",  # Best for NVIDIA GPUs
        
        # ============================================================
        # DATA LOADING OPTIMIZATION
        # ============================================================
        dataloader_num_workers=8, 
        dataloader_pin_memory=True,  
        dataloader_prefetch_factor=2,
        
        # ============================================================
        # EVALUATION & SAVING
        # ============================================================
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # ============================================================
        # LOGGING
        # ============================================================
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        logging_first_step=True,
        report_to="wandb",  # or "tensorboard" / "wandb"
        run_name="saferbench_ettin_encoder_1b_finetuned_no_thinking_sentences",
        
        # ============================================================
        # CHECKPOINTING
        # ============================================================
        save_total_limit=5, 
        
        # ============================================================
        # MISC
        # ============================================================
        seed=42,
        remove_unused_columns=False,
        
        torch_compile=True,
    )

    # Calculate effective batch size
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = (
        BATCH_SIZE * 
        training_args.gradient_accumulation_steps * 
        num_gpus
    )

    print("="*60)
    print("OPTIMIZED TRAINING CONFIGURATION")
    print("="*60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Batch size per GPU: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Precision: {'BF16' if USE_BF16 else 'FP32'}")
    print(f"  FlashAttention 2: {USE_FLASH_ATTENTION}")
    print(f"  TF32: {training_args.tf32}")
    print(f"  DataLoader workers: {training_args.dataloader_num_workers}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")    

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator
    )
    
    print("Trainer initialized")
    print(f"Early stopping enabled (patience=3)\n")
    
    # Train
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    train_result = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
    print(f"Final training loss: {train_result.metrics['train_loss']:.4f}\n")
    
    # Save final model
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print(f"Model saved to {OUTPUT_DIR}/final_model\n")
    
    # Evaluate on test set
    print("="*60)
    print("EVALUATING ON TEST SET")
    print("="*60 + "\n")
    
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Set Results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}") 
    
    # Generate predictions for detailed analysis
    print("\nGenerating predictions...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_labels, pred_labels, target_names=[id2label[i] for i in range(num_labels)]))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\n" + "="*60)
    print("SCRIPT COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()

