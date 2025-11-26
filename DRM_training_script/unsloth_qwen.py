#!/usr/bin/env python3
"""
Unsloth Fine-tuning Script
Adapted from Unsloth examples for Qwen3-1.7B
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
import os
# Workaround for torchvision compatibility issue
# os.environ['TORCHVISION_DISABLE_IMAGE_BACKEND'] = '1'

# os.environ["UNSLOTH_NUM_PROC"] = "4"   # or any number your node can handle
# os.environ["HF_DATASETS_NUM_PROC"] = "4"  # optional, same intent
# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"  # Qwen3-1.7B from Unsloth
DATA_DIR = "./LLaMA-Factory/data"
OUTPUT_DIR = "/data/user_data/spachary/temp_output"

# Training hyperparameters
MAX_SEQ_LENGTH = 4096 
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 5e-6
NUM_EPOCHS = 3
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

# Model loading
LOAD_IN_4BIT = True 
DTYPE = None 
NUM_PROC = 4
# ============================================================================
# LOAD DATA
# ============================================================================
def load_json_data(file_path):
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

print("="*60)
print("LOADING DATA")
print("="*60)

train_data = load_json_data(f"{DATA_DIR}/train.json")
val_data = load_json_data(f"{DATA_DIR}/val.json")

print(f"Loaded {len(train_data)} training samples")
print(f"Loaded {len(val_data)} validation samples")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "="*60)
print("LOADING MODEL")
print("="*60)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    device_map={"": f"cuda:{local_rank}"},
    # token="hf_...",  # Uncomment if using gated models
)

print(f"Model loaded: {MODEL_NAME}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")

# ============================================================================
# ADD LoRA ADAPTERS
# ============================================================================
print("\n" + "="*60)
print("ADDING LoRA ADAPTERS")
print("="*60)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print(f"LoRA adapters added (r={LORA_R}, alpha={LORA_ALPHA})")
# ============================================================================
# FORMAT DATA
# ============================================================================
print("\n" + "="*60)
print("FORMATTING DATA")
print("="*60)

# Alpaca-style prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    """Format examples into the prompt template."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Format the prompt and add EOS token
        text = alpaca_prompt.format(
            instruction,
            input_text if input_text else "",  # Handle empty input
            output
        ) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

# Apply formatting
train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
    # num_proc=NUM_PROC,
    batch_size=100,
    desc="Formatting training data",
    cache_file_name=f"{OUTPUT_DIR}/.cache/train_formatted_no_think.arrow"
)

val_dataset = val_dataset.map(
    formatting_prompts_func,
    batched=True,
    # num_proc=NUM_PROC,
    batch_size=100,
    cache_file_name=f"{OUTPUT_DIR}/.cache/val_formatted_no_think.arrow",
    desc="Formatting validation data"
)

print("Data formatted")

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)

# GPU stats
if torch.cuda.is_available():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Starting memory: {start_gpu_memory} GB")

print(f"\nTraining hyperparameters:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRAD_ACCUM}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Weight decay: {WEIGHT_DECAY}")

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Add validation set
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Larger for eval
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="wandb",  # Change to "none" if not using wandb
        run_name="qwen3_1.7b_saferbench_classification",
        # run_name="test_run",
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        bf16=True,  # Use bf16 for better performance
        dataset_num_proc=8, 
        ddp_find_unused_parameters = False
    ),
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

# Train
trainer_stats = trainer.train()

# ============================================================================
# TRAINING STATS
# ============================================================================
print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)

if torch.cuda.is_available():
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak reserved memory: {used_memory} GB")
    print(f"Peak reserved memory for training: {used_memory_for_lora} GB")
    print(f"Peak reserved memory % of max: {used_percentage}%")
    print(f"Peak reserved memory for training % of max: {lora_percentage}%")

print(f"\nFinal training loss: {trainer_stats.metrics['train_loss']:.4f}")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save LoRA adapters
model.save_pretrained(f"{OUTPUT_DIR}/lora_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_model")
print(f"✓ LoRA adapters saved to {OUTPUT_DIR}/lora_model")

# Save merged model
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged_model",
    tokenizer,
    save_method="merged_16bit",  # or "merged_4bit"
)
print(f"Merged model saved to {OUTPUT_DIR}/merged_model")

print("\n" + "="*60)
print("SCRIPT COMPLETE")
print("="*60)