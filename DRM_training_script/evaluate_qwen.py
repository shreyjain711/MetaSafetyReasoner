#!/usr/bin/env python3
"""
Evaluate fine-tuned Qwen3 model on chunk classification.
Supports either LoRA adapters (default) or merged checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import TextStreamer
from tqdm import tqdm

# --------------------------------------------------------------------
# Prompt template must match training
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
# --------------------------------------------------------------------


def load_data(test_path: Path) -> Dataset:
    """Load test.json into a HuggingFace Dataset."""
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f" Loaded {len(data)} test samples from {test_path}")
    return Dataset.from_list(data)


def format_example(instruction: str, input_text: str) -> str:
    """Format a single example using the Alpaca prompt."""
    return ALPACA_PROMPT.format(instruction, input_text or "", "")


def clean_prediction(text: str) -> str:
    """Extract the first token/word (label) from the model output."""
    # Remove everything before the Response marker if present
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()
    # Split by whitespace/newlines and keep first token
    return text.split()[0].strip()


# def load_model(args):
#     """Load either LoRA adapters or merged checkpoint."""
#     if args.model_type == "lora":
#         print(f"Loading base model + LoRA adapters from {args.model_path}")
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=str(args.model_path),
#             max_seq_length=args.max_seq_length,
#             load_in_4bit=False,
#             dtype="bfloat16",
#         )
#     else:
#         print(f"Loading merged model from {args.model_path}")
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=str(args.model_path),
#             max_seq_length=args.max_seq_length,
#             load_in_4bit=False,
#             dtype="bfloat16",  # or "float16" depending on GPU
#         )
#     FastLanguageModel.for_inference(model)
#     return model, tokenizer

def load_model(args):
    """Load either LoRA adapters or merged checkpoint."""
    if args.model_type == "lora":
        print(f"Loading base model + LoRA adapters from {args.model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.model_path),
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            # dtype="bfloat16",
            dtype=None,
        )
        FastLanguageModel.for_inference(model)
    else:
        print(f"Loading merged model from {args.model_path}")
        # For merged models, use standard transformers loading
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 chunk classifier")
    parser.add_argument("--data_dir", type=Path, default=Path("./LLaMA-Factory/data"))
    parser.add_argument("--test_file", type=str, default="test_with_think.json")
    parser.add_argument("--model_path", type=Path,
                        default=Path("/data/user_data/spachary/qwen3_1.7B_with_think_qlora/lora_model"))
    parser.add_argument("--model_type", choices=["lora", "merged"], default="lora",
                        help="Use 'lora' for adapters, 'merged' for merged model.")
    parser.add_argument("--max_seq_length", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Set >0 for sampling; 0 for greedy.")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    dataset = load_data(args.data_dir / args.test_file)
    model, tokenizer = load_model(args)

    gold_labels = []
    pred_labels = []

    # add tqdm progress bar
    for example in tqdm(dataset):
        prompt = format_example(example["instruction"], example["input"])
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        pred = clean_prediction(generated)
        pred_labels.append(pred)
        gold_labels.append(example["output"])

        # write the prompt, gold_label, and pred_label to a json file
        with open("qwen3_1.7B_with_think_qlora_predictions.jsonl", "a") as f:
            f.write(json.dumps({"prompt": format_example(example["instruction"], example["input"]), "gold_label": example["output"], "pred_label": pred}) + "\n")

    # Metrics
    print("\n=== Classification Report ===")
    print(classification_report(gold_labels, pred_labels, digits=4))

    acc = accuracy_score(gold_labels, pred_labels)
    macro_f1 = f1_score(gold_labels, pred_labels, average="macro")
    weighted_f1 = f1_score(gold_labels, pred_labels, average="weighted")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(gold_labels, pred_labels))

if __name__ == "__main__":
    main()