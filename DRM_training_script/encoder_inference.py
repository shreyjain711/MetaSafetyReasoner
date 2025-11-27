#!/usr/bin/env python3
"""
Inference script for ModernBERT fine-tuned model
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Configuration
MODEL_PATH = "./saferbench_ettin_encoder_1b_finetuned_epoch10_no_thinking_sentences/final_model"
TEST_FILE = "./finetuning_data/test_no_think_sentences.jsonl"
OUTPUT_FILE = "./test_predictions_ettin_encoder_1b_finetuned_epoch10_no_thinking_sentences.jsonl"
BATCH_SIZE = 16  # Can be larger for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_text_and_label(sample: Dict[str, Any]) -> tuple:
    """Extract text and label from sample."""
    return sample['prompt'], sample['completion']


class InferenceDataset(Dataset):
    """Simple dataset for inference."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }


def main():
    print("="*60)
    print("LOADING MODEL")
    print("="*60)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load label mappings
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Number of classes: {len(id2label)}")
    print(f"Labels: {list(id2label.values())}\n")
    
    # Move model to device
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    model.eval()
    
    # Load test data
    print("="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    test_data = load_jsonl(TEST_FILE)
    texts = []
    true_labels = []
    
    for sample in test_data:
        text, label = extract_text_and_label(sample)
        texts.append(text)
        true_labels.append(label)
    
    print(f"Loaded {len(texts)} test samples\n")
    
    # Convert labels to IDs
    true_label_ids = [label2id[label] for label in true_labels]
    
    # Create dataset and dataloader
    dataset = InferenceDataset(texts, tokenizer, max_length=tokenizer.model_max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    print("="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(pred_ids.cpu().numpy())
            # all_probs.extend(probs.cpu().numpy())
            all_probs.extend(probs.cpu().float().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(dataloader)} batches")
    
    # Convert predictions to labels
    pred_labels = [id2label[pred_id] for pred_id in all_predictions]
    
    print(f"Inference complete\n")
    
    # Compute metrics
    print("="*60)
    print("METRICS")
    print("="*60)
    
    accuracy = accuracy_score(true_label_ids, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_label_ids, all_predictions, average='weighted', zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        true_label_ids, all_predictions, average='macro', zero_division=0
    )[2]
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}\n")
    
    # Classification report
    print("="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        true_label_ids, 
        all_predictions, 
        target_names=[id2label[i] for i in range(len(id2label))]
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_label_ids, all_predictions)
    print("="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(cm)
    print()
    
    # Save predictions
    print("="*60)
    print("SAVING PREDICTIONS")
    print("="*60)
    
    predictions_data = []
    for i, (true_label, pred_label, probs) in enumerate(zip(true_labels, pred_labels, all_probs)):
        predictions_data.append({
            'text': texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],  # Truncate for readability
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': float(np.max(probs)),
            'probabilities': {id2label[j]: float(probs[j]) for j in range(len(probs))}
        })
    
    with open(OUTPUT_FILE, 'w') as f:
        for item in predictions_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Predictions saved to {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()