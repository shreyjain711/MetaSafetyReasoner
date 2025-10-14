#!/bin/bash

# Run this script from prompt_engg directory: bash scripts/generate_response.sh

set -euo pipefail

# Models:
    # meta-llama/Llama-3.2-3B-Instruct
    # Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
    # open-thoughts/OpenThinker3-7B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Datasets:
    # val_data/multi_turn_subset_224.json
    # val_data/MSR_BeaverTails_4x56_subset.json

MODEL_NAME="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
CLIENT="vllm"
DATA_FILE="val_data/multi_turn_subset_224.json"
BATCH_SIZE=250
SCORE_PROMPT=True
PROMPT_FEILD="user_input"

python code/main.py \
    --model_name $MODEL_NAME \
    --client=$CLIENT \
    --data_file=$DATA_FILE \
    --batch_size=$BATCH_SIZE \
    --score_prompt=$SCORE_PROMPT \
    --prompt_feild=$PROMPT_FEILD
