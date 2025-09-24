#!/bin/bash

# Run this script from prompt_engg directory: bash scripts/generate_response.sh

set -euo pipefail

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
CLIENT="vllm"
DATA_FILE="val_data/MSR_BeaverTails_4x56_subset.json"
BATCH_SIZE=250
SCORE_PROMPT=True

python code/main.py \
    --model_name $MODEL_NAME \
    --client=$CLIENT \
    --data_file=$DATA_FILE \
    --batch_size=$BATCH_SIZE \
    --score_prompt=$SCORE_PROMPT
