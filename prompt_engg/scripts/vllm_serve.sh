#!/usr/bin/env bash

# Run this script from prompt_engg directory: bash scripts/vllm_serve.sh

set -euo pipefail

# Models:
  # meta-llama/Llama-3.2-3B-Instruct
  # Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
  # open-thoughts/OpenThinker3-7B
  # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  # Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

MODEL_REPO="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
HOST="0.0.0.0"
PORT="11632"
TP_SIZE="1"  # use multiple GPUs for larger models
MAX_LEN="127000"  # 127k tokens, enough for inference
GPU_UTIL="0.9"  # reserve 90% GPU memory for vllm
DTYPE="bfloat16"  # auto, bfloat16, fp8
API_KEY="vllm-api-key"

GPU_ID=0  # specify the GPU id to use when you have multiple GPUs

python -m vllm.entrypoints.api_server 
  --model "$MODEL_REPO" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --api-key "$API_KEY"
  --enable-expert-parallel

# vllm serve "$LOCAL/Qwen3-30B-local/models--Qwen--Qwen3-30B-A3B-Instruct-2027" \
#     --tensor-parallel-size 1 \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 11632 \
#     --api-key "vllm-api-key"