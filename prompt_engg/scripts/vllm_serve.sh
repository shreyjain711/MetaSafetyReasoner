#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"  # e.g. Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
HOST="0.0.0.0"
PORT="8000"
TP_SIZE="1"  # use multiple GPUs for larger models
MAX_LEN="131072"  # 131k tokens, enough for inference
GPU_UTIL="0.9"  # reserve 90% GPU memory for vllm
DTYPE="auto"  # auto, bf16, fp8
API_KEY="vllm-api-key"

vllm serve "$MODEL_REPO" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --api-key "$API_KEY"

# example usage:
# MODEL_REPO="Qwen/Qwen3-30B-A3B-Instruct-2507" bash prompt_engg/scripts/vllm_serve.sh