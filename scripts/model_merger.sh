#!/usr/bin/env bash
set -x

# Run inside MetaSafetyReasoner/verl
cd verl

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /data/user_data/jamesdin/outputs/qwen_msr_grpo_drm/qwen_msr_7_5_k_grpo_4g_gs4_bs64/global_step_1500/actor \
    --target_dir /data/user_data/jamesdin/ckpts/MetaSafetyReasoner-RL/qwen3_0.6b_msr_step1500_hf
