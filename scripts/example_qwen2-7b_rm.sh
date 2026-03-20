# Discliamer: the model used in the script is only for academic purpose.
set -x

# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
#   python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
#   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

export DATA_DIR=/data/user_data/jamesdin

# environment variables
export RAY_DEBUG=0  # set to 1 for debugging

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export n_gpus_per_node=2
# export n_cpus=4
export nnodes=1
export group_size=1  # 8, 16
export rollout_batch_size=8  # TODO: set to very small for testing
export update_batch_size=8  # # TODO: set to very small for testing, one step per episode
export ppo_micro_batch_size_per_device=1  # divisor of group_size * update_batch_size / n_gpus_per_node
export prob_ref_micro_batch_size_per_device=1  # divisor of group_size * update_batch_size / n_gpus_per_node
# export dataloader_num_workers=8  # 16

export train_files=[data/processed/msr_7_5k/train.parquet]
export test_files=[data/processed/msr_7_5k/test.parquet]

# project paths
export project_name=qwen_rm_gae
export experiment_name=qwen_rm_gae_${n_gpus_per_node}g_gs${group_size}_bs${rollout_batch_size}
export DATA_DIR=/data/user_data/jamesdin
export SAVE_PATH=${DATA_DIR}/outputs/${project_name}/${experiment_name}

# model paths
export model_path=${DATA_DIR}/models/Qwen3-0.6B  # very small model for testing
export reward_model_path=${DATA_DIR}/models/FsfairX-LLaMA3-RM-v0.1

# environment variables
export WANDB_API_KEY=$(jq -r '.WANDB_API_KEY' secret.json)
export WANDB_MODE=online
export WANDB_DIR=${SAVE_PATH}
export WANDB_CONFIG_DIR=${SAVE_PATH}
export DATE="$(TZ='Asia/Shanghai' date +%m%d_%H%M%S)"
# print, logging, and debug
export HYDRA_FULL_ERROR=1 
export PYTHONUNBUFFERED=1 
export MY_WORK_DIR=$(pwd)
export VERL_LOGGING_LEVEL=INFO
export CONSOLE_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_training.log
export LOGGER_OUTPUT_FILE=${SAVE_PATH}/${DATE}_verl_logging.log


mkdir -p $SAVE_PATH
touch $CONSOLE_OUTPUT_FILE $LOGGER_OUTPUT_FILE
chown -R tiger $SAVE_PATH 

echo "start rl=grpo, write to ${SAVE_PATH}"
echo "MY_PROMPT_TEMPLATE=$MY_PROMPT_TEMPLATE"

export RAY_DEDUP_LOGS=0

# prepare model ckpt
# huggingface-cli download Qwen/Qwen3-0.6B --local-dir $DATA_DIR/models/Qwen3-0.6B &
# huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir $DATA_DIR/models/Qwen3-4B-Instruct-2507 &
# huggingface-cli download sfairXC/FsfairX-LLaMA3-RM-v0.1 --local-dir $DATA_DIR/models/FsfairX-LLaMA3-RM-v0.1 &
# wait

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$rollout_batch_size \
    data.max_prompt_length=256 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$update_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_device \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$prob_ref_micro_batch_size_per_device \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$n_gpus_per_node \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_device \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.model_dtype=bf16 \
    reward_model.enable=True \
    reward_model.model.path=$reward_model_path \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    +reward_model.model.fsdp_config.model_dtype=bf16 \
    reward_model.micro_batch_size_per_gpu=$ppo_micro_batch_size_per_device \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.val_before_train=False \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
