set -x

# environment variables
# export RAY_DEBUG=1  # set to 1 for debugging

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export n_gpus_per_node=2
# export n_cpus=4
export nnodes=1
export group_size=2  # 8, 16
export rollout_batch_size=32  # TODO: set to very small for testing
export update_batch_size=16  # # TODO: set to very small for testing, one step per episode
export ppo_micro_batch_size_per_device=4  # divisor of group_size * update_batch_size / n_gpus_per_node
export prob_ref_micro_batch_size_per_device=4  # divisor of group_size * update_batch_size / n_gpus_per_node
# export dataloader_num_workers=8  # 16

export train_files=[data/processed/msr_7_5k/train.parquet]
export test_files=[data/processed/msr_7_5k/test.parquet]

# project paths
export project_name=qwen_msr_grpo_drm
export experiment_name=qwen_msr_7_5_k_grpo_${n_gpus_per_node}g_gs${group_size}_bs${rollout_batch_size}
export DATA_DIR=/data/user_data/jamesdin
export SAVE_PATH=${DATA_DIR}/outputs/${project_name}/${experiment_name}

# model paths
export model_path=${DATA_DIR}/models/Qwen3-0.6B  # very small model for testing
export reward_model_path=${DATA_DIR}/models/ettin-1b-drm-safety-classifier

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

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_process \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$rollout_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$update_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_device \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_device \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$n_gpus_per_node \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$group_size \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$prob_ref_micro_batch_size_per_device \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    reward_model.enable=True \
    reward_model.worker_type=chunk \
    reward_model.strategy=none \
    reward_model.model.path=$reward_model_path \
    reward_model.model.trust_remote_code=True \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    +reward_model.model.fsdp_config.model_dtype=bf16 \
    reward_model.micro_batch_size_per_gpu=$ppo_micro_batch_size_per_device \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.rollout_data_dir=$SAVE_PATH/rollouts \
    trainer.validation_data_dir=$SAVE_PATH/evaluations \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15