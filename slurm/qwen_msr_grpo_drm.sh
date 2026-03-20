#!/usr/bin/bash
#SBATCH --job-name=qwen3-0.6B-grpo-drm
#SBATCH --partition=general          # general partition
#SBATCH --time=47:00:00              # <= 48h limit on general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # 1 task per GPU
#SBATCH --gres=gpu:A6000:4           # 4 GPUs
#SBATCH --cpus-per-task=8            # 8 * 4 = 32 CPUs total
#SBATCH --mem=180GB                  # reasonable cpu memory
#SBATCH --output=/home/jamesdin/logs/verl-rl-%j.out
#SBATCH --error=/home/jamesdin/logs/verl-rl-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jamesdin@andrew.cmu.edu

SCRATCH_BASE=/scratch/$USER
export TMPDIR=${SCRATCH_BASE}/job_${SLURM_JOB_ID}

mkdir -p "${TMPDIR}"

# Ray will put sessions/logs here instead of /tmp
export RAY_TEMP_DIR="${TMPDIR}/ray"
mkdir -p "${RAY_TEMP_DIR}"

echo "Using TMPDIR=${TMPDIR}"
echo "Using RAY_TEMP_DIR=${RAY_TEMP_DIR}"
echo "==== Job started on $(hostname) at $(date) ===="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

mkdir -p /home/jamesdin/logs
export PYTHONUNBUFFERED=1

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

export nnodes=${SLURM_NNODES:-1}
export n_gpus_per_node=${SLURM_GPUS_ON_NODE:-4}
export WORLD_SIZE=$(( nnodes * n_gpus_per_node ))

echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"

# Launch your RL script (which sets n_gpus_per_node=4, n_cpus=16, etc.)
RAY_DEBUG=0 bash scripts/babel_qwen_msr_grpo_drm.sh

echo "==== Job finished at $(date) ===="
