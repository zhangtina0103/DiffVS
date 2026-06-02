#!/bin/bash
# DiffVS HEMIT stage 1 (Marigold-style, HF UNet).
#
#   cd /home/zhangtin/DiffVS
#   sbatch slurm/train_hemit_stage1_marigold.sbatch
#
# Cluster max walltime is 6h — 100 epochs may need several jobs (no resume in stage1 yet).
#   sbatch --export=ALL,NUM_EPOCHS=15 slurm/train_hemit_stage1_marigold.sbatch
# Overrides (optional):
#   sbatch --export=ALL,TRAIN_BATCH_SIZE=8 slurm/train_hemit_stage1_marigold.sbatch
#   sbatch --export=ALL,PRETRAINED_MODEL=/home/zhangtin/DiffVS/pretrained/sd21-base slurm/train_hemit_stage1_marigold.sbatch
#SBATCH --job-name=diffvs_s1
#SBATCH --output=logs/diffvs_hemit_stage1_%j.out
#SBATCH --error=logs/diffvs_hemit_stage1_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env

export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hemit_stage1_marigold}"
# stabilityai/stable-diffusion-2-1-base removed from HF; mirror or local dir
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# Hugging Face cache (optional; set on cluster if shared cache exists)
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

echo "=== DiffVS HEMIT stage 1 ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} NUM_EPOCHS=${NUM_EPOCHS}"
nvidia-smi -L || true

bash "${REPO_ROOT}/scripts/train_hemit_stage1_marigold.sh" "$@"
