#!/bin/bash
# DiffVS HEMIT stage 2 (Diffusion-FT). Run after stage 1 finishes.
#
#   STAGE1_CHECKPOINT_DIR=./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100 \
#   sbatch slurm/train_hemit_stage2_diffusion_ft.sbatch
#SBATCH --job-name=diffvs_s2
#SBATCH --output=logs/diffvs_hemit_stage2_%j.out
#SBATCH --error=logs/diffvs_hemit_stage2_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env

export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
export STAGE1_CHECKPOINT_DIR="${STAGE1_CHECKPOINT_DIR:-${REPO_ROOT}/outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hemit_stage2_diffusion_ft}"
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"

echo "=== DiffVS HEMIT stage 2 ==="
echo "STAGE1_CHECKPOINT_DIR=${STAGE1_CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

bash "${REPO_ROOT}/scripts/train_hemit_stage2_diffusion_ft.sh" "$@"
