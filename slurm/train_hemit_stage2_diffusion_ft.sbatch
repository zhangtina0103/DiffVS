#!/bin/bash
# README — HEMIT stage 2:
#   STAGE1_CHECKPOINT_DIR=./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100 \
#   OUTPUT_DIR=./outputs/hemit_stage2_diffusion_ft \
#   NUM_PROCESSES=1 TRAIN_BATCH_SIZE=16 NUM_EPOCHS=5 \
#   bash scripts/train_hemit_stage2_diffusion_ft.sh
#
#   cd ~/DiffVS && sbatch slurm/train_hemit_stage2_diffusion_ft.sbatch
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

export STAGE1_CHECKPOINT_DIR="${STAGE1_CHECKPOINT_DIR:-./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hemit_stage2_diffusion_ft}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export NUM_EPOCHS="${NUM_EPOCHS:-5}"

bash scripts/train_hemit_stage2_diffusion_ft.sh
