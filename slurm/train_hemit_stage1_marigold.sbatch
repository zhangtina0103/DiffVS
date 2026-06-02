#!/bin/bash
# README — HEMIT stage 1:
#   OUTPUT_DIR=./outputs/hemit_stage1_marigold \
#   NUM_PROCESSES=1 TRAIN_BATCH_SIZE=16 NUM_EPOCHS=100 \
#   bash scripts/train_hemit_stage1_marigold.sh
#
#   cd ~/DiffVS && sbatch slurm/train_hemit_stage1_marigold.sbatch
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

export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hemit_stage1_marigold}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export NUM_EPOCHS="${NUM_EPOCHS:-100}"

bash scripts/train_hemit_stage1_marigold.sh
