#!/bin/bash
# One-tile sanity check (paths + fixed stage-2 inference).
#
#   cd ~/DiffVS && sbatch slurm/verify_diffvs_checkpoint.sbatch
#
#   CHECKPOINT_DIR=./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100 \
#   sbatch slurm/verify_diffvs_checkpoint.sbatch
#SBATCH --job-name=diffvs_verify
#SBATCH --output=logs/diffvs_verify_%j.out
#SBATCH --error=logs/diffvs_verify_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"

bash scripts/verify_diffvs_checkpoint.sh
