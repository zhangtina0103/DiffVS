#!/bin/bash
# README — HEMIT inference:
#   CHECKPOINT_DIR=./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5 \
#   OUTPUT_DIR=./outputs/hemit_inference \
#   bash scripts/infer_hemit_diffusion_ft.sh
#
#   cd ~/DiffVS && sbatch slurm/infer_hemit_diffusion_ft.sbatch
#SBATCH --job-name=diffvs_inf
#SBATCH --output=logs/diffvs_hemit_infer_%j.out
#SBATCH --error=logs/diffvs_hemit_infer_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
export OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hemit_inference}"

bash scripts/infer_hemit_diffusion_ft.sh
