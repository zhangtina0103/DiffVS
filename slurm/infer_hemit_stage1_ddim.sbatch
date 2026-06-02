#!/bin/bash
# README-style inference using STAGE-1 checkpoint (25 DDIM steps — use if stage-2 looks like noise).
#
#   sbatch slurm/infer_hemit_stage1_ddim.sbatch
#SBATCH --job-name=diffvs_s1inf
#SBATCH --output=logs/diffvs_hemit_s1_infer_%j.out
#SBATCH --error=logs/diffvs_hemit_s1_infer_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hemit_stage1_inference}"

bash scripts/infer_hemit_diffusion_ft.sh --num_inference_steps 25
