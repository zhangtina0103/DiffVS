#!/bin/bash
# DiffVS HEMIT stage-1 only: multi-step DDIM (NOT for stage-2 checkpoints).
#
#   CHECKPOINT_DIR=./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100 \
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

export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hemit_stage1_inference}"
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"

echo "=== DiffVS HEMIT stage-1 DDIM ==="
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS}"

_extra=(--num_inference_steps "${NUM_INFERENCE_STEPS}")
[[ -n "${MAX_ROWS:-}" ]] && _extra+=(--max_rows "${MAX_ROWS}")

bash "${REPO_ROOT}/scripts/infer_hemit_diffusion_ft.sh" "${_extra[@]}" "$@"
