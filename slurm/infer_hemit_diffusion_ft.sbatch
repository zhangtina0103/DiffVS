#!/bin/bash
# DiffVS HEMIT inference (stage-2 Diffusion-FT, fixed init @ t=999).
#
#   cd ~/DiffVS && sbatch slurm/infer_hemit_diffusion_ft.sbatch
#
# Quick test (8 tiles):
#   MAX_ROWS=8 OUTPUT_DIR=./outputs/hemit_inference_smoke sbatch slurm/infer_hemit_diffusion_ft.sbatch
#
# Then metrics (dual-branch post_process):
#   INFERENCE_DIR=./outputs/hemit_inference sbatch slurm/eval_hemit_metrics.sbatch
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

export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/hemit_inference}"
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"
export SINGLE_STEP_TIMESTEP="${SINGLE_STEP_TIMESTEP:-999}"
export INIT_MODE="${INIT_MODE:-source}"

if [[ ! -d "${CHECKPOINT_DIR}/unet" ]]; then
  echo "ERROR: missing ${CHECKPOINT_DIR}/unet (use stage2-checkpoint-epoch-*)" >&2
  exit 1
fi

echo "=== DiffVS HEMIT inference (stage 2) ==="
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "PRETRAINED_MODEL=${PRETRAINED_MODEL}"
echo "SINGLE_STEP_TIMESTEP=${SINGLE_STEP_TIMESTEP} INIT_MODE=${INIT_MODE}"
nvidia-smi -L || true

_extra=(--single_step_timestep "${SINGLE_STEP_TIMESTEP}" --init_mode "${INIT_MODE}")
[[ -n "${MAX_ROWS:-}" ]] && _extra+=(--max_rows "${MAX_ROWS}")

bash "${REPO_ROOT}/scripts/infer_hemit_diffusion_ft.sh" "${_extra[@]}" "$@"
