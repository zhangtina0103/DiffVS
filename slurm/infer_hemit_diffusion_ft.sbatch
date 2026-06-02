#!/bin/bash
# DiffVS HEMIT inference (stage 2 checkpoint, test split).
#
#   sbatch slurm/infer_hemit_diffusion_ft.sbatch
#   sbatch --export=ALL,CHECKPOINT_DIR=./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5 \
#     slurm/infer_hemit_diffusion_ft.sbatch
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
# Optional: export MAX_ROWS=32 for a quick run

echo "=== DiffVS HEMIT inference ==="
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"
nvidia-smi -L || true

_extra=()
if [[ -n "${MAX_ROWS:-}" ]]; then
  _extra+=(--max_rows "${MAX_ROWS}")
fi

bash "${REPO_ROOT}/scripts/infer_hemit_diffusion_ft.sh" "${_extra[@]}" "$@"
