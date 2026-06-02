#!/usr/bin/env bash
# README: CHECKPOINT_DIR=./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5 \
#         OUTPUT_DIR=./outputs/hemit_inference \
#         bash scripts/infer_hemit_diffusion_ft.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hemit_inference}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python src/diffvs/infer_diffusion_ft.py \
  --dataset hemit \
  --dataset_root "${DATASET_ROOT}" \
  --split test \
  --markers HEMIT \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_inference_steps 1 \
  "$@"
