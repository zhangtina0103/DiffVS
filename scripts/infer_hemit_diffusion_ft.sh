#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/data}"
if [[ ! -d "${DATASET_ROOT}/test/input" ]]; then
  echo "ERROR: HEMIT test data not found at ${DATASET_ROOT}/test/input" >&2
  exit 1
fi
echo "DATASET_ROOT=${DATASET_ROOT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hemit_inference}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"

if [[ ! -d "${CHECKPOINT_DIR}/unet" ]]; then
  echo "ERROR: missing ${CHECKPOINT_DIR}/unet" >&2
  exit 1
fi

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
