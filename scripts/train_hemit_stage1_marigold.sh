#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/data}"
if [[ ! -d "${DATASET_ROOT}/train/input" ]]; then
  echo "ERROR: HEMIT data not found at ${DATASET_ROOT}/train/input" >&2
  echo "  Put data under DiffVS/data/{train,val,test}/{input,label}/ or set:" >&2
  echo "  DATASET_ROOT=/path/to/HEMIT bash scripts/train_hemit_stage1_marigold.sh" >&2
  exit 1
fi
echo "DATASET_ROOT=${DATASET_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hemit_stage1_marigold}"
# stabilityai/stable-diffusion-2-1-base was removed from HF (late 2025); use mirror or local dir
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

accelerate launch --num_processes "${NUM_PROCESSES}" src/diffvs/train_stage1_marigold.py \
  --dataset hemit \
  --dataset_root "${DATASET_ROOT}" \
  --split train \
  --markers HEMIT \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --gradient_accumulation_steps 1 \
  --mixed_precision "${MIXED_PRECISION}" \
  --save_every 5 \
  --output_dir "${OUTPUT_DIR}" \
  "$@"

