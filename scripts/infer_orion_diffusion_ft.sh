#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

DATASET_ROOT="${DATASET_ROOT:-/path/to/ORIONCRC_dataset_tile_20x}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./outputs/orion_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/orion_inference}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-Manojb/stable-diffusion-2-1-base}"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python src/diffvs/infer_diffusion_ft.py \
  --dataset orion \
  --dataset_root "${DATASET_ROOT}" \
  --split test \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_inference_steps 1 \
  "$@"
