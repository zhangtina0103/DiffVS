#!/usr/bin/env bash
# Use this if stage-2 panels look like rainbow noise (common on HEMIT).
# Stage-1 + 25 DDIM steps matches what the released UNet actually learned best.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/hemit_stage1_inference}"

exec bash "${REPO_DIR}/scripts/infer_hemit_diffusion_ft.sh" --num_inference_steps 25 "$@"
