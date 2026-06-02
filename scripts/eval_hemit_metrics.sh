#!/usr/bin/env bash
# After README inference; scores via Pix2pix post_process.py (same CSV as dual-branch).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

INFERENCE_DIR="${INFERENCE_DIR:-${REPO_DIR}/outputs/hemit_inference}"
PIX2PIX_ROOT="${PIX2PIX_ROOT:-/home/zhangtin/Pix2pix_DualBranch}"

python "${REPO_DIR}/scripts/eval_hemit_metrics.py" \
  --inference_dir "${INFERENCE_DIR}" \
  --pix2pix-root "${PIX2PIX_ROOT}" \
  "$@"
