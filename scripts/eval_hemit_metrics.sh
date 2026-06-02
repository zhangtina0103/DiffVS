#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

INFERENCE_DIR="${INFERENCE_DIR:-${REPO_DIR}/outputs/hemit_inference}"
# Dual-branch repo — post_process.py is invoked for identical metrics
export PIX2PIX_ROOT="${PIX2PIX_ROOT:-/home/zhangtin/Pix2pix_DualBranch}"

_extra=()
[[ -n "${EXPORT_DIR:-}" ]] && _extra+=(--export-dir "${EXPORT_DIR}")
[[ -n "${OUTPUT_CSV:-}" ]] && _extra+=(--output-csv "${OUTPUT_CSV}")

python "${REPO_DIR}/scripts/eval_hemit_metrics.py" \
  --inference_dir "${INFERENCE_DIR}" \
  --pix2pix-root "${PIX2PIX_ROOT}" \
  "${_extra[@]}" \
  "$@"
