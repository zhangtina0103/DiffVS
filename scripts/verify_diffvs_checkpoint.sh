#!/usr/bin/env bash
# Compare 3 inference modes on ONE test tile.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

STAGE2_CKPT="${STAGE2_CKPT:-./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
STAGE1_CKPT="${STAGE1_CKPT:-./outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/data}"
export STAGE2_CKPT STAGE1_CKPT DATASET_ROOT REPO_DIR

echo "=== DiffVS one-tile comparison ==="
echo "STAGE2_CKPT=${STAGE2_CKPT}"
echo "STAGE1_CKPT=${STAGE1_CKPT}"

if [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_DIR}/.venv/bin/activate"
fi
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

run_one() {
  local label="$1" ckpt="$2" out="$3"
  shift 3
  echo ""
  echo "========== ${label} =========="
  rm -rf "${out}"
  CHECKPOINT_DIR="${ckpt}" OUTPUT_DIR="${out}" bash scripts/infer_hemit_diffusion_ft.sh --max_rows 1 "$@"
  grep infer_mode "${out}/inference_meta.json" || true
  ls "${out}/panels/" 2>/dev/null || true
}

# A) Stage-2, correct scheduler, H&E latent init (what you ran)
run_one "A: stage2 / source init" "${STAGE2_CKPT}" "${REPO_DIR}/outputs/verify_A_stage2_source"

# B) Stage-2, oracle init (matches training noise distribution — uses GT)
run_one "B: stage2 / target init ORACLE" "${STAGE2_CKPT}" "${REPO_DIR}/outputs/verify_B_stage2_oracle" \
  --stage2_init target

# C) Stage-1, 25 DDIM steps (best shot at usable images)
run_one "C: stage1 / 25 DDIM steps" "${STAGE1_CKPT}" "${REPO_DIR}/outputs/verify_C_stage1_25step" \
  --num_inference_steps 25

echo ""
echo "Open panels (right column = prediction):"
echo "  A ${REPO_DIR}/outputs/verify_A_stage2_source/panels/"
echo "  B ${REPO_DIR}/outputs/verify_B_stage2_oracle/panels/"
echo "  C ${REPO_DIR}/outputs/verify_C_stage1_25step/panels/"
echo ""
echo "How to read:"
echo "  B good, A bad  → stage-2 weights OK, fix test-time init (hard)"
echo "  B bad, A bad   → stage-2 training did not work"
echo "  C good         → use stage-1 checkpoint for results, not stage-2"
