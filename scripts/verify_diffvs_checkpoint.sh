#!/usr/bin/env bash
# Quick sanity check before trusting inference panels.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./outputs/hemit_stage2_diffusion_ft/stage2-checkpoint-epoch-5}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/data}"
export CHECKPOINT_DIR DATASET_ROOT REPO_DIR

echo "=== DiffVS checkpoint verify ==="
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "DATASET_ROOT=${DATASET_ROOT}"

[[ -d "${CHECKPOINT_DIR}/unet" ]] || { echo "MISSING unet/"; exit 1; }
[[ -f "${CHECKPOINT_DIR}/marker_encoder.pt" ]] || { echo "MISSING marker_encoder.pt"; exit 1; }

CFG="${CHECKPOINT_DIR}/../config.json"
if [[ -f "${CFG}" ]]; then
  echo "--- ${CFG} ---"
  grep -E "pretrained_model|stage|single_step" "${CFG}" || true
else
  echo "WARN: no config.json at ${CFG}"
fi

if [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_DIR}/.venv/bin/activate"
fi
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

python - <<PY
import json
import os
from pathlib import Path

ckpt = Path(os.environ["CHECKPOINT_DIR"]).resolve()
cfg_path = ckpt.parent / "config.json"
if cfg_path.is_file():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print("stage:", cfg.get("stage"))
    print("pretrained_model:", cfg.get("pretrained_model"))
    print("single_step_timestep:", cfg.get("single_step_timestep"))
else:
    print("config not found:", cfg_path)
PY

echo "--- one-tile inference (fixed sampler) ---"
export OUTPUT_DIR="${REPO_DIR}/outputs/verify_one_tile"
rm -rf "${OUTPUT_DIR}"
bash scripts/infer_hemit_diffusion_ft.sh --max_rows 1

echo ""
echo "Panel: ${OUTPUT_DIR}/panels/"
ls -la "${OUTPUT_DIR}/panels/" 2>/dev/null || true
if [[ -f "${OUTPUT_DIR}/inference_meta.json" ]]; then
  echo "--- inference_meta.json ---"
  grep -E "infer_mode|pretrained" "${OUTPUT_DIR}/inference_meta.json" || cat "${OUTPUT_DIR}/inference_meta.json"
fi
echo ""
echo "Expect infer_mode: stage2_ddpm_t999 (not upstream_ddim_1step)"
