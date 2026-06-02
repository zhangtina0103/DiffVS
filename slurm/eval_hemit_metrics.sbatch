#!/bin/bash
# Score DiffVS HEMIT inference with dual-branch post_process metrics (SSIM / Pearson / PSNR).
#
#   sbatch slurm/eval_hemit_metrics.sbatch
# Uses Pix2pix_DualBranch/post_process.py (set PIX2PIX_ROOT if needed)
#SBATCH --job-name=diffvs_eval
#SBATCH --output=logs/diffvs_hemit_eval_%j.out
#SBATCH --error=logs/diffvs_hemit_eval_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env

export INFERENCE_DIR="${INFERENCE_DIR:-${REPO_ROOT}/outputs/hemit_inference}"
export PIX2PIX_ROOT="${PIX2PIX_ROOT:-/home/zhangtin/Pix2pix_DualBranch}"

echo "INFERENCE_DIR=${INFERENCE_DIR}"
echo "PIX2PIX_ROOT=${PIX2PIX_ROOT}"
bash "${REPO_ROOT}/scripts/eval_hemit_metrics.sh" "$@"
