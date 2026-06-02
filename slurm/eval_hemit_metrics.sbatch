#!/bin/bash
# README — metrics (after inference):
#   export PIX2PIX_ROOT=/home/zhangtin/Pix2pix_DualBranch
#   bash scripts/eval_hemit_metrics.sh
#
#   INFERENCE_DIR=./outputs/hemit_inference sbatch slurm/eval_hemit_metrics.sbatch
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

export PIX2PIX_ROOT="${PIX2PIX_ROOT:-/home/zhangtin/Pix2pix_DualBranch}"
export INFERENCE_DIR="${INFERENCE_DIR:-./outputs/hemit_inference}"

bash scripts/eval_hemit_metrics.sh
