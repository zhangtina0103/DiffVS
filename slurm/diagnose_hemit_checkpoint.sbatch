#!/bin/bash
# GPU: VAE roundtrip + stage-1 DDIM on one tile (epoch-100 can still be broken).
#   cd ~/DiffVS && sbatch slurm/diagnose_hemit_checkpoint.sbatch
#SBATCH --job-name=hemit_diag
#SBATCH --output=logs/hemit_diag_%j.out
#SBATCH --error=logs/hemit_diag_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
export CKPT="${CKPT:-${REPO_ROOT}/outputs/hemit_stage1_marigold/stage1-checkpoint-epoch-100}"
bash scripts/diagnose_hemit_checkpoint.sh
