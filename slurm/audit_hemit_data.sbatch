#!/bin/bash
# CPU-only: check HEMIT paths and save one input/label PNG per split.
#   cd ~/DiffVS && sbatch slurm/audit_hemit_data.sbatch
#SBATCH --job-name=hemit_audit
#SBATCH --output=logs/hemit_audit_%j.out
#SBATCH --error=logs/hemit_audit_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh"
activate_diffvs_env
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data}"
bash scripts/audit_hemit_data.sh
