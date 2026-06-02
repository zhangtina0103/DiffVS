# Shared SLURM setup. Submit from DiffVS repo root:
#   cd /path/to/DiffVS && sbatch slurm/<job>.sbatch

_submit="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${_submit}"

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
mkdir -p logs

activate_diffvs_env() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck source=/dev/null
    source /home/zhangtin/miniforge3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"
  elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.venv/bin/activate"
  else
    echo "ERROR: set CONDA_ENV or create ${REPO_ROOT}/.venv" >&2
    exit 1
  fi
}
