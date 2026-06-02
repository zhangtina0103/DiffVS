# Shared setup for DiffVS SLURM jobs. Submit from repo root:
#   sbatch slurm/train_hemit_stage1_marigold.sbatch

_submit="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "${_submit}/slurm/_common.sh" ]]; then
  REPO_ROOT="${_submit}"
elif [[ -f "${_submit}/../slurm/_common.sh" ]]; then
  REPO_ROOT="$(cd "${_submit}/.." && pwd)"
else
  echo "ERROR: submit sbatch from DiffVS repo root (need slurm/_common.sh)" >&2
  exit 1
fi

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
    echo "ERROR: no .venv and CONDA_ENV unset under ${REPO_ROOT}" >&2
    exit 1
  fi
}
