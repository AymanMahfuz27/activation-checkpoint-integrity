#!/usr/bin/env bash
#SBATCH --job-name=vg-ba-dev-smoke
#SBATCH --account=CCR26063
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/ba-20261107/provenance-dev-%j.out
#SBATCH --error=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/ba-20261107/provenance-dev-%j.err
set -euo pipefail
project=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity
checkout="$project/slurm/ba-20261107/provenance-dev-env-source"
python="$project/tools/python/cpython-3.13.7-linux-x86_64-gnu/bin/python3.13"
uv="$project/tools/uv-0.12.3/uv"
record="$project/artifacts/$SLURM_JOB_ID"
mkdir -p "$record"
exec > >(tee -a "$record/ba-provenance-runner.log") 2>&1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MAX_JOBS=4
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$project/cache/uv" UV_PYTHON_INSTALL_DIR="$project/tools/python"
export PYTHONPATH="$checkout/experiments/ba_20261107"
export TORCH_EXTENSIONS_DIR="$project/cache/torch-extensions-ba-20261107"
worker_pid=
finish() {
  status=$?
  trap - EXIT
  trap '' USR1 TERM INT
  if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
    kill -TERM "$worker_pid" 2>/dev/null || true
    for ((second=0; second<550; second++)); do
      if ! kill -0 "$worker_pid" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$worker_pid" 2>/dev/null; then kill -KILL "$worker_pid" 2>/dev/null || true; fi
    wait "$worker_pid" || true
  fi
  printf '%s\n' "$status" > "$record/ba-provenance-exit-code.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 124' USR1 TERM INT
module load gcc/13.2.0 cuda/12.8
module -t list 2> "$record/ba-provenance-modules.txt"
local_root=$(mktemp -d "/tmp/aci-ba-provenance-$SLURM_JOB_ID.XXXXXX")
"$uv" venv --python "$python" "$local_root/environment"
"$uv" pip sync --python "$local_root/environment/bin/python" --require-hashes --link-mode copy --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match "$checkout/experiments/ba_20261107/provenance-environment.lock"
"$uv" pip freeze --python "$local_root/environment/bin/python" > "$record/ba-provenance-installed.txt"
"$local_root/environment/bin/python" "$checkout/experiments/ba_20261107/provenance_job.py" &
worker_pid=$!
wait "$worker_pid"
worker_pid=
