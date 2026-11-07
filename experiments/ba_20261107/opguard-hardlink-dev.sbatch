#!/usr/bin/env bash
#SBATCH --job-name=vg-ba-opguard-hlink-dev
#SBATCH --account=CCR26063
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/ba-20261107/opguard-hardlink-dev-%j.out
#SBATCH --error=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/ba-20261107/opguard-hardlink-dev-%j.err
set -euo pipefail
project=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity
checkout="$project/slurm/ba-20261107/opguard-hardlink-source"
python="$project/tools/python/cpython-3.13.7-linux-x86_64-gnu/bin/python3.13"
record="$project/artifacts/$SLURM_JOB_ID"
mkdir -p "$record"
exec > >(tee -a "$record/ba-opguard-runner.log") 2>&1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MAX_JOBS=4
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1 UV_COMPILE_BYTECODE=0
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPGUARD_AUTOSTART=0 OPGUARD_XOR_REGIONAL=4 OPGUARD_RUNTIME_CALLSTACK=1
export OPGUARD_TRACING_BUILD=1 OPGUARD_CALLSITE_TRACER=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
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
  printf '%s\n' "$status" > "$record/ba-opguard-exit-code.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 124' USR1 TERM INT
module load gcc/13.2.0 cuda/12.8
module -t list 2> "$record/ba-opguard-modules.txt"
"$python" -B "$checkout/experiments/ba_20261107/hardlink_runtime.py" --study opguard &
worker_pid=$!
wait "$worker_pid"
worker_pid=
cp "$record/ba-opguard-environment-setup/installed-freeze.stdout" "$record/ba-opguard-installed.txt"
"$project/environments/ba-20261107/$SLURM_JOB_ID/bin/python" -B "$checkout/experiments/ba_20261107/opguard_job.py" &
worker_pid=$!
wait "$worker_pid"
worker_pid=
