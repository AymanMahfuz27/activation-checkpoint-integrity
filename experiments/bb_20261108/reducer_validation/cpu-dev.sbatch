#!/usr/bin/env bash
#SBATCH --job-name=vg-bb-reducer-validation-cpu-dev
#SBATCH --account=CCR26063
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/bb-20261108/reducer-validation-cpu-dev-%j.out
#SBATCH --error=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/bb-20261108/reducer-validation-cpu-dev-%j.err
set -euo pipefail
project=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity
checkout="$project/slurm/bb-20261108/reducer-validation-cpu-dev-source"
python="$project/tools/python/cpython-3.13.7-linux-x86_64-gnu/bin/python3.13"
record="$project/artifacts/$SLURM_JOB_ID"
mkdir -p "$record"
exec > >(tee -a "$record/bb-reducer-validation-runner.log") 2>&1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MAX_JOBS=4
export PYTHONDONTWRITEBYTECODE=1 UV_COMPILE_BYTECODE=0
export CUDA_VISIBLE_DEVICES= BB_TEST_DEVICE=cpu
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$project/cache/uv" UV_PYTHON_INSTALL_DIR="$project/tools/python"
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
  printf '%s\n' "$status" > "$record/bb-reducer-validation-exit-code.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 124' USR1 TERM INT
module load gcc/13.2.0 cuda/12.8
module -t list 2> "$record/bb-reducer-validation-modules.txt"
"$python" -B - "$checkout" <<'PREFLIGHT'
import hashlib,json,os,sys
from pathlib import Path
checkout=Path(sys.argv[1]);path=Path(os.environ['BA_CONTRACT_PATH'])
expected=checkout/'experiments/bb_20261108/reducer_validation/contract.json'
if path!=expected:raise RuntimeError('Exact reducer fixture contract required')
if hashlib.sha256(path.read_bytes()).hexdigest()!=os.environ['BA_CONTRACT_SHA256']:raise RuntimeError('Contract hash differs')
contract=json.loads(path.read_text())
if contract.get('draft') is not False:raise RuntimeError('Draft validation contract cannot install or execute')
PREFLIGHT
"$python" -B "$checkout/experiments/bb_20261108/copy_runtime.py" --study e1 &
worker_pid=$!
wait "$worker_pid"
worker_pid=
cp "$record/bb-e1-environment-setup/installed-freeze.stdout" "$record/bb-reducer-validation-installed.txt"
"/tmp/aci-bb-copy-$SLURM_JOB_ID/env/bin/python" -B "$checkout/experiments/bb_20261108/reducer_validation/runner.py" &
worker_pid=$!
wait "$worker_pid"
worker_pid=
