#!/usr/bin/env bash
#SBATCH --job-name=vg-bb-timing-v2-dev
#SBATCH --account=CCR26063
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:40:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/bb-20261108/timing-v2-dev-%j.out
#SBATCH --error=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/bb-20261108/timing-v2-dev-%j.err
set -euo pipefail
project=/scratch/09999/aymanmahfuz/activation-checkpoint-integrity
python="$project/tools/python/cpython-3.13.7-linux-x86_64-gnu/bin/python3.13"
record="$project/artifacts/$SLURM_JOB_ID"
mkdir -p "$record"
exec > >(tee -a "$record/bb-selector-runner.log") 2>&1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MAX_JOBS=4
export PYTHONDONTWRITEBYTECODE=1 UV_COMPILE_BYTECODE=0
export CUDA_VISIBLE_DEVICES=0,1,2 BB_TEST_DEVICE=cuda:0
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
  printf '%s\n' "$status" > "$record/bb-selector-exit-code.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 124' USR1 TERM INT
case "${BA_CONTRACT_PATH:-}" in
  "$project/slurm/bb-20261108/timing-v2-0-dev-source/experiments/bb_20261108/timing-v2-0-dev-contract.json")
    checkout="$project/slurm/bb-20261108/timing-v2-0-dev-source" ;;
  "$project/slurm/bb-20261108/timing-v2-1-dev-source/experiments/bb_20261108/timing-v2-1-dev-contract.json")
    checkout="$project/slurm/bb-20261108/timing-v2-1-dev-source" ;;
  "$project/slurm/bb-20261108/timing-v2-2-dev-source/experiments/bb_20261108/timing-v2-2-dev-contract.json")
    checkout="$project/slurm/bb-20261108/timing-v2-2-dev-source" ;;
  *)
    printf '%s\n' 'Exact sealed timing contract/index path required' >&2
    exit 65 ;;
esac
"$python" -B - "$BA_CONTRACT_PATH" "$checkout" <<'PREFLIGHT'
import hashlib,json,os,re,sys
from pathlib import Path
path,checkout=map(Path,sys.argv[1:])
if path.is_symlink() or not path.is_file():
    raise RuntimeError('Regular sealed timing contract required')
raw=path.read_bytes()
if hashlib.sha256(raw).hexdigest()!=os.environ['BA_CONTRACT_SHA256']:
    raise RuntimeError('Sealed timing contract SHA differs')
contract=json.loads(raw)
match=re.fullmatch(r'timing-v2-([012])-dev-source',checkout.name)
if match is None:
    raise RuntimeError('Exact v2 timing checkout required')
index=int(match[1])
if contract.get('draft') is not False or type(contract.get('timing_job_index')) is not int or contract['timing_job_index']!=index:
    raise RuntimeError('Draft or mismatched timing template cannot install/run')
PREFLIGHT
module load gcc/13.2.0 cuda/12.8
module -t list 2> "$record/bb-selector-modules.txt"
"$python" -B "$checkout/experiments/bb_20261108/copy_runtime.py" --study selector &
worker_pid=$!
wait "$worker_pid"
worker_pid=
cp "$record/bb-selector-environment-setup/installed-freeze.stdout" "$record/bb-selector-installed.txt"
"/tmp/aci-bb-copy-$SLURM_JOB_ID/env/bin/python" -B "$checkout/experiments/bb_20261108/timing_job.py" &
worker_pid=$!
wait "$worker_pid"
worker_pid=
