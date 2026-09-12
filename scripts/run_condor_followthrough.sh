#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
profile=${1:?Pass core or upstream}
[[ "$profile" == core || "$profile" == upstream ]]
[[ -n "${_CONDOR_SCRATCH_DIR:-}" ]]
[[ -z "$(git -C "$repo_dir" status --porcelain)" ]]
job_id="${CONDOR_CLUSTER_ID:?}.${CONDOR_PROCESS_ID:?}"
record_dir="$repo_dir/artifacts/followthrough/$job_id"
mkdir -p "$record_dir"

# Freeze code before execution: later laptop/remote commits cannot change a run.
source_dir="$_CONDOR_SCRATCH_DIR/source"
git clone --quiet --no-hardlinks "$repo_dir" "$source_dir"
git -C "$source_dir" checkout --quiet --detach "$(git -C "$repo_dir" rev-parse HEAD)"
export ACI_REQUIREMENTS_FILE="$source_dir/condor/production-cu126.lock"
if [[ "$profile" == upstream ]]; then
    export ACI_REQUIREMENTS_FILE="$source_dir/condor/upstream-cu126.lock"
fi
"$repo_dir/scripts/bootstrap_condor_env.sh" "$_CONDOR_SCRATCH_DIR"
lock_hash=$(sha256sum "$ACI_REQUIREMENTS_FILE" | awk '{print $1}')
python_bin="$_CONDOR_SCRATCH_DIR/aci-condor-venv-${lock_hash:0:16}/bin/python"
export PYTHONPATH="$source_dir/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$_CONDOR_SCRATCH_DIR/inductor-cache"
export ACI_SOURCE_DATA="$repo_dir/data/fineweb-edu-100m"
export ACI_JOB_RECORD="$record_dir"
export ACI_DEPENDENCY_LOCK_SHA256="$lock_hash"
cd "$source_dir"
mkdir -p artifacts
nvidia-smi > "$record_dir/nvidia-smi.txt"
"$python_bin" - "$profile" <<'PY'
import json, os, pathlib, shutil, sys
import torch
from ac_integrity.config import load_config, write_config
from ac_integrity.state import environment, write_json

assert torch.__version__ == "2.13.0+cu126", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert torch.cuda.is_available()
x = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
assert torch.equal(x @ x, torch.tensor([[7., 10.], [15., 22.]], device="cuda"))
torch.cuda.synchronize()
c = load_config("configs/lm40m.toml")
c.threads = 4
c.data.path = os.environ["ACI_SOURCE_DATA"]
if sys.argv[1] == "upstream":
    c.training.sequence_length = 64
    c.training.microbatch_size = 1
    c.training.accumulation = 1
write_config(c, "artifacts/job-config.toml")
record = {"environment": environment(), "profile": sys.argv[1],
          "scratch": os.environ["_CONDOR_SCRATCH_DIR"],
          "scratch_free_bytes": shutil.disk_usage(os.environ["_CONDOR_SCRATCH_DIR"]).free,
          "source": str(pathlib.Path.cwd()), "cuda_capability": torch.cuda.get_device_capability(),
          "dependency_lock_sha256": os.environ["ACI_DEPENDENCY_LOCK_SHA256"]}
write_json(pathlib.Path(os.environ["ACI_JOB_RECORD"]) / "location.json", record)
required = 100_000_000_000 if sys.argv[1] == "core" else 40_000_000_000
assert record["scratch_free_bytes"] >= required, record["scratch_free_bytes"]
print(json.dumps(record), flush=True)
PY

run_root="$_CONDOR_SCRATCH_DIR/evidence"
exit_code=0
if [[ "$profile" == core ]]; then
    "$python_bin" -m ac_integrity.validation suite --config artifacts/job-config.toml \
        --output "$run_root" --budget-bytes 80000000000 || exit_code=$?
else
    "$python_bin" -m ac_integrity.upstream matrix --config artifacts/job-config.toml \
        --output "$run_root" --steps 1 --record \
        --cells eager_fp32,sdpa_fp32,dropout_fp32,eager_fp16,compile_fp32,eager_bf16 \
        --backend inductor || exit_code=$?
fi
printf '%s\n' "$exit_code" > "$record_dir/experiment-exit-code"
if [[ -f "$run_root/summary.json" ]]; then
    cp "$run_root/summary.json" "$record_dir/summary.json"
fi
if [[ -d "$run_root" ]]; then
    cp "$record_dir/location.json" "$run_root/scheduler.json"
    cp "$record_dir/nvidia-smi.txt" "$run_root/nvidia-smi.txt"
    printf 'Awaiting verified artifact collection in %s/transfer\n' "$record_dir"
    "$python_bin" -m ac_integrity.archive_transfer send --root "$run_root" --inbox "$record_dir/transfer"
fi
printf 'Completed and archived job %s; scientific exit %s\n' "$job_id" "$exit_code"
exit "$exit_code"
