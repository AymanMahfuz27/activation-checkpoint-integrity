#!/usr/bin/env bash
set -euo pipefail

# Input: one Condor GPU allocation and the project-local pinned environment.
# Output: E0-E2 FP32 artifacts plus scheduler, code, environment, and GPU facts.
# Sequence: fail-fast environment checks, tests, three E0 runs, all repeated E1
# arms, then E2 with exit 1 accepted only for PUBLIC_HOOKS_INSUFFICIENT.

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
artifact_root="$repo_dir/artifacts/starter"
condor_artifacts="$repo_dir/artifacts/condor"
requirements="$repo_dir/condor/requirements-cu126.lock"
lock_hash=$(sha256sum "$requirements" | awk '{print $1}')
environment_dir="$repo_dir/.condor-venv-${lock_hash:0:16}"
python_bin="$environment_dir/bin/python"
uv_bin="$repo_dir/.condor-tools/uv"

cd "$repo_dir"
test -x "$python_bin"
test -x "$uv_bin"
mkdir -p "$artifact_root" "$condor_artifacts"
export PYTHONPATH="$repo_dir/src"
export PYTHONUNBUFFERED=1

printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'hostname=%s\n' "$(hostname -f)"
printf 'condor_cluster_id=%s\n' "${CONDOR_CLUSTER_ID:-unknown}"
printf 'condor_process_id=%s\n' "${CONDOR_PROCESS_ID:-unknown}"
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
if [[ -n "$(git status --short)" ]]; then
    printf 'ERROR: scheduled run requires a clean worktree\n' >&2
    git status --short >&2
    exit 2
fi

nvidia-smi
"$uv_bin" pip freeze --python "$python_bin"
"$python_bin" - <<'PY'
import json
import platform
import subprocess
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
device = torch.device("cuda")
properties = torch.cuda.get_device_properties(device)
record = {
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "driver": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip(),
    "gpu_name": properties.name,
    "gpu_memory_bytes": properties.total_memory,
    "compute_capability": list(torch.cuda.get_device_capability(device)),
    "cudnn": torch.backends.cudnn.version(),
}
print(json.dumps(record, sort_keys=True))
if not str(torch.__version__).startswith("2.12.1+cu126"):
    raise SystemExit(f"unexpected PyTorch build: {torch.__version__}")
if torch.version.cuda != "12.6":
    raise SystemExit(f"unexpected CUDA runtime: {torch.version.cuda}")
if tuple(record["compute_capability"]) != (6, 1):
    raise SystemExit(f"expected Pascal compute capability 6.1: {record['compute_capability']}")
PY

"$python_bin" -m pytest -q
for repetition in 1 2 3; do
    printf 'e0_repetition=%s\n' "$repetition"
    "$python_bin" -m ac_integrity.cli run E0 \
        --device cuda --dtype float32 --artifact-root "$artifact_root"
done

"$python_bin" -m ac_integrity.cli run E1 --case all \
    --device cuda --dtype float32 --artifact-root "$artifact_root"

set +e
e2_output=$("$python_bin" -m ac_integrity.cli run E2 \
    --device cuda --dtype float32 --artifact-root "$artifact_root" 2>&1)
e2_status=$?
set -e
printf '%s\n' "$e2_output"
if [[ "$e2_status" -eq 2 ]]; then
    exit 2
fi
if [[ "$e2_status" -eq 1 ]] && [[ "$e2_output" != *'"result": "PUBLIC_HOOKS_INSUFFICIENT"'* ]]; then
    exit 1
fi
if [[ "$e2_status" -ne 0 ]] && [[ "$e2_status" -ne 1 ]]; then
    exit "$e2_status"
fi

printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
