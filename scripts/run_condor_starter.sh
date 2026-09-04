#!/usr/bin/env bash
set -euo pipefail

# Input: one Condor GPU allocation, per-job scratch, and the pinned lock file.
# Output: E0-E2 FP32 artifacts plus scheduler, code, environment, and GPU facts.
# Sequence: build the environment in scratch; fail fast on artifact storage and
# a real CUDA kernel; run tests, three E0 runs, all repeated E1 arms, then E2
# with exit 1 accepted only for PUBLIC_HOOKS_INSUFFICIENT.

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
artifact_root="$repo_dir/artifacts/starter"
requirements="$repo_dir/condor/requirements-cu126.lock"
lock_hash=$(sha256sum "$requirements" | awk '{print $1}')
if [[ -z "${_CONDOR_SCRATCH_DIR:-}" ]] || [[ ! -d "$_CONDOR_SCRATCH_DIR" ]]; then
    printf 'ERROR: _CONDOR_SCRATCH_DIR is unavailable\n' >&2
    exit 2
fi
environment_dir="$_CONDOR_SCRATCH_DIR/aci-condor-venv-${lock_hash:0:16}"
python_bin="$environment_dir/bin/python"
uv_bin="$repo_dir/.condor-tools/uv"

cd "$repo_dir"
if ! "$repo_dir/scripts/bootstrap_condor_env.sh" "$_CONDOR_SCRATCH_DIR"; then
    printf 'ERROR: scratch environment bootstrap failed\n' >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]] || [[ ! -x "$uv_bin" ]]; then
    printf 'ERROR: pinned Python environment or project-local uv is unavailable\n' >&2
    exit 2
fi
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

ACI_ARTIFACT_ROOT="$artifact_root" "$python_bin" - <<'PY'
import json
import os
import uuid
from pathlib import Path

root = Path(os.environ["ACI_ARTIFACT_ROOT"])
probe = root / f".write-preflight-{uuid.uuid4().hex}"
payload = b"activation-checkpoint-integrity-artifact-preflight\n"
try:
    root.mkdir(parents=True, exist_ok=True)
    with probe.open("xb") as handle:
        bytes_written = handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if bytes_written != len(payload):
        raise OSError(f"artifact preflight short write: {bytes_written}/{len(payload)}")
    observed = probe.read_bytes()
    if observed != payload:
        raise OSError("artifact preflight read-back mismatch")
    probe.unlink()
except OSError as error:
    try:
        probe.unlink(missing_ok=True)
    except OSError:
        pass
    print(
        json.dumps(
            {
                "artifact_preflight": "BLOCKED",
                "artifact_root": str(root),
                "error": f"{type(error).__name__}: {error}",
            },
            sort_keys=True,
        )
    )
    raise SystemExit(2)
print(
    json.dumps(
        {
            "artifact_preflight": "PASS",
            "artifact_root": str(root),
            "bytes_written_read_removed": len(payload),
        },
        sort_keys=True,
    )
)
PY

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

left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
right = torch.tensor([[2.0, 0.5], [-1.0, 3.0]], device=device)
observed = left @ right + torch.tensor([0.25, -0.5], device=device)
expected = torch.tensor([[0.25, 6.0], [2.25, 13.0]], device=device)
torch.cuda.synchronize(device)
if observed.device.type != "cuda" or not torch.equal(observed, expected):
    raise SystemExit(
        f"CUDA tensor-kernel preflight mismatch: device={observed.device}, "
        f"observed={observed.cpu().tolist()}"
    )
print(
    json.dumps(
        {
            "cuda_tensor_kernel_preflight": "PASS",
            "operation": "2x2 FP32 matmul plus bias",
            "device": str(observed.device),
            "result": observed.cpu().tolist(),
        },
        sort_keys=True,
    )
)
PY

"$python_bin" -m pytest -q -p no:cacheprovider
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
