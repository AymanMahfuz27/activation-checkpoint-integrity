#!/usr/bin/env bash
set -euo pipefail

# Execute only inside a scheduled GPU allocation. Preserve the immutable source
# revision, scratch environment lock and caller-selected experiment command.
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
if [[ -z "${_CONDOR_SCRATCH_DIR:-}" ]]; then
    printf 'A scheduled Condor scratch allocation is required.\n' >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Scheduled experiments require a clean synchronized worktree.\n' >&2
    exit 2
fi
export ACI_REQUIREMENTS_FILE="$repo_dir/condor/production-cu126.lock"
"$repo_dir/scripts/bootstrap_condor_env.sh" "$_CONDOR_SCRATCH_DIR"
lock_hash=$(sha256sum "$ACI_REQUIREMENTS_FILE" | awk '{print $1}')
python_bin="$_CONDOR_SCRATCH_DIR/aci-condor-venv-${lock_hash:0:16}/bin/python"
export PYTHONPATH="$repo_dir/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'host=%s\n' "$(hostname -f)"
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'lock_sha256=%s\n' "$lock_hash"
nvidia-smi
"$python_bin" - <<'PY'
import torch
from ac_integrity.state import environment
import json
assert torch.__version__ == "2.13.0+cu126", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert torch.cuda.get_device_capability() == (6, 1)
x = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
assert torch.equal(x @ x, torch.tensor([[7., 10.], [15., 22.]], device="cuda"))
torch.cuda.synchronize()
print(json.dumps(environment(), indent=2))
PY
if [[ "${1:-probe}" == "probe" ]]; then
    "$python_bin" "$repo_dir/scripts/probe_model_fit.py" --config "${2:-configs/lm125m.toml}"
else
    "$python_bin" -m ac_integrity.production_cli "$@"
fi
printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
