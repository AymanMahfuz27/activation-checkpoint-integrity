#!/usr/bin/env bash
set -euo pipefail

# Input: the synchronized repository on the Condor submit host.
# Output: a content-addressed, repository-local Python environment.
# Sequence: verify the host, install a checksum-verified local uv binary, then
# sync the complete hash-locked CUDA 12.6 dependency graph.

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
uv_version=0.12.3
uv_archive=uv-x86_64-unknown-linux-gnu.tar.gz
uv_sha256=600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101
uv_dir="$repo_dir/.condor-tools"
uv_bin="$uv_dir/uv"
requirements="$repo_dir/condor/requirements-cu126.lock"
python_bin=/u/ayman27/miniconda3/bin/python3

cd "$repo_dir"
test -x "$python_bin"
test -f "$requirements"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'

if [[ ! -x "$uv_bin" ]] || [[ "$($uv_bin --version)" != "uv $uv_version"* ]]; then
    temporary_dir=$(mktemp -d)
    trap 'rm -rf "$temporary_dir"' EXIT
    curl -fsSL \
        "https://github.com/astral-sh/uv/releases/download/$uv_version/$uv_archive" \
        -o "$temporary_dir/$uv_archive"
    printf '%s  %s\n' "$uv_sha256" "$temporary_dir/$uv_archive" | sha256sum --check --status
    tar -xzf "$temporary_dir/$uv_archive" -C "$temporary_dir"
    mkdir -p "$uv_dir"
    install -m 0755 "$temporary_dir/uv-x86_64-unknown-linux-gnu/uv" "$uv_bin"
fi

lock_hash=$(sha256sum "$requirements" | awk '{print $1}')
environment_dir="$repo_dir/.condor-venv-${lock_hash:0:16}"
if [[ ! -x "$environment_dir/bin/python" ]]; then
    "$uv_bin" venv --python "$python_bin" "$environment_dir"
fi

"$uv_bin" pip sync \
    --python "$environment_dir/bin/python" \
    --python-platform x86_64-manylinux_2_28 \
    --require-hashes \
    --index https://download.pytorch.org/whl/cu126 \
    --index https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    "$requirements"

mkdir -p "$repo_dir/artifacts/condor"
printf '%s\n' "$environment_dir" > "$repo_dir/artifacts/condor/environment_path.txt"
"$uv_bin" pip freeze --python "$environment_dir/bin/python" \
    > "$repo_dir/artifacts/condor/environment-freeze.txt"

printf 'environment=%s\n' "$environment_dir"
printf 'lock_sha256=%s\n' "$lock_hash"
