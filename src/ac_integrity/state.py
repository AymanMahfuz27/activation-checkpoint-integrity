"""Atomic snapshots and explicit reproducibility metadata for trusted local runs."""

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import zipfile
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def backend_state():
    return {"deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def configure(config):
    if torch.__version__ != config.expected_torch:
        raise RuntimeError(f"Expected PyTorch {config.expected_torch}, got {torch.__version__}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)


def environment():
    repo = Path(__file__).resolve().parents[2]
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=repo, text=True,
                              capture_output=True, check=True).stdout.strip()
    code_files = {}
    for folder in ("src", "configs", "scripts"):
        for path in sorted((repo / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                code_files[str(path.relative_to(repo))] = sha256(path)
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "code_files": code_files,
            "code_sha256": hashlib.sha256(json.dumps(code_files, sort_keys=True).encode()).hexdigest(),
            "lock_sha256": sha256(repo / "uv.lock"),
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "cuda_runtime": torch.version.cuda,
            "gpus": [torch.cuda.get_device_properties(i).__repr__() for i in range(torch.cuda.device_count())],
            "command": sys.argv, "command_sha256": hashlib.sha256(json.dumps(sys.argv).encode()).hexdigest(),
            "backend": backend_state()}


def archive_source(destination, record):
    """Retain executable source even when a locally tested tree is uncommitted."""
    repo = Path(__file__).resolve().parents[2]
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, digest in record["code_files"].items():
            path = repo / relative
            if sha256(path) != digest:
                raise RuntimeError("Source changed while recording the run")
            archive.write(path, relative)
        for relative in ("pyproject.toml", "uv.lock"):
            archive.write(repo / relative, relative)


def rng_state(generators=None):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "custom": {name: generator.get_state() for name, generator in (generators or {}).items()}}


def restore_rng(state, generators=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])
    for name, value in state["custom"].items():
        if name not in (generators or {}):
            raise ValueError(f"Unregistered custom generator: {name}")
        generators[name].set_state(value)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_snapshot(path, model, optimizer, scheduler, cursor, step, batches, provenance,
                  trigger_state=None, generators=None):
    """Persist pre-step state once; the digest sidecar commits the snapshot."""
    path = Path(path)
    if path.exists() or path.with_suffix(".sha256").exists():
        raise FileExistsError(f"Immutable snapshot already exists: {path}")
    state = {"schema_version": 1, "model": cpu_tree(model.state_dict()),
             "optimizer": cpu_tree(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
             "gradients": {n: None if p.grad is None else p.grad.detach().cpu().clone()
                           for n, p in model.named_parameters()},
             "rng": rng_state(generators), "cursor": cursor, "step": step,
             "batches": cpu_tree(batches), "backend": backend_state(),
             "trigger": trigger_state or {}, "provenance": provenance,
             "scaler": None, "topology": {"world_size": 1, "rank": 0}}
    save_tensor_file(path, state)
    atomic_write(path.with_suffix(".sha256"), (sha256(path) + "\n").encode())


def save_tensor_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_snapshot(path, model=None, optimizer=None, scheduler=None):
    path = Path(path)
    if sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Snapshot checksum mismatch")
    # Snapshots contain Python and NumPy RNG states: load only trusted project artifacts.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schema_version"] != 1 or state["topology"]["world_size"] != 1:
        raise ValueError("Unsupported snapshot schema/topology")
    if model is not None:
        if state["backend"] != backend_state():
            raise ValueError("Snapshot precision/backend state differs from the configured run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for name, parameter in model.named_parameters():
            gradient = state["gradients"][name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
        restore_rng(state["rng"])
    return state
