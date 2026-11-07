"""Qualify an exact Scratch hardlink environment on allocated CPU development.

Admission precedes environment creation or installation. The job retains source,
command logs, installed versions, inode evidence and failures. It never runs a
model, numerical fixture, CUDA experiment or scientific timing comparison.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

from allocated_common import ROOT, admit, retain, sha, write
from setup_preflight.checks import (
    hardlink_proof,
    parse_lock,
    require,
    stat_record,
    verify_versions,
)


def interrupted(signum, frame):
    raise KeyboardInterrupt(
        "Setup processing deadline reached; retain partial evidence"
    )


def run_command(out, label, argv, *, maximum_seconds=None):
    """Bound this job's own process group and keep raw streams even on failure."""
    remaining = float(os.environ["ACI_SCIENCE_DEADLINE_EPOCH"]) - time.time()
    require(remaining >= 5, "Processing cutoff before " + label)
    limit = remaining if maximum_seconds is None else min(remaining, maximum_seconds)
    started = time.monotonic()
    process = None
    error = None
    try:
        with (
            (out / (label + ".stdout")).open("xb") as stdout,
            (out / (label + ".stderr")).open("xb") as stderr,
        ):
            process = subprocess.Popen(
                argv, stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                process.wait(timeout=limit)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                raise
        require(process.returncode == 0, label + " failed; raw output retained")
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        write(
            out / (label + ".command.json"),
            {
                "argv": [str(item) for item in argv],
                "returncode": process.returncode if process is not None else None,
                "elapsed_seconds": time.monotonic() - started,
                "timeout_seconds": limit,
                "error": error,
                "automatic_retry": False,
            },
        )
    return (out / (label + ".stdout")).read_text()


def qualify(checkout, out, contract):
    """Build one fresh non-WORK environment and verify exact versions and links."""
    started = time.monotonic()
    require(
        contract["experiment_id"] == "BA-SETUP-DEV-20261107-v1"
        and contract["partition"] == "development"
        and contract["time_limit"] == "00:30:00"
        and contract["max_su"] == 0.5
        and contract["cpus"] == 8,
        "Registered CPU development resource envelope differs",
    )
    require(tuple(sys.version_info[:3]) == (3, 13, 7), "Base Python must be 3.13.7")
    require(
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and os.environ.get("UV_COMPILE_BYTECODE") == "0"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "Read-only package imports and hidden CUDA environment required",
    )
    lock = checkout / contract["requirements_lock"]
    expected = parse_lock(lock)
    require(
        len(expected) == contract["expected_package_count"] == 58,
        "Exact 58-package lock census differs",
    )
    write(out / "expected-package-versions.json", expected)
    uv = ROOT / "tools/uv-0.12.3/uv"
    base_python = ROOT / "tools/python/cpython-3.13.7-linux-x86_64-gnu/bin/python3.13"
    require(
        Path(sys.executable).resolve() == base_python.resolve(),
        "Base interpreter path differs",
    )
    uv_version = run_command(
        out, "uv-version", [str(uv), "--version"], maximum_seconds=30
    )
    require(uv_version.split()[:2] == ["uv", "0.12.3"], "Pinned uv version differs")
    write(
        out / "bootstrap-runtime.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "uv": str(uv),
            "uv_version": uv_version.strip(),
            "uv_sha256": sha(uv),
            "lock_sha256": sha(lock),
            "scientific_execution": False,
        },
    )
    job = os.environ["SLURM_JOB_ID"]
    environment_parent = ROOT / "environments/ba-20261107"
    environment_parent.mkdir(parents=True, exist_ok=True)
    environment = environment_parent / job
    require(
        not environment.exists(),
        "Exclusive job environment already exists; no retry or mutation",
    )
    cache = ROOT / "cache/uv"
    require(
        Path(os.environ["UV_CACHE_DIR"]) == cache and cache.is_dir(),
        "Registered uv cache differs or is absent",
    )
    parent_device, cache_device = environment_parent.stat().st_dev, cache.stat().st_dev
    write(
        out / "filesystem-before-install.json",
        {
            "environment": str(environment),
            "environment_parent_st_dev": parent_device,
            "cache": str(cache),
            "cache_st_dev": cache_device,
            "same_filesystem": parent_device == cache_device,
            "cache_top_level_names": sorted(path.name for path in cache.iterdir()),
            "shared_files_modified": False,
        },
    )
    require(
        parent_device == cache_device, "Environment and cache must share a filesystem"
    )
    run_command(
        out,
        "create-environment",
        [str(uv), "venv", "--python", str(base_python), str(environment)],
    )
    write(
        out / "filesystem-after-create.json",
        {
            "environment": str(environment),
            "environment_st_dev": environment.stat().st_dev,
            "cache_st_dev": cache.stat().st_dev,
        },
    )
    require(
        environment.stat().st_dev == cache.stat().st_dev,
        "Actual venv and cache filesystems differ",
    )
    python = environment / "bin/python"
    run_command(
        out,
        "install-locked-environment",
        [
            str(uv),
            "pip",
            "sync",
            "--python",
            str(python),
            "--require-hashes",
            "--link-mode",
            "hardlink",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu126",
            "--index-strategy",
            "unsafe-best-match",
            str(lock),
        ],
    )
    run_command(
        out, "installed-freeze", [str(uv), "pip", "freeze", "--python", str(python)]
    )
    metadata_script = """import importlib.metadata as m,json,sys,sysconfig
print(json.dumps({'prefix':sys.prefix,'python':list(sys.version_info[:3]),'purelib':sysconfig.get_path('purelib'),'distributions':[{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()]}))
"""
    metadata = json.loads(
        run_command(
            out,
            "installed-metadata",
            [str(python), "-B", "-c", metadata_script],
            maximum_seconds=60,
        )
    )
    require(
        Path(metadata["prefix"]) == environment and metadata["python"] == [3, 13, 7],
        "Fresh environment prefix or interpreter version differs",
    )
    actual = verify_versions(expected, metadata["distributions"])
    write(out / "installed-package-versions.json", actual)
    library = Path(metadata["purelib"]) / "torch/lib/libtorch_cuda.so"
    require(
        library.is_relative_to(environment),
        "Representative package file leaves new environment",
    )
    write(
        out / "representative-library-before-proof.json",
        stat_record(library, library.stat()),
    )
    proof = hardlink_proof(
        library, cache, float(os.environ["ACI_SCIENCE_DEADLINE_EPOCH"])
    )
    write(out / "hardlink-proof.json", proof)
    before = library.stat()
    import_script = """import json,sys,torch,numpy
print(json.dumps({'python':list(sys.version_info[:3]),'torch':torch.__version__,'numpy':numpy.__version__,'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'visible_cuda_device_count':torch.cuda.device_count(),'torch_file':torch.__file__,'numpy_file':numpy.__file__,'cuda_qualification':False,'model_or_numerical_fixture_run':False}))
"""
    runtime = json.loads(
        run_command(
            out,
            "import-runtime",
            [str(python), "-B", "-c", import_script],
            maximum_seconds=120,
        )
    )
    require(
        runtime["python"] == [3, 13, 7]
        and runtime["torch"] == "2.13.0+cu126"
        and runtime["numpy"] == "2.5.2"
        and runtime["torch_cuda_build"] == "12.6"
        and runtime["cuda_available"] is False
        and runtime["visible_cuda_device_count"] == 0,
        "Pinned CPU-visible imports differ",
    )
    require(
        all(
            Path(runtime[field]).is_relative_to(environment)
            for field in ("torch_file", "numpy_file")
        ),
        "Imported package does not belong to the fresh environment",
    )
    after = library.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "Representative shared library changed during read-only imports",
    )
    write(out / "import-runtime.json", runtime)
    write(out / "representative-library-after-import.json", stat_record(library, after))
    elapsed = time.monotonic() - started
    require(
        elapsed <= contract["processing_budget_seconds"] == 1200,
        "Setup exceeded registered 20-minute processing budget",
    )
    result = {
        "status": "SETUP_QUALIFIED_PENDING_INDEPENDENT_ACTUAL_REVIEW",
        "execution_complete": True,
        "environment_path": str(environment),
        "package_count": len(actual),
        "all_exact_versions_match": True,
        "actual_hardlink_proven": True,
        "setup_elapsed_seconds": elapsed,
        "processing_budget_seconds": 1200,
        "scientific_speedup_estimate": None,
        "cuda_qualification": False,
        "gpu_environment_reuse_released": False,
        "shared_files_modified": False,
        "work_environment_copy": False,
        "scientific_acceptance": False,
    }
    write(out / "results.json", result)
    return result


def main():
    checkout = Path(__file__).resolve().parents[3]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        sys.platform == "linux" and job.isdecimal(),
        "Owned allocated Linux CPU job required",
    )
    out = ROOT / "artifacts" / job / "ba-setup"
    out.mkdir(parents=True, exist_ok=False)
    for signum in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, interrupted)
    contract, exit_code = None, 1
    try:
        path = Path(os.environ["BA_CONTRACT_PATH"])
        # Admission is deliberately before uv invocation, venv creation or imports.
        contract = admit(checkout, path, out)
        write(out / "contract.json", contract)
        write(
            out / "execution-release.json",
            json.loads(Path(os.environ["BA_RELEASE_PATH"]).read_text()),
        )
        with tarfile.open(
            out / "exact-setup-source-and-config.tar.gz", "x:gz"
        ) as archive:
            for name in sorted(contract["files"]):
                archive.add(checkout / name, arcname=name, recursive=False)
        qualify(checkout, out, contract)
        exit_code = 0
    except BaseException as error:
        write(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "scientific_acceptance": False,
                "automatic_retry": False,
                "partial_environment_preserved": True,
            },
        )
    finally:
        signal.alarm(0)
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        write(
            out / "terminal.json",
            {
                "job_id": job,
                "exit_code": exit_code,
                "scientific_acceptance": False,
                "automatic_retry": False,
            },
        )
        if contract is not None:
            retain(out, contract["campaign_id"], contract["output_max_bytes"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
