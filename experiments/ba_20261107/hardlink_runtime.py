"""Qualify fresh Scratch hardlink environments without numerical imports.

The bootstrap has its own source/resource/storage admission and compact custody.
The scientific wrapper must independently admit again after installation, then
call qualify() before importing Torch or invoking any scientific fixture.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tarfile
import time
import traceback

from allocated_common import ROOT, admit, command, retain, sha, write
from setup_preflight.checks import hardlink_proof, parse_lock, require, verify_versions


def paths(study):
    require(study in ("provenance", "placement", "opguard"), "Unknown hardlink study")
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        sys.platform == "linux" and job.isdecimal(), "Real Linux Slurm job required"
    )
    environment = ROOT / "environments/ba-20261107" / job
    cache = ROOT / "cache/uv"
    bootstrap = ROOT / "artifacts" / job / ("ba-" + study + "-environment-setup")
    return environment, cache, bootstrap


def prerequisites(contract):
    release = json.loads(Path(os.environ["BA_RELEASE_PATH"]).read_text())
    actual = release.get("prerequisite_reviews", {})
    required = contract["required_actual_reviews"]
    require(
        isinstance(actual, dict) and set(actual) == set(required),
        "Exact conditional actual-review census missing from release",
    )
    for name in required:
        item = actual[name]
        require(
            isinstance(item, dict)
            and item.get("verdict") == "PROMOTE"
            and isinstance(item.get("job_id"), str)
            and item["job_id"].isdecimal()
            and isinstance(item.get("review_sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", item["review_sha256"]) is not None,
            "Conditional actual review is absent or unsupported: " + name,
        )
    return actual


def no_numerical_imports():
    require(
        not any(
            name.split(".")[0] in ("torch", "numpy", "transformers")
            for name in sys.modules
        ),
        "Environment qualification must precede numerical imports",
    )


def expected_versions(checkout, contract):
    require(
        contract.get("environment_installation") == "FRESH_SCRATCH_HARDLINK_V1",
        "Unregistered hardlink environment contract",
    )
    require(
        contract["bootstrap_output_max_bytes"] == 100_000_000
        and contract["bootstrap_custody_campaign"]
        == contract["campaign_id"] + "-environment-setup",
        "Registered compact setup custody envelope differs",
    )
    expected = parse_lock(checkout / contract["requirements_lock"])
    require(
        len(expected) == contract["expected_package_count"]
        and len(expected) in (58, 65),
        "Exact lock package census differs",
    )
    return expected


def run_command(out, label, argv, deadline):
    """Bound one owned subprocess group and retain exact raw streams on failure."""
    limit = deadline - time.time()
    require(limit >= 5, "Environment setup cutoff before " + label)
    process, error = None, None
    started = time.monotonic()
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
                "elapsed_seconds": time.monotonic() - started,
                "timeout_seconds": limit,
                "returncode": process.returncode if process else None,
                "error": error,
                "automatic_retry": False,
            },
        )
    return (out / (label + ".stdout")).read_text()


def interrupted(signum, frame):
    raise KeyboardInterrupt("Hardlink setup deadline reached; retain partial evidence")


def bootstrap(checkout, study):
    """Install once, leaving at least 1260 seconds for fresh scientific admission."""
    environment, cache, out = paths(study)
    out.mkdir(parents=True, exist_ok=False)
    for signum in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, interrupted)
    contract, status = None, 1
    try:
        contract = admit(checkout, Path(os.environ["BA_CONTRACT_PATH"]), out)
        write(out / "contract.json", contract)
        write(
            out / "execution-release.json",
            json.loads(Path(os.environ["BA_RELEASE_PATH"]).read_text()),
        )
        write(out / "conditional-prerequisite-reviews.json", prerequisites(contract))
        with tarfile.open(
            out / "exact-environment-source-and-config.tar.gz", "x:gz"
        ) as archive:
            for name in sorted(contract["files"]):
                archive.add(checkout / name, arcname=name, recursive=False)
        no_numerical_imports()
        expected = expected_versions(checkout, contract)
        require(
            contract["hardlink_study"] == study, "Bootstrap study differs from contract"
        )
        require(sys.version_info[:3] == (3, 13, 7), "Base Python must be 3.13.7")
        base = ROOT / "tools/python/cpython-3.13.7-linux-x86_64-gnu/bin/python3.13"
        require(
            Path(sys.executable).resolve() == base.resolve(), "Base interpreter differs"
        )
        queue = command(
            out, "bootstrap-owned-queue", ["squeue", "--me", "-h", "-o", "%i"]
        )
        require(
            queue.split() == [os.environ["SLURM_JOB_ID"]],
            "Other owned work is unreconciled",
        )
        # common.admit set the live deadline to remaining-minus-600. Keep another
        # 660 seconds unused so the post-install gate still has 1200 plus margin.
        cutoff = float(os.environ["ACI_SCIENCE_DEADLINE_EPOCH"]) - 660
        require(cutoff - time.time() >= 5, "Insufficient fresh setup window")
        signal.alarm(max(1, int(cutoff - time.time())))
        parent = environment.parent
        parent.mkdir(parents=True, exist_ok=True)
        require(
            not environment.exists() and not environment.is_symlink(),
            "Exclusive job environment exists; no retry or mutation",
        )
        require(
            not parent.is_symlink() and parent.resolve() == parent,
            "Environment parent must be its canonical Scratch path",
        )
        require(
            cache.is_dir()
            and cache.resolve() == cache
            and os.environ.get("UV_CACHE_DIR") == str(cache),
            "Registered cache differs",
        )
        write(
            out / "filesystem-before-install.json",
            {
                "environment": str(environment),
                "cache": str(cache),
                "environment_parent_st_dev": parent.stat().st_dev,
                "cache_st_dev": cache.stat().st_dev,
                "same_filesystem": parent.stat().st_dev == cache.stat().st_dev,
                "shared_files_modified": False,
            },
        )
        require(
            parent.stat().st_dev == cache.stat().st_dev,
            "Cache and environment filesystem differ",
        )
        uv = ROOT / "tools/uv-0.12.3/uv"
        version = run_command(
            out, "uv-version", [str(uv), "--version"], min(cutoff, time.time() + 30)
        )
        require(version.split()[:2] == ["uv", "0.12.3"], "Pinned uv differs")
        write(
            out / "bootstrap-runtime.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "uv_sha256": sha(uv),
                "lock_sha256": sha(checkout / contract["requirements_lock"]),
                "expected_package_count": len(expected),
                "scientific_execution": False,
                "setup_cutoff_epoch": cutoff,
                "minimum_postsetup_admission_seconds": 1200,
                "postsetup_control_margin_seconds": 60,
            },
        )
        run_command(
            out,
            "create-environment",
            [str(uv), "venv", "--python", str(base), str(environment)],
            cutoff,
        )
        require(
            environment.stat().st_dev == cache.stat().st_dev,
            "Created environment filesystem differs",
        )
        run_command(
            out,
            "install-locked-environment",
            [
                str(uv),
                "pip",
                "sync",
                "--python",
                str(environment / "bin/python"),
                "--require-hashes",
                "--link-mode",
                "hardlink",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
                "--index-strategy",
                "unsafe-best-match",
                str(checkout / contract["requirements_lock"]),
            ],
            cutoff,
        )
        run_command(
            out,
            "installed-freeze",
            [str(uv), "pip", "freeze", "--python", str(environment / "bin/python")],
            cutoff,
        )
        write(
            out / "results.json",
            {
                "status": "INSTALLED_PENDING_FRESH_ADMISSION_AND_METADATA_QUALIFICATION",
                "scientific_acceptance": False,
                "environment_path": str(environment),
                "work_environment_copy": False,
                "shared_files_modified": False,
            },
        )
        status = 0
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
                "exit_code": status,
                "scientific_acceptance": False,
                "scope": "ENVIRONMENT_SETUP_ONLY",
            },
        )
        if contract is not None:
            retain(
                out,
                contract["bootstrap_custody_campaign"],
                contract["bootstrap_output_max_bytes"],
            )
    return status


def qualify(checkout, out, contract):
    """Verify the actual executing environment before Torch/CUDA imports."""
    no_numerical_imports()
    study = contract["hardlink_study"]
    environment, cache, setup = paths(study)
    proof_out = out / "environment-qualification"
    proof_out.mkdir(exist_ok=False)
    write(proof_out / "conditional-prerequisite-reviews.json", prerequisites(contract))
    require(
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and os.environ.get("UV_COMPILE_BYTECODE") == "0",
        "Read-only package flags differ",
    )
    require(
        Path(sys.prefix) == environment
        and Path(sys.prefix).resolve() == environment
        and Path(sys.executable) == environment / "bin/python"
        and sys.version_info[:3] == (3, 13, 7),
        "Executing isolated interpreter differs",
    )
    expected = expected_versions(checkout, contract)
    metadata = [
        {"name": item.metadata["Name"], "version": item.version}
        for item in importlib.metadata.distributions()
    ]
    write(
        proof_out / "installed-metadata.json",
        {
            "prefix": sys.prefix,
            "executable": sys.executable,
            "python": list(sys.version_info[:3]),
            "distributions": metadata,
        },
    )
    actual = verify_versions(expected, metadata)
    write(proof_out / "installed-package-versions.json", actual)
    require(
        cache.resolve() == cache
        and os.environ.get("UV_CACHE_DIR") == str(cache)
        and environment.stat().st_dev == cache.stat().st_dev,
        "Actual environment/cache filesystem differs",
    )
    write(
        proof_out / "filesystem.json",
        {
            "environment": str(environment),
            "environment_st_dev": environment.stat().st_dev,
            "cache": str(cache),
            "cache_st_dev": cache.stat().st_dev,
            "same_filesystem": True,
            "shared_files_modified": False,
        },
    )
    library = Path(sysconfig.get_path("purelib")) / "torch/lib/libtorch_cuda.so"
    require(
        library.is_relative_to(environment)
        and library.resolve().is_relative_to(environment),
        "Representative library leaves isolated environment",
    )
    proof = hardlink_proof(
        library, cache, float(os.environ["ACI_SCIENCE_DEADLINE_EPOCH"])
    )
    write(proof_out / "hardlink-proof.json", proof)
    # Copy compact installation evidence only. Never walk the environment/cache.
    require(
        setup.is_dir() and not setup.is_symlink(), "Original setup evidence missing"
    )
    terminal = json.loads((setup / "terminal.json").read_text())
    require(terminal["exit_code"] == 0, "Environment setup did not complete")
    for path in setup.rglob("*"):
        require(not path.is_symlink(), "Setup evidence symlink forbidden")
        if path.is_file():
            destination = proof_out / "setup" / path.relative_to(setup)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            require(sha(path) == sha(destination), "Copied setup evidence differs")
    result = {
        "status": "EXACT_ENVIRONMENT_AND_REPRESENTATIVE_HARDLINK_QUALIFIED",
        "environment_path": str(environment),
        "exact_package_count": len(actual),
        "torch_imported": False,
        "cuda_qualification": False,
        "scientific_acceptance": False,
        "representative_hardlink_only": True,
        "shared_files_modified": False,
        "work_environment_copy": False,
    }
    write(proof_out / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study", required=True, choices=("provenance", "placement", "opguard")
    )
    args = parser.parse_args()
    raise SystemExit(bootstrap(Path(__file__).resolve().parents[2], args.study))
