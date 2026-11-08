"""Fresh CPU admission and custody for fixture-only reducer validation.

No numerical library loads at import. The immutable reducer executes only after
real physical/source/runtime/input qualification and the fresh admission token.
"""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tarfile
import traceback


CAMPAIGN = "bb-20261108-reducer-validation-cpu-dev-v1"
ROOT = Path("/scratch/09999/aymanmahfuz/activation-checkpoint-integrity")
OUTPUT_MAX_BYTES = 2_000_000_000
SOURCE_FILES = (
    "experiments/bb_20261108/reducer_validation/runner.py",
    "experiments/bb_20261108/reducer_validation/test_reducer_allocated.py",
    "experiments/bb_20261108/reducer_validation/cpu-dev.sbatch",
    "experiments/bb_20261108/copy_runtime.py",
    "experiments/bb_20261108/aggregate_timing.py",
    "experiments/bb_20261108/e1_diagnostics.py",
    "experiments/bb_20261108/method-plan.json",
    "experiments/ba_20261107/allocated_common.py",
    "experiments/ba_20261107/setup_preflight/checks.py",
    "experiments/ba_20261107/provenance-environment.lock",
)
SUPPLEMENT_SHA = "e3bdeab25a77cf439b1ade4dc33d1b7d7e2e0cdaf238886c35738581aa992779"
CLARIFICATION_SHA = "90d0e0e979caccad5652a642f44e0bbf514e222f92d973deef74d255587091c6"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def interrupted(signum, frame):
    raise KeyboardInterrupt("Reducer fixture deadline reached; retain partial evidence")


def load_runtime(checkout):
    spec = importlib.util.spec_from_file_location(
        "_bb_reducer_copy_runtime", checkout / "experiments/bb_20261108/copy_runtime.py"
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.no_numerical_imports()
    return runtime


def verify_controls(checkout, path, contract, runtime, out):
    """Validate source-only fixture scope and exact three ACTUAL prerequisites."""
    require(contract.get("draft") is False, "Draft reducer fixture contract cannot run")
    require(
        path == checkout / "experiments/bb_20261108/reducer_validation/contract.json",
        "Exact fixture contract path required",
    )
    require(
        contract["campaign_id"] == CAMPAIGN
        and contract["output_study"] == "bb-reducer-validation"
        and contract["execution_phase"] == "CPU_DEV_REDUCER_FIXTURES"
        and contract["output_max_bytes"] == OUTPUT_MAX_BYTES,
        "Reducer fixture scope differs",
    )
    require(
        set(contract["files"]) == set(SOURCE_FILES), "Fixture source census differs"
    )
    require(
        contract["supplement_sha256"] == SUPPLEMENT_SHA
        and contract["reproduction_clarification_sha256"] == CLARIFICATION_SHA,
        "Prospective fixture controls differ",
    )
    actual = runtime.prerequisites(contract)
    expected = {
        "ACTUAL_BB_TIMING_V2_%d_VALIDITY_ACCEPTED" % index for index in range(3)
    }
    require(
        set(actual) == set(contract["required_review_sha256"]) == expected,
        "All three terminal actual reviews are required",
    )
    sources = contract["timing_validation_sources"]
    require(
        len(sources) == 3 and len({row["job_id"] for row in sources}) == 3,
        "Three distinct actual input origins required",
    )
    require(
        {row["role"] for row in contract["inputs"]}
        == {
            "timing-v2-result-0",
            "timing-v2-result-1",
            "timing-v2-result-2",
            "actual-third-pooled-analysis",
        }
        and len(contract["inputs"]) == 4,
        "Four exact validation inputs required",
    )
    for index, source in enumerate(sources):
        key = "ACTUAL_BB_TIMING_V2_%d_VALIDITY_ACCEPTED" % index
        item = actual[key]
        require(
            item["verdict"] == "PROMOTE"
            and item["job_id"]
            == source["job_id"]
            == contract["required_review_job_ids"][key]
            and item["review_sha256"]
            == source["review_sha256"]
            == contract["required_review_sha256"][key]
            and item["scope"]
            == contract["required_review_scopes"][key]
            == "ACTUAL_BB_TIMING_JOB_AND_CUSTODY_REVIEW"
            and item["validity_for_registered_three_job_pooling"] is True
            and item["result_sha256"] == source["sha256"]
            and item["custody_sha256"] == source["custody_sha256"],
            "Actual timing review/result/custody binding differs",
        )
        require(
            source["job_id"] != "3565495"
            and source["timing_job_index"] == index
            and type(source["timing_job_index"]) is int
            and source["cohort_id"] == "bb-20261108-timing-cap32-v2"
            and source["result_path"]
            == str(
                ROOT / "artifacts" / source["job_id"] / "bb-selector/timing-study.json"
            ),
            "Only the registered fresh cohort may supply fixture data",
        )
        row = next(
            value
            for value in contract["inputs"]
            if value["role"] == "timing-v2-result-%d" % index
        )
        require(
            row["path"] == source["result_path"]
            and row["sha256"] == source["sha256"]
            and row["bytes"] == source["bytes"],
            "Fixture input registry differs",
        )
    require(
        sources[0]["method_binding"]
        == sources[1]["method_binding"]
        == sources[2]["method_binding"],
        "Actual source/runtime/graph bindings differ",
    )
    for item in contract["inputs"]:
        source = Path(item["path"])
        require(
            not any(part.is_symlink() for part in (source, *source.parents)),
            "Input path contains symlink",
        )
        require(
            type(item["bytes"]) is int and 0 < item["bytes"] <= 33_554_432,
            "Validation JSON input size differs",
        )
        runtime._common.verify_file(source, item["sha256"])
        require(
            source.stat().st_size == item["bytes"],
            "Validation input byte count differs",
        )
    expected_analysis = next(
        value
        for value in contract["inputs"]
        if value["role"] == "actual-third-pooled-analysis"
    )
    require(
        expected_analysis["path"]
        == str(
            ROOT
            / "artifacts"
            / sources[2]["job_id"]
            / "bb-selector/timing-pooled-analysis.json"
        )
        and actual["ACTUAL_BB_TIMING_V2_2_VALIDITY_ACCEPTED"]["pooled_analysis_sha256"]
        == expected_analysis["sha256"],
        "Expected allocated third-job analysis is not bound",
    )
    runtime._common.write(out / "actual-prerequisite-review-bindings.json", actual)
    runtime._common.write(out / "input-pins-verified.json", contract["inputs"])
    return actual


def failure_retention_admission(common, checkout, out, job):
    """Bound failure custody to this owned CPU job without granting numerical work."""
    descriptor = common.command(
        out, "failure-retention-owned-job", ["scontrol", "show", "job", job, "-o"]
    )
    fields = dict(word.split("=", 1) for word in descriptor.split() if "=" in word)
    required = {
        "JobId": job,
        "UserId": "aymanmahfuz(893489)",
        "Account": "CCR26063",
        "JobState": "RUNNING",
        "JobName": "vg-bb-reducer-validation-cpu-dev",
        "Partition": "development",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "OverSubscribe": "NO",
        "Requeue": "0",
        "Restarts": "0",
        "TimeLimit": "01:00:00",
        "Command": str(
            checkout / "experiments/bb_20261108/reducer_validation/cpu-dev.sbatch"
        ),
    }
    require(
        all(fields.get(key) == value for key, value in required.items()),
        "Owned failure-retention resource descriptor differs",
    )
    nodes = common.command(
        out,
        "failure-retention-owned-nodes",
        ["scontrol", "show", "hostnames", fields["NodeList"]],
    ).splitlines()
    require(
        nodes == [socket.gethostname().split(".")[0]],
        "Failure collector is outside its assigned node",
    )
    batch = checkout / "experiments/bb_20261108/reducer_validation/cpu-dev.sbatch"
    actual = common.command(
        out,
        "failure-retention-actual-batch",
        ["scontrol", "write", "batch_script", job, "-"],
    )
    actual_sha = hashlib.sha256(actual.encode()).hexdigest()
    require(
        actual_sha == common.sha(batch),
        "Failure collector batch differs from scheduler",
    )
    # Retain fixed, nonsecret source paths even when source/release admission
    # failed. Missing files are explicit; this archive grants no source approval.
    source_names = SOURCE_FILES + (
        "experiments/bb_20261108/reducer_validation/contract.json",
    )
    sources = {}
    with tarfile.open(out / "failure-source-and-config.tar.gz", "x:gz") as archive:
        for name in source_names:
            source = checkout / name
            if source.is_file() and not source.is_symlink():
                archive.add(source, arcname=name, recursive=False)
                sources[name] = {
                    "status": "PRESENT_UNQUALIFIED",
                    "sha256": common.sha(source),
                }
            else:
                sources[name] = {"status": "MISSING_OR_NONREGULAR"}
    common.write(
        out / "failure-retention-admission.json",
        {
            "scope": "OWNED_JOB_FAILURE_CUSTODY_ONLY",
            "job_id": job,
            "account": "CCR26063",
            "node": nodes[0],
            "scientific_acceptance": False,
            "numerical_admission": False,
            "source_release_admission": "FAILED_OR_NOT_COMPLETED",
            "campaign": CAMPAIGN,
            "max_bytes": OUTPUT_MAX_BYTES,
            "actual_batch_sha256": actual_sha,
            "unqualified_sources": sources,
        },
    )


def main():
    checkout = Path(__file__).resolve().parents[3]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        sys.platform == "linux" and os.getuid() == 893489 and job.isdecimal(),
        "Owned Linux Slurm allocation required",
    )
    require(
        checkout == ROOT / "slurm/bb-20261108/reducer-validation-cpu-dev-source",
        "Exclusive reducer validation checkout required",
    )
    require(
        "BB_NUMERICAL_ADMISSION" not in os.environ,
        "Numerical admission cannot be inherited",
    )
    runtime = load_runtime(checkout)
    common = runtime._common
    out = ROOT / "artifacts" / job / "bb-reducer-validation"
    out.mkdir(parents=True, exist_ok=False)
    for signum in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, interrupted)
    signal.alarm(3000)
    admitted, status = False, 1
    try:
        path = Path(os.environ["BA_CONTRACT_PATH"])
        contract = common.admit(checkout, path, out)
        admitted = True
        common.write(out / "contract.json", contract)
        common.write(
            out / "execution-release.json",
            json.loads(Path(os.environ["BA_RELEASE_PATH"]).read_text()),
        )
        with tarfile.open(
            out / "exact-experiment-source-and-config.tar.gz", "x:gz"
        ) as archive:
            for name in sorted(contract["files"]):
                archive.add(checkout / name, arcname=name, recursive=False)
            archive.add(
                path, arcname="__controls__/execution-contract.json", recursive=False
            )
            archive.add(
                Path(os.environ["BA_RELEASE_PATH"]),
                arcname="__controls__/execution-release.json",
                recursive=False,
            )
        runtime.qualify(checkout, out, contract)
        verify_controls(checkout, path, contract, runtime, out)
        require(
            os.environ.get("BB_TEST_DEVICE") == "cpu"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "CPU-only fixture scope required",
        )
        runtime.no_numerical_imports()
        common.write(
            out / "numerical-admission.json",
            {
                "status": "PASS",
                "job_id": job,
                "scope": "FIXTURE_ONLY_EXISTING_THREE_PAYLOAD_REPRODUCTION_AND_REJECTION_GUARDS",
                "contract_sha256": common.sha(path),
                "physical_admission": "physical-admission.json",
                "runtime_qualification": "environment-qualification/results.json",
                "independent_timing_unit_added": False,
                "cuda_qualification": False,
                "scientific_acceptance": False,
            },
        )
        os.environ["BB_NUMERICAL_ADMISSION"] = "PASS:" + job
        sys.path.insert(0, str(checkout / "experiments/bb_20261108"))
        module = importlib.import_module("test_reducer_allocated")
        require(
            Path(module.__file__).resolve()
            == Path(__file__).resolve().parent / "test_reducer_allocated.py",
            "Fixture harness source differs",
        )
        result = module.run_allocated(
            out / "fixtures", contract, common.write, common.sha
        )
        common.write(
            out / "results.json",
            {
                "status": "FIXTURES_COMPLETE_PENDING_INDEPENDENT_ACTUAL_REVIEW",
                "fixture_count": result["tests_run"],
                "execution_complete": True,
                "independent_timing_unit_added": False,
                "numerical_method_changed": False,
                "cuda_qualification": False,
                "scientific_acceptance": False,
            },
        )
        status = 0
    except BaseException as error:
        common.write(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "scientific_acceptance": False,
                "automatic_retry": False,
            },
        )
    finally:
        signal.alarm(0)
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        common.write(
            out / "terminal.json",
            {
                "exit_code": status,
                "job_id": job,
                "source_admission_completed": admitted,
                "scientific_acceptance": False,
            },
        )
        try:
            if not admitted:
                failure_retention_admission(common, checkout, out, job)
            common.retain(out, CAMPAIGN, OUTPUT_MAX_BYTES)
        except BaseException as error:
            common.write(
                out / "retention-failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "durable_custody_complete": False,
                    "scratch_evidence_preserved": True,
                    "scientific_acceptance": False,
                },
            )
            raise
    return status


if __name__ == "__main__":
    raise SystemExit(main())
