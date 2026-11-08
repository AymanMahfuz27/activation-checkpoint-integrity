"""Run the sealed BB GPU pilot only after fresh allocated admission.

Top-level imports are standard library only. Fixtures precede historical model
work. Execution completion remains separate from independent actual review.
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
import unittest


CAMPAIGN = "bb-20261108-selector-gpu-dev-v1"
OUTPUT_MAX_BYTES = 2_000_000_000


def require(value, message):
    if not value:
        raise RuntimeError(message)


def interrupted(signum, frame):
    raise KeyboardInterrupt("GPU pilot deadline reached; retain partial evidence")


def load_runtime(checkout):
    """Load the exact BB stdlib runtime without a global import-path change."""
    spec = importlib.util.spec_from_file_location(
        "_bb_gpu_copy_runtime", checkout / "experiments/bb_20261108/copy_runtime.py"
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.no_numerical_imports()
    return runtime


def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


class RecordedResult(unittest.TextTestResult):
    """Retain which exact tests started/finished even if execution is interrupted."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.started_ids = []
        self.finished_ids = []
        self.passed_ids = []

    def startTest(self, test):
        self.started_ids.append(test.id())
        super().startTest(test)

    def stopTest(self, test):
        self.finished_ids.append(test.id())
        super().stopTest(test)

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def run_fixtures(out, contract, write):
    """Run only the ordered sealed IDs, retaining full outcomes and interruptions."""
    expected = contract["fixture_ids"]
    require(
        isinstance(expected, list)
        and len(expected) == 20
        and len(set(expected)) == len(expected),
        "Exactly twenty unique GPU fixture IDs are required",
    )
    write(out / "fixture-request.json", {"test_ids": expected})
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(expected)
    loaded = list(test_ids(suite))
    write(out / "fixture-census.json", {"expected": expected, "loaded": loaded})
    require(
        not loader.errors and loaded == expected, "Exact GPU fixture census differs"
    )
    for module_name in (
        "test_selector_allocated",
        "test_e1_allocated",
        "selector",
        "e1_diagnostics",
    ):
        module = importlib.import_module(module_name)
        require(
            Path(module.__file__).resolve()
            == Path(__file__).resolve().parent / (module_name + ".py"),
            "Imported fixture/core source differs: " + module_name,
        )
    holder = []
    normal_exit = False

    def make_result(stream, descriptions, verbosity):
        result = RecordedResult(stream, descriptions, verbosity)
        holder.append(result)
        return result

    try:
        with (out / "fixtures.log").open("x") as stream:
            result = unittest.TextTestRunner(
                stream=stream, verbosity=2, resultclass=make_result
            ).run(suite)
        normal_exit = True
    finally:
        if holder:
            result = holder[0]
            record = {
                "test_ids": loaded,
                "started_ids": result.started_ids,
                "finished_ids": result.finished_ids,
                "passed_ids": result.passed_ids,
                "tests_run": result.testsRun,
                "failures": [
                    {"id": test.id(), "traceback": text}
                    for test, text in result.failures
                ],
                "errors": [
                    {"id": test.id(), "traceback": text} for test, text in result.errors
                ],
                "skipped": [
                    {"id": test.id(), "reason": text} for test, text in result.skipped
                ],
                "expected_failures": [
                    {"id": test.id(), "traceback": text}
                    for test, text in result.expectedFailures
                ],
                "unexpected_successes": [
                    test.id() for test in result.unexpectedSuccesses
                ],
                "normal_exit": normal_exit,
                "successful": result.wasSuccessful(),
                "cuda_qualification": False,
                "scientific_acceptance": False,
            }
            write(out / "fixture-result.json", record)
    require(
        normal_exit
        and result.wasSuccessful()
        and not result.skipped
        and not result.expectedFailures
        and not result.unexpectedSuccesses
        and result.testsRun == len(expected)
        and result.started_ids == expected
        and result.finished_ids == expected
        and result.passed_ids == expected,
        "GPU fixtures failed, skipped, interrupted or did not complete the exact census",
    )
    return record


def verify_prerequisites(runtime, contract):
    """Require the exact accepted ACTUAL review hashes/scopes sealed by the operator."""
    actual = runtime.prerequisites(contract)
    require(bool(actual), "Nonempty ACTUAL prerequisite census required")
    pins = contract["required_review_sha256"]
    scopes = contract["required_review_scopes"]
    require(
        set(actual) == set(pins) == set(scopes), "ACTUAL review binding census differs"
    )
    for name, item in actual.items():
        require(
            item["review_sha256"] == pins[name] and item.get("scope") == scopes[name],
            "Exact ACTUAL review hash or scope differs: " + name,
        )
    return actual


def failure_retention_admission(common, checkout, out, job):
    """Bound failure custody to this owned GPU job without granting numerical work."""
    descriptor = common.command(
        out, "failure-retention-owned-job", ["scontrol", "show", "job", job, "-o"]
    )
    fields = dict(word.split("=", 1) for word in descriptor.split() if "=" in word)
    required = {
        "JobId": job,
        "UserId": "aymanmahfuz(893489)",
        "Account": "CCR26063",
        "JobState": "RUNNING",
        "JobName": "vg-bb-selector-gpu-dev",
        "Partition": "gpu-a100-dev",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "OverSubscribe": "NO",
        "Requeue": "0",
        "Restarts": "0",
        "TimeLimit": "00:40:00",
        "Command": str(checkout / "experiments/bb_20261108/selector-gpu-dev.sbatch"),
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
    batch = checkout / "experiments/bb_20261108/selector-gpu-dev.sbatch"
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
    source_names = (
        "experiments/bb_20261108/gpu_job.py",
        "experiments/bb_20261108/selector-gpu-dev.sbatch",
        "experiments/bb_20261108/selector-gpu-dev-contract.json",
        "experiments/bb_20261108/copy_runtime.py",
        "experiments/bb_20261108/method-plan.json",
        "experiments/bb_20261108/selector.py",
        "experiments/bb_20261108/selector_integration.py",
        "experiments/bb_20261108/e1_replay.py",
        "experiments/ba_20261107/opguard_baseline/run_arm.py",
        "experiments/bb_20261108/test_selector_allocated.py",
        "experiments/bb_20261108/e1_diagnostics.py",
        "experiments/bb_20261108/test_e1_allocated.py",
        "experiments/ba_20261107/allocated_common.py",
        "experiments/ba_20261107/setup_preflight/checks.py",
        "experiments/ba_20261107/provenance-environment.lock",
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


def verify_inputs(out, contract, common):
    """Hash the three opaque registered inputs only inside the admitted allocation."""
    inputs = contract["inputs"]
    roles = [item["role"] for item in inputs]
    require(
        len(roles) == len(set(roles)) == 3
        and set(roles) == {"train", "corrected210-history1234", "early-history1234"},
        "Exact pilot input role census differs",
    )
    records = []
    for item in inputs:
        path = Path(item["path"])
        common.verify_file(path, item["sha256"])
        require(path.stat().st_size == item["bytes"], "Input byte count differs")
        record = {
            "role": item["role"],
            "path": str(path),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "status": "ALLOCATED_BYTES_VERIFIED",
        }
        records.append(record)
        common.write(out / ("input-verified-" + item["role"] + ".json"), record)
    return records


def complete_stage(result, name):
    require(
        isinstance(result, dict)
        and result.get("status") == "COMPLETE_PENDING_INDEPENDENT_REVIEW"
        and result.get("execution_complete") is True,
        name + " is incomplete or invalid",
    )


def main():
    checkout = Path(__file__).resolve().parents[2]
    job = os.environ.get("SLURM_JOB_ID", "")
    require(
        sys.platform == "linux" and os.getuid() == 893489 and job.isdecimal(),
        "Owned Linux Slurm allocation required",
    )
    require(
        "BB_NUMERICAL_ADMISSION" not in os.environ,
        "Numerical admission may not be inherited",
    )
    runtime = load_runtime(checkout)
    common = runtime._common
    out = common.ROOT / "artifacts" / job / "bb-selector"
    out.mkdir(parents=True, exist_ok=False)
    for signum in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, interrupted)
    signal.alarm(3000)
    exit_code, admitted = 1, False
    try:
        path = Path(os.environ["BA_CONTRACT_PATH"])
        contract = common.admit(checkout, path, out)
        admitted = True
        common.write(out / "contract.json", contract)
        release = json.loads(Path(os.environ["BA_RELEASE_PATH"]).read_text())
        common.write(out / "execution-release.json", release)
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
        require(
            contract.get("draft") is False,
            "Draft GPU contracts cannot open numerical work",
        )
        require(
            contract["campaign_id"] == CAMPAIGN
            and contract["output_max_bytes"] == OUTPUT_MAX_BYTES
            and contract["execution_phase"] == "GPU_DEV_PILOT",
            "GPU scope or output envelope differs",
        )
        method = json.loads((checkout / contract["method_plan_path"]).read_text())
        require(
            method == contract["method_plan"]
            and method["gpu_fixture_ids"] == contract["fixture_ids"],
            "Embedded method or GPU fixture census differs from frozen source",
        )
        runtime.qualify(checkout, out, contract)
        reviews = verify_prerequisites(runtime, contract)
        common.write(out / "actual-prerequisite-review-bindings.json", reviews)
        historical = Path(contract["source_path"])
        original = contract["historical_contract"]
        revision = common.command(
            out, "historical-head", ["git", "-C", str(historical), "rev-parse", "HEAD"]
        ).strip()
        require(
            revision == original["execution_source_revision"],
            "Historical source revision differs",
        )
        for name, pin in original["source_files"].items():
            require(
                not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Historical source member leaves its tree",
            )
            common.verify_file(historical / name, pin)
        with tarfile.open(out / "exact-historical-source.tar.gz", "x:gz") as archive:
            for name in sorted(original["source_files"]):
                archive.add(historical / name, arcname=name, recursive=False)
        inputs = verify_inputs(out, contract, common)
        require(
            os.environ.get("BB_TEST_DEVICE") == "cuda:0"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2",
            "Frozen GPU/RNG visibility differs",
        )
        runtime.no_numerical_imports()
        common.write(
            out / "numerical-admission.json",
            {
                "status": "PASS",
                "scope": "GPU_FIXTURES_AND_REGISTERED_PILOT_E1_PROBES_ONLY",
                "job_id": job,
                "contract_sha256": common.sha(path),
                "source_revision": revision,
                "physical_admission": "physical-admission.json",
                "runtime_qualification": "environment-qualification/results.json",
                "scientific_acceptance": False,
            },
        )
        os.environ["BB_NUMERICAL_ADMISSION"] = "PASS:" + job
        sys.path.insert(0, str(historical / "src"))
        import torch
        import numpy
        import transformers

        require(
            torch.__version__ == "2.13.0+cu126" and numpy.__version__ == "2.5.2",
            "Pinned imported Torch/NumPy versions differ",
        )
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 3,
            "Exactly three visible CUDA devices required for frozen RNG topology",
        )
        devices = []
        for index in range(3):
            properties = torch.cuda.get_device_properties(index)
            require(
                "A100" in properties.name
                and (properties.major, properties.minor) == (8, 0),
                "Registered A100 sm_80 devices required",
            )
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "memory_bytes": properties.total_memory,
                }
            )
        common.command(out, "nvidia-smi", ["nvidia-smi"])
        processes = common.command(
            out,
            "gpu-processes",
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
        )
        require(
            not any(
                line.split(",")[0].strip() != str(os.getpid())
                for line in processes.splitlines()
                if line.strip()
            ),
            "Unexpected GPU compute process on exclusive node",
        )
        fingerprint = importlib.import_module("ac_integrity.capture.fingerprint")
        require(
            Path(fingerprint.__file__).resolve()
            == historical / "src/ac_integrity/capture/fingerprint.py",
            "Imported fingerprint source differs",
        )
        common.write(
            out / "runtime.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "prefix": sys.prefix,
                "torch": torch.__version__,
                "numpy": numpy.__version__,
                "transformers": transformers.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": True,
                "cuda_device_count": 3,
                "devices": devices,
                "test_device": "cuda:0",
            },
        )
        fixture = run_fixtures(out, contract, common.write)
        native = importlib.import_module("ac_integrity.capture.cuda_fingerprint")
        require(
            Path(native.__file__).resolve()
            == historical / "src/ac_integrity/capture/cuda_fingerprint.py",
            "Native fingerprint loader source differs",
        )
        setup = native.preload_cuda_fingerprint(torch.device("cuda:0"))
        common.write(out / "native-fingerprint-setup.json", setup)
        require(
            setup["architecture"] == "sm_80"
            and setup["source_sha256"]
            == contract["expected_native_extension_source_sha256"],
            "Native fingerprint source/architecture differs",
        )
        model_release = out / "model-work-release.json"
        common.write(
            model_release,
            {
                "status": "PASS",
                "job_id": job,
                "account": "CCR26063",
                "source_revision": revision,
                "contract_sha256": common.sha(path),
                "scope": "BB_SELECTOR_GPU_DEV_PILOT_AND_E1_PROBES_ONLY",
                "fixtures_passed": fixture["tests_run"],
                "scientific_acceptance": False,
            },
        )
        os.environ["ACI_MODEL_WORK_RELEASE"] = str(model_release)
        os.environ["ACI_CONTRACT_SHA256"] = common.sha(path)
        from ac_integrity.research import completion

        completion.source_allocation_guard()
        completion.cutoff()
        from selector_integration import run_pilot

        pilot = run_pilot(out / "selector-study", contract)
        common.write(out / "selector-result.json", pilot)
        complete_stage(pilot, "Selector pilot")
        require(
            pilot.get("evidence_complete") is True
            and pilot["cases_declared"] == len(method["model_protocol"]["cases"])
            and pilot["supported_cases"] + pilot["unsupported_cases"]
            == pilot["cases_declared"]
            and pilot["arms_declared"] == method["model_protocol"]["model_arms_planned"]
            and pilot["arms_executed"] + pilot["arms_not_run_unsupported"]
            == pilot["arms_declared"],
            "Complete pilot denominator or evidence census differs",
        )
        # Coherent unsupported/falsified hypotheses are retained normal outcomes.
        # Source/runtime/parity/injection/native execution exceptions fail fast.
        completion.source_allocation_guard()
        completion.cutoff()
        from e1_replay import run_e1_probes

        e1 = run_e1_probes(out / "e1-probes", contract)
        common.write(out / "e1-result.json", e1)
        complete_stage(e1, "E1 diagnostic replay")
        require(
            e1["arms_executed"] == 12
            and len(e1["paired_contexts"]) == 6
            and all(
                row["noninterference_valid"] is True for row in e1["paired_contexts"]
            )
            and e1["original_638_call_census_and_26_flags_unchanged"] is True
            and e1["native_clipping_implementation_changed"] is False,
            "E1 diagnostic census/parity or preserved criteria differ",
        )
        common.write(
            out / "results.json",
            {
                "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
                "execution_complete": True,
                "fixture_count": fixture["tests_run"],
                "inputs": inputs,
                "selector": pilot,
                "e1": e1,
                "timing_not_released": True,
                "cuda_fixtures_execution_complete": True,
                "scientific_acceptance": False,
            },
        )
        exit_code = 0
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
                "exit_code": exit_code,
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
                    "scientific_acceptance": False,
                    "durable_custody_complete": False,
                    "scratch_evidence_preserved": True,
                },
            )
            raise
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
