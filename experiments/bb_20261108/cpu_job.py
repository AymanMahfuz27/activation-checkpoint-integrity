"""Run the sealed BB CPU fixtures after fresh physical/runtime admission.

Only standard-library modules load at import time. No checkpoint or model is
restored. A passing CPU job remains pending actual review and makes no CUDA or
scientific acceptance claim.
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


CAMPAIGN = "bb-20261108-e1-cpu-dev-v1"
OUTPUT_MAX_BYTES = 2_000_000_000


def require(value, message):
    if not value:
        raise RuntimeError(message)


def interrupted(signum, frame):
    raise KeyboardInterrupt("CPU fixture deadline reached; retain partial evidence")


def load_runtime(checkout):
    """Load the exact BB stdlib runtime without a global import-path change."""
    spec = importlib.util.spec_from_file_location(
        "_bb_cpu_copy_runtime", checkout / "experiments/bb_20261108/copy_runtime.py"
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
        "Exactly twenty unique CPU fixture IDs are required",
    )
    write(out / "fixture-request.json", {"test_ids": expected})
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(expected)
    loaded = list(test_ids(suite))
    write(out / "fixture-census.json", {"expected": expected, "loaded": loaded})
    require(
        not loader.errors and loaded == expected, "Exact CPU fixture census differs"
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
        "CPU fixtures failed, skipped, interrupted or did not complete the exact census",
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


def retained_e1(out, contract, common):
    """Export the unchanged small historical strict-failure census as metadata."""
    inputs = contract["inputs"]
    require(
        len(inputs) == 1 and inputs[0]["role"] == "retained-e1-strict-dispositions",
        "CPU input registry must contain only the retained E1 disposition JSON",
    )
    item = inputs[0]
    source = Path(item["path"])
    common.verify_file(source, item["sha256"])
    require(
        source.stat().st_size == item["bytes"] <= 100_000,
        "Disposition input size differs",
    )
    raw = source.read_bytes()
    original = json.loads(raw)
    rows = original["rows"]
    require(
        original["strict_validation_preserved"] is True
        and original["scientific_acceptance"] is False
        and len(rows) == 26
        and len({row["id"] for row in rows}) == 26
        and all(
            row["strict_validation_preserved"] is True
            and row["scientific_acceptance"] is False
            for row in rows
        ),
        "Historical strict E1 disposition metadata differs",
    )
    counts = {}
    for row in rows:
        category = row["disposition"]
        counts[category] = counts.get(category, 0) + 1
    require(
        counts == contract["retained_e1_disposition_counts"],
        "Historical disposition census differs",
    )
    target = out / "retained-e1-validation-dispositions.json"
    with target.open("xb") as stream:
        stream.write(raw)
    require(common.sha(target) == item["sha256"], "Exported E1 bytes differ")
    summary = {
        "source": str(source),
        "sha256": item["sha256"],
        "rows": len(rows),
        "disposition_counts": counts,
        "strict_validation_preserved": True,
        "scientific_acceptance": False,
        "interpretation": original["interpretation"],
        "purpose": "Retained metadata only; no old criterion or denominator is changed",
    }
    common.write(out / "retained-e1-census.json", summary)
    return summary


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
        "JobName": "vg-bb-e1-cpu-dev",
        "Partition": "development",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "OverSubscribe": "NO",
        "Requeue": "0",
        "Restarts": "0",
        "TimeLimit": "01:00:00",
        "Command": str(checkout / "experiments/bb_20261108/e1-cpu-dev.sbatch"),
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
    batch = checkout / "experiments/bb_20261108/e1-cpu-dev.sbatch"
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
        "experiments/bb_20261108/cpu_job.py",
        "experiments/bb_20261108/e1-cpu-dev.sbatch",
        "experiments/bb_20261108/e1-cpu-dev-contract.json",
        "experiments/bb_20261108/copy_runtime.py",
        "experiments/bb_20261108/method-plan.json",
        "experiments/bb_20261108/selector.py",
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
    out = common.ROOT / "artifacts" / job / "bb-e1"
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
            contract["campaign_id"] == CAMPAIGN
            and contract["output_max_bytes"] == OUTPUT_MAX_BYTES
            and contract["execution_phase"] == "CPU_DEV_FIXTURES",
            "CPU scope or output envelope differs",
        )
        plan = json.loads((checkout / contract["method_plan_path"]).read_text())
        require(
            plan["cpu_fixture_ids"] == contract["fixture_ids"],
            "CPU fixtures differ from the frozen method plan",
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
        census = retained_e1(out, contract, common)
        require(
            os.environ.get("BB_TEST_DEVICE") == "cpu"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "CPU-only fixture environment differs",
        )
        runtime.no_numerical_imports()
        common.write(
            out / "numerical-admission.json",
            {
                "status": "PASS",
                "scope": "CPU_FIXTURES_ONLY",
                "job_id": job,
                "contract_sha256": common.sha(path),
                "source_revision": revision,
                "physical_admission": "physical-admission.json",
                "runtime_qualification": "environment-qualification/results.json",
                "cuda_qualification": False,
                "scientific_acceptance": False,
            },
        )
        os.environ["BB_NUMERICAL_ADMISSION"] = "PASS:" + job
        sys.path.insert(0, str(historical / "src"))
        import torch
        import numpy

        require(
            torch.__version__ == "2.13.0+cu126" and numpy.__version__ == "2.5.2",
            "Pinned imported Torch/NumPy versions differ",
        )
        require(
            not torch.cuda.is_available() and torch.cuda.device_count() == 0,
            "CPU fixture process unexpectedly exposes CUDA",
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
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": False,
                "cuda_device_count": 0,
                "test_device": "cpu",
                "model_restored": False,
            },
        )
        fixture = run_fixtures(out, contract, common.write)
        common.write(
            out / "results.json",
            {
                "status": "CPU_FIXTURES_COMPLETE_PENDING_INDEPENDENT_REVIEW",
                "execution_complete": True,
                "fixture_count": fixture["tests_run"],
                "retained_e1": census,
                "historical_model_arms": 0,
                "cuda_qualification": False,
                "historical_strict_validation_changed": False,
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
