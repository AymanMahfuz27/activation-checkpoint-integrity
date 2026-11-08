"""Run one registered BB timing job after fresh allocated admission.

Top-level imports are standard library only. Fixtures precede historical model
work. Only index2 reduces allthree jobs; actual review remains separate.
"""

import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
from pathlib import Path
import signal
import socket
import sys
import tarfile
import traceback
import unittest


REMOTE_ROOT = Path("/scratch/09999/aymanmahfuz/activation-checkpoint-integrity")
MAX_TIMING_RESULT_BYTES = 33_554_432
OUTPUT_MAX_BYTES = 2_000_000_000
COHORT_ID = "bb-20261108-timing-cap32-v2"
TRANSPORT_AMENDMENT_SHA256 = (
    "d439e623a43f773c2bd4132e5032dd3ecca187965b16d0f85107c72587cd61fc"
)
REDUCTION_CLARIFICATION_SHA256 = (
    "ff09eeac86e46bcd87f7909e6ad2756177a504ef7420c80dd3626fe718e5a741"
)
FAILURE_REVIEW_KEY = "ACTUAL_BB_TIMING_V1_CAP_FAILURE_REVIEWED"
CLOSED_JOB_IDS = {"3565495"}
PILOT_REVIEW_KEY = "ACTUAL_BB_SELECTOR_PILOT_ACCEPTED"
BASE_REVIEW_KEYS = {
    "ACTUAL_BA_PRIMARY_PROVENANCE_ACCEPTED",
    "ACTUAL_BA_PLACEMENT_CONTROL_ACCEPTED",
    "ACTUAL_BA_SIX_CASE_MATRIX_ACCEPTED",
    "ACTUAL_BA_CONFIGURED_OFFICIAL_BASELINE_ACCEPTED",
    "ACTUAL_BA_RETAINED_CLOSEOUT_REVIEWED",
    "ACTUAL_BB_CPU_QUALIFICATION_ACCEPTED",
    PILOT_REVIEW_KEY,
    FAILURE_REVIEW_KEY,
}
POSITIVE_PILOT_GATE = {
    "execution_complete": True,
    "evidence_complete": True,
    "arms_declared": 84,
    "arms_executed": 84,
    "arms_not_run_unsupported": 0,
    "cases_declared": 12,
    "supported_cases": 12,
    "unsupported_cases": 0,
    "parameter_count_delta": 0,
    "all_declared_cases_supported": True,
    "all_detection_expectations_met": True,
    "strict_fingerprint_byte_reduction": True,
    "timing_prerequisites_met": True,
    "scientific_outcome": "EXPECTED_PATTERN",
}
SOURCE_FILES = (
    "experiments/bb_20261108/timing_job.py",
    "experiments/bb_20261108/timing-dev.sbatch",
    "experiments/bb_20261108/copy_runtime.py",
    "experiments/bb_20261108/selector.py",
    "experiments/bb_20261108/test_selector_allocated.py",
    "experiments/bb_20261108/selector_integration.py",
    "experiments/bb_20261108/e1_diagnostics.py",
    "experiments/bb_20261108/test_e1_allocated.py",
    "experiments/bb_20261108/method-plan.json",
    "experiments/bb_20261108/aggregate_timing.py",
    "experiments/ba_20261107/allocated_common.py",
    "experiments/ba_20261107/setup_preflight/checks.py",
    "experiments/ba_20261107/provenance-environment.lock",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def checkout_index(checkout):
    """Derive an index only from one of the three exclusive source namespaces."""
    checkout = Path(checkout)
    require(
        checkout.parent == REMOTE_ROOT / "slurm/bb-20261108",
        "Timing checkout parent differs",
    )
    match = re.fullmatch(r"timing-v2-([012])-dev-source", checkout.name)
    require(match is not None, "Timing checkout does not name a registered index")
    return int(match[1])


def campaign(index):
    require(type(index) is int and index in (0, 1, 2), "Invalid timing index")
    return "bb-20261108-timing-v2-%d-dev-v1" % index


def exact_fields(actual, expected, label):
    """Compare metadata with explicit types, so boolean True cannot stand for1."""
    require(
        isinstance(actual, dict) and set(actual) == set(expected),
        label + " census differs",
    )
    for name, value in expected.items():
        require(
            type(actual[name]) is type(value) and actual[name] == value,
            label + " differs: " + name,
        )


def verify_timing_scope(checkout, path, contract, index):
    require(
        contract.get("draft") is False, "Timing template cannot open numerical work"
    )
    require(
        type(contract["timing_job_index"]) is int
        and contract["timing_job_index"] == index,
        "Timing index differs from exclusive checkout",
    )
    require(
        path
        == checkout
        / ("experiments/bb_20261108/timing-v2-%d-dev-contract.json" % index),
        "Sealed timing contract path differs",
    )
    exact_fields(
        {
            key: contract[key]
            for key in (
                "campaign_id",
                "source_checkout_name",
                "control_directory_name",
                "job_name",
                "partition",
                "time_limit",
                "cpus",
                "max_su",
                "output_max_bytes",
                "execution_phase",
            )
        },
        {
            "campaign_id": campaign(index),
            "source_checkout_name": "timing-v2-%d-dev-source" % index,
            "control_directory_name": "timing-v2-%d-dev-control" % index,
            "job_name": "vg-bb-timing-v2-dev",
            "partition": "gpu-a100-dev",
            "time_limit": "00:40:00",
            "cpus": 8,
            "max_su": 2,
            "output_max_bytes": OUTPUT_MAX_BYTES,
            "execution_phase": "GPU_DEV_TIMING",
        },
        "Registered timing scope",
    )
    require(
        set(contract["files"]) == set(SOURCE_FILES),
        "Common thirteen-source census differs",
    )
    exact_fields(
        {
            name: contract[name]
            for name in (
                "cohort_id",
                "transport_amendment_sha256",
                "reduction_clarification_sha256",
                "raw_timing_result_max_bytes",
            )
        },
        {
            "cohort_id": COHORT_ID,
            "transport_amendment_sha256": TRANSPORT_AMENDMENT_SHA256,
            "reduction_clarification_sha256": REDUCTION_CLARIFICATION_SHA256,
            "raw_timing_result_max_bytes": MAX_TIMING_RESULT_BYTES,
        },
        "Fresh cap32 cohort binding",
    )
    exact_fields(
        contract["required_pilot_timing_eligibility"],
        POSITIVE_PILOT_GATE,
        "Required positive pilot gate",
    )


def timing_review_key(index):
    require(
        type(index) is int and index in (0, 1),
        "Only first two jobs are prior timing results",
    )
    return "ACTUAL_BB_TIMING_V2_%d_VALIDITY_ACCEPTED" % index


def verify_timing_reviews(actual, contract, index):
    """Require positive pilot eligibility; timing prerequisites qualify validity only."""
    extras = set(actual) - BASE_REVIEW_KEYS
    allowed = {timing_review_key(0)} if index == 1 else set()
    if index == 2:
        allowed = {timing_review_key(0), timing_review_key(1)}
    require(
        BASE_REVIEW_KEYS <= set(actual) and extras <= allowed,
        "Timing prerequisite census differs",
    )
    require(extras == allowed, "Fresh cohort needs all earlier actual validity reviews")
    require(
        actual[PILOT_REVIEW_KEY]["verdict"] == "PROMOTE",
        "Pilot actual review is not promoted",
    )
    failure = actual[FAILURE_REVIEW_KEY]
    require(
        failure["verdict"] == "MODIFY"
        and failure["job_id"] == "3565495"
        and failure["scope"] == "ACTUAL_BB_TIMING_JOB_AND_CUSTODY_REVIEW"
        and failure["review_sha256"]
        == "1d848a04629a69f7a78311736b2086eccd0b0e68ebd8a23bee8c51f5d49ca0a3",
        "Closed v1 failure disposition is not bound",
    )
    exact_fields(
        actual[PILOT_REVIEW_KEY].get("timing_eligibility"),
        POSITIVE_PILOT_GATE,
        "Actual pilot timing eligibility",
    )
    for name in extras:
        require(
            actual[name]["verdict"] == "PROMOTE"
            and actual[name]["job_id"] not in CLOSED_JOB_IDS,
            "Prior fresh-cohort timing validity was not accepted",
        )


def interrupted(signum, frame):
    raise KeyboardInterrupt("Timing deadline reached; retain partial evidence")


def load_runtime(checkout):
    """Load the exact BB stdlib runtime without a global import-path change."""
    spec = importlib.util.spec_from_file_location(
        "_bb_timing_copy_runtime", checkout / "experiments/bb_20261108/copy_runtime.py"
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


def failure_retention_admission(common, checkout, out, job, index):
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
        "JobName": "vg-bb-timing-v2-dev",
        "Partition": "gpu-a100-dev",
        "NumNodes": "1",
        "NumTasks": "1",
        "CPUs/Task": "8",
        "OverSubscribe": "NO",
        "Requeue": "0",
        "Restarts": "0",
        "TimeLimit": "00:40:00",
        "Command": str(checkout / "experiments/bb_20261108/timing-dev.sbatch"),
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
    batch = checkout / "experiments/bb_20261108/timing-dev.sbatch"
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
        "experiments/bb_20261108/timing-v2-%d-dev-contract.json" % index,
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
            "campaign": campaign(index),
            "max_bytes": OUTPUT_MAX_BYTES,
            "actual_batch_sha256": actual_sha,
            "unqualified_sources": sources,
        },
    )


def sha_text(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def prior_input_role(index):
    return "timing-v2-result-%d" % index


def verify_input_registry(contract, index):
    """Validate prior-result declarations without reading their timing bytes yet."""
    inputs = contract["inputs"]
    roles = [item["role"] for item in inputs]
    expected = {"train", "corrected210-history1234"}
    if index == 2:
        expected |= {prior_input_role(0), prior_input_role(1)}
    require(
        len(roles) == len(set(roles)) and set(roles) == expected,
        "Timing input census differs",
    )
    wrappers = contract["timing_analysis_prior_results"]
    require(
        isinstance(wrappers, list) and len(wrappers) == (2 if index == 2 else 0),
        "Prior timing wrapper census differs",
    )
    require(
        all(
            isinstance(row, dict) and isinstance(row.get("job_id"), str)
            for row in wrappers
        ),
        "Prior timing job metadata is malformed",
    )
    require(
        len({row["job_id"] for row in wrappers}) == len(wrappers),
        "Prior timing jobs are not distinct",
    )
    for previous_index, wrapper in enumerate(wrappers):
        require(
            isinstance(wrapper, dict)
            and set(wrapper)
            == {"job_id", "result_path", "method_binding", "evidence_attestation"},
            "Prior wrapper fields differ",
        )
        job = wrapper["job_id"]
        require(
            isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job) is not None,
            "Prior timing job ID is invalid",
        )
        require(
            job not in CLOSED_JOB_IDS, "Closed old-cohort timing data cannot enter v2"
        )
        path = REMOTE_ROOT / "artifacts" / job / "bb-selector/timing-study.json"
        item = next(
            row for row in inputs if row["role"] == prior_input_role(previous_index)
        )
        require(
            wrapper["result_path"] == str(path) == item["path"]
            and item["job_id"] == job
            and type(item["timing_job_index"]) is int
            and item["timing_job_index"] == previous_index,
            "Prior result path/index differs",
        )
        require(
            item["cohort_id"] == COHORT_ID,
            "Prior timing input names a different cohort",
        )
        require(
            type(item["bytes"]) is int
            and 0 < item["bytes"] <= MAX_TIMING_RESULT_BYTES
            and sha_text(item["sha256"]),
            "Prior result pin or size is invalid",
        )
        evidence = wrapper["evidence_attestation"]
        require(
            isinstance(evidence, dict)
            and set(evidence)
            == {"status", "job_id", "result_sha256", "custody_sha256", "review_sha256"},
            "Prior evidence fields differ",
        )
        require(
            evidence["status"] == "ACTUAL_PROMOTED"
            and evidence["job_id"] == job
            and evidence["result_sha256"] == item["sha256"]
            and all(
                sha_text(evidence[name])
                for name in ("result_sha256", "review_sha256", "custody_sha256")
            ),
            "Prior result lacks exact actual evidence pins",
        )
        binding = wrapper["method_binding"]
        fields = {
            "method_plan_sha256",
            "selector_core_sha256",
            "selector_integration_sha256",
            "requirements_lock_sha256",
            "model_graph_sha256",
            "runtime_sha256",
        }
        require(
            isinstance(binding, dict)
            and set(binding) == fields | {"production_recipe"}
            and all(sha_text(binding[name]) for name in fields),
            "Prior numerical binding schema differs",
        )
        require(
            binding["method_plan_sha256"] == contract["method_plan_sha256"]
            and binding["selector_core_sha256"]
            == contract["files"]["experiments/bb_20261108/selector.py"]
            and binding["selector_integration_sha256"]
            == contract["files"]["experiments/bb_20261108/selector_integration.py"]
            and binding["requirements_lock_sha256"]
            == contract["files"][contract["requirements_lock"]]
            and binding["production_recipe"]
            == contract["method_plan"]["timing_protocol"]["production_batch"],
            "Prior static method/source/recipe binding differs",
        )
    return inputs


def verify_opaque_inputs(out, inputs, common):
    """Hash only corpus/checkpoint inputs before measurements; never inspect prior timings."""
    records = []
    for item in inputs:
        if item["role"] not in {"train", "corrected210-history1234"}:
            continue
        path = Path(item["path"])
        common.verify_file(path, item["sha256"])
        require(path.stat().st_size == item["bytes"], "Opaque input byte count differs")
        record = {
            "role": item["role"],
            "path": str(path),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
            "status": "ALLOCATED_BYTES_VERIFIED",
        }
        records.append(record)
        common.write(out / ("input-verified-" + item["role"] + ".json"), record)
    require(len(records) == 2, "Both timing opaque inputs are required")
    return records


def exact_parity(value):
    return (
        isinstance(value, dict)
        and value.get("valid") is True
        and value.get("byte_equal") is True
        and value.get("all_finite") is True
        and value.get("changed_scalar_paths") == []
    )


def validate_timing_result(result, method, index):
    """Check completeness/parity, without selecting jobs by timing direction."""
    require(
        isinstance(result, dict)
        and result.get("experiment_id") == method["experiment_id"]
        and result.get("status") == "COMPLETE_PENDING_INDEPENDENT_REVIEW"
        and result.get("execution_complete") is True
        and result.get("phase") == "TIMING"
        and type(result.get("independent_job_index")) is int
        and result["independent_job_index"] == index,
        "Timing result is incomplete or names another index",
    )
    require(
        not {"job_id", "result_path", "method_binding", "evidence_attestation"}
        & set(result),
        "Base timing payload contains wrapper metadata",
    )
    for name, expected in (
        ("updates_executed", 128),
        ("timed_arm_updates", 128),
        ("production_profile_updates", 1),
        ("total_updates_executed", 129),
    ):
        require(
            type(result.get(name)) is int and result[name] == expected,
            "Timing update census differs: " + name,
        )
    require(
        result["updates_executed_scope"] == "timed arms only"
        and result["independent_unit"] == "Slurm job"
        and result["scientific_acceptance"] is False,
        "Timing scope or acceptance label differs",
    )
    protocol = method["timing_protocol"]
    schedule = [
        row for row in protocol["frozen_pair_schedule"] if row["job_index"] == index
    ]
    require(
        len(result["pairs"]) == len(schedule) == 8
        and len(result["primary_direct_blocks"]) == 2,
        "Timing pair/block census differs",
    )
    for position, (pair, planned) in enumerate(
        zip(result["pairs"], schedule, strict=True)
    ):
        exact_fields(
            {
                key: pair[key]
                for key in ("schedule_position", "policy", "block_index", "arm_order")
            },
            {
                "schedule_position": position,
                "policy": planned["policy"],
                "block_index": planned["block_index"],
                "arm_order": planned["arm_order"],
            },
            "Frozen timing pair schedule",
        )
        require(
            set(pair["arms"]) == {"disabled", planned["policy"]},
            "Timing pair arm census differs",
        )
        parity = pair["final_eight_update_parity"]
        require(
            set(parity) == {"state", "gradients"}
            and all(exact_parity(row) for row in parity.values())
            and pair["final_gradient_phase"] == "post-clipping",
            "Final eight-update parity is not exact",
        )
        for arm in pair["arms"].values():
            require(
                len(arm["warmup_seconds"]) == 2 and len(arm["measured_seconds"]) == 6,
                "Timing warmup/measured census differs",
            )
            require(
                all(
                    type(value) in (float, int) and math.isfinite(value) and value > 0
                    for value in arm["warmup_seconds"] + arm["measured_seconds"]
                ),
                "A measured duration is not finite positive",
            )
    census = result["model_census"]
    require(
        census["status"] == "PASS"
        and census["validation_scope"] == "COMPILED_MODEL_PLAN"
        and census["outside_binding_map_validated"] is True
        and census["recipe"] == protocol["production_batch"]
        and bool(census["checks"])
        and all(value is True for value in census["checks"].values()),
        "Production model census is not qualified",
    )
    require(
        len(census["outside_bindings"]) == 96
        and all(row["valid"] is True for row in census["outside_bindings"]),
        "Production outside-scope census differs",
    )
    require(
        result["profile_cost_is_not_free"] is True
        and result["no_pooled_interval_from_one_job"] is True
        and result["primary_estimator"] == protocol["estimator"],
        "Timing estimator or profiling scope differs",
    )


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def method_binding(out, contract, result, driver_version):
    """Bind comparable numerical source/runtime/graph metadata, excluding job controls."""
    plan = json.loads((out / "timing-study/production-plan.json").read_text())
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    require(
        canonical_sha(body) == plan["plan_sha256"] == result["production_plan_sha256"],
        "Production plan digest differs",
    )
    envelope = plan["declared_envelope"]
    require(
        envelope["recipe"]
        == contract["method_plan"]["timing_protocol"]["production_batch"],
        "Production recipe binding differs",
    )
    identity = envelope["admitted_identity"]
    require(
        identity["source_file_pins"] == contract["files"]
        and identity["historical_source_file_pins"]
        == contract["historical_contract"]["source_files"]
        and identity["historical_source_revision"]
        == contract["historical_contract"]["execution_source_revision"],
        "Admitted scientific identity differs",
    )
    return {
        "method_plan_sha256": contract["method_plan_sha256"],
        "selector_core_sha256": contract["files"][
            "experiments/bb_20261108/selector.py"
        ],
        "selector_integration_sha256": contract["files"][
            "experiments/bb_20261108/selector_integration.py"
        ],
        "requirements_lock_sha256": contract["files"][contract["requirements_lock"]],
        "model_graph_sha256": canonical_sha(envelope["allowed_model_graph"]),
        "runtime_sha256": canonical_sha(
            {"admitted_identity": identity, "nvidia_driver_version": driver_version}
        ),
        "production_recipe": envelope["recipe"],
    }


def read_prior_result(item):
    """Read bounded regular bytes after current measurements, never following symlinks."""
    path = Path(item["path"])
    require(
        all(not part.is_symlink() for part in (path, *path.parents)),
        "Prior result path contains a symlink",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= MAX_TIMING_RESULT_BYTES
            and before.st_size == item["bytes"],
            "Prior result is not a bounded regular declared file",
        )
        raw = stream.read(MAX_TIMING_RESULT_BYTES)
        after = os.fstat(stream.fileno())

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    require(
        identity(before) == identity(after)
        and len(raw) == before.st_size
        and hashlib.sha256(raw).hexdigest() == item["sha256"],
        "Prior timing bytes changed or differ from their exact pin",
    )
    return raw


def analysis_records(out, contract, actual_reviews, binding, job, common):
    """Build separate attestations only after the immutable current base file exists."""
    records = json.loads(json.dumps(contract["timing_analysis_prior_results"]))
    require(
        len(records) == 2 and len({row["job_id"] for row in records} | {job}) == 3,
        "Three distinct analysis jobs are required",
    )
    current = out / "timing-study.json"
    records.append(
        {
            "job_id": job,
            "result_path": str(current),
            "method_binding": binding,
            "evidence_attestation": {
                "status": "CURRENT_COMPLETE_PENDING_REVIEW",
                "job_id": job,
                "result_sha256": common.sha(current),
                "review_sha256": None,
                "custody_sha256": None,
            },
        }
    )
    common.write(out / "analysis-input-bindings.json", records)
    common.write(
        out / "analysis-validation-pending.json",
        {"status": "PENDING_INPUT_BYTES_AND_REDUCER", "scientific_acceptance": False},
    )
    copies = []
    for index, wrapper in enumerate(records[:2]):
        key = timing_review_key(index)
        actual = actual_reviews[key]
        evidence = wrapper["evidence_attestation"]
        require(
            actual["job_id"] == wrapper["job_id"]
            and actual["review_sha256"] == evidence["review_sha256"]
            and actual.get("result_sha256") == evidence["result_sha256"]
            and actual.get("custody_sha256") == evidence["custody_sha256"],
            "Prior actual review/result/custody attestation differs",
        )
        require(
            wrapper["method_binding"] == binding,
            "Prior scientific method/runtime/graph binding differs",
        )
        item = next(
            row for row in contract["inputs"] if row["role"] == prior_input_role(index)
        )
        raw = read_prior_result(item)
        destination = (
            out / "analysis-prior-results" / wrapper["job_id"] / "timing-study.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        require(
            common.sha(destination) == item["sha256"],
            "Retained prior result copy differs",
        )
        copies.append(
            {
                "source": item["path"],
                "retained_copy": str(destination.relative_to(out)),
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
        )
    common.write(
        out / "analysis-input-custody.json",
        {
            "prior_result_copies": copies,
            "current_actual_review": None,
            "current_custody_review": None,
        },
    )
    return records


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
    index = checkout_index(checkout)
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
        verify_timing_scope(checkout, path, contract, index)
        method = json.loads((checkout / contract["method_plan_path"]).read_text())
        require(
            method == contract["method_plan"]
            and method["gpu_fixture_ids"] == contract["fixture_ids"],
            "Embedded method or GPU fixture census differs from frozen source",
        )
        runtime.qualify(checkout, out, contract)
        reviews = verify_prerequisites(runtime, contract)
        verify_timing_reviews(reviews, contract, index)
        common.write(out / "actual-prerequisite-review-bindings.json", reviews)
        common.write(
            out / "timing-cohort-binding.json",
            {
                "cohort_id": COHORT_ID,
                "transport_amendment_sha256": TRANSPORT_AMENDMENT_SHA256,
                "reduction_clarification_sha256": REDUCTION_CLARIFICATION_SHA256,
                "raw_timing_result_max_bytes": MAX_TIMING_RESULT_BYTES,
                "closed_unpooled_job_ids": sorted(CLOSED_JOB_IDS),
                "scientific_acceptance": False,
            },
        )
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
        input_registry = verify_input_registry(contract, index)
        require(
            job
            not in {row["job_id"] for row in contract["timing_analysis_prior_results"]},
            "Current job cannot be an earlier analysis input",
        )
        inputs = verify_opaque_inputs(out, input_registry, common)
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
                "scope": "GPU_FIXTURES_AND_REGISTERED_PRODUCTION_TIMING_ONLY",
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
        for device_index in range(3):
            properties = torch.cuda.get_device_properties(device_index)
            require(
                "A100" in properties.name
                and (properties.major, properties.minor) == (8, 0),
                "Registered A100 sm_80 devices required",
            )
            devices.append(
                {
                    "index": device_index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "memory_bytes": properties.total_memory,
                }
            )
        common.command(out, "nvidia-smi", ["nvidia-smi"])
        expected_driver = contract["expected_nvidia_driver_version"]
        require(
            isinstance(expected_driver, str)
            and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", expected_driver) is not None,
            "Expected native NVIDIA driver is not sealed",
        )
        driver_rows = common.command(
            out,
            "nvidia-driver-version",
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        ).splitlines()
        require(
            len(driver_rows) == 3
            and all(row.strip() == expected_driver for row in driver_rows),
            "Allthree native GPU drivers must match the sealed version",
        )
        driver_version = expected_driver
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
                "nvidia_driver_version": driver_version,
                "timing_job_index": index,
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
                "scope": "BB_SELECTOR_GPU_DEV_TIMING_ONLY",
                "fixtures_passed": fixture["tests_run"],
                "scientific_acceptance": False,
            },
        )
        os.environ["ACI_MODEL_WORK_RELEASE"] = str(model_release)
        os.environ["ACI_CONTRACT_SHA256"] = common.sha(path)
        from ac_integrity.research import completion

        completion.source_allocation_guard()
        completion.cutoff()
        from selector_integration import run_timing

        timing = run_timing(out / "timing-study", contract)
        # This unwrapped base payload is immutable and written exactly once.
        common.write(out / "timing-study.json", timing)
        require(
            0 < (out / "timing-study.json").stat().st_size <= MAX_TIMING_RESULT_BYTES,
            "Current timing result exceeds its bounded input size",
        )
        validate_timing_result(timing, method, index)
        binding = method_binding(out, contract, timing, driver_version)
        common.write(out / "timing-method-binding.json", binding)
        base_sha = common.sha(out / "timing-study.json")
        if index == 2:
            completion.cutoff()
            records = analysis_records(out, contract, reviews, binding, job, common)
            from aggregate_timing import aggregate_timing

            pooled = aggregate_timing(records, method, contract["method_plan_sha256"])
            common.write(out / "timing-pooled-analysis.json", pooled)
            require(
                common.sha(out / "timing-study.json") == base_sha,
                "Analysis changed the immutable current result",
            )
            common.write(
                out / "analysis-validation-complete.json",
                {
                    "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
                    "input_jobs": [row["job_id"] for row in records],
                    "current_review_sha256": None,
                    "scientific_acceptance": False,
                },
            )
        common.write(
            out / "results.json",
            {
                "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
                "execution_complete": True,
                "timing_job_index": index,
                "fixture_count": fixture["tests_run"],
                "timed_arm_updates": 128,
                "production_profile_updates": 1,
                "total_updates_executed": 129,
                "timing_pairs": 8,
                "inputs": inputs,
                "cohort_id": COHORT_ID,
                "cohort_binding_file": "timing-cohort-binding.json",
                "timing_result_file": "timing-study.json",
                "timing_result_sha256": base_sha,
                "method_binding_file": "timing-method-binding.json",
                "pooled_analysis": "timing-pooled-analysis.json"
                if index == 2
                else "NOT_RUN_BEFORE_THREE_JOBS",
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
                failure_retention_admission(common, checkout, out, job, index)
            common.retain(out, campaign(index), OUTPUT_MAX_BYTES)
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
