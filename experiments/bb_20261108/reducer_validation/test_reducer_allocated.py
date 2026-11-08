"""Allocated fixture controls for the exact, unchanged three-job reducer.

Only RESULT_ROOT is sandboxed. The real allocation guard, measured bytes,
method, numerical operations and rejection guards remain unchanged.
"""

import copy
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import unittest


FIXTURE_IDS = [
    "test_reducer_allocated.ReducerFixtures.test_positive_same_three_payloads",
    "test_reducer_allocated.ReducerFixtures.test_missing_record",
    "test_reducer_allocated.ReducerFixtures.test_duplicate_job_id",
    "test_reducer_allocated.ReducerFixtures.test_wrong_raw_sha256",
    "test_reducer_allocated.ReducerFixtures.test_mismatched_runtime_binding",
    "test_reducer_allocated.ReducerFixtures.test_falsely_promoted_current",
]
PRIMARY_KEYS = {
    "contrast",
    "job_weighting",
    "mean_log_ratio",
    "sample_variance_log_ratio",
    "standard_error_log_ratio",
    "student_t_critical_95_df2",
    "log_interval_95",
    "geometric_mean_ratio",
    "ratio_interval_95",
    "direction",
    "elapsed_change_percent",
    "elapsed_change_percent_interval_95",
    "time_saving_percent",
    "time_saving_percent_interval_95",
    "percent_fields",
    "conditional_latency_gain_supported",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compare_primary(actual, expected, reducer):
    """Check all primary fields with the reducer's frozen numeric tolerance.

    Retain strict serialized equality separately from tolerant reproduction;
    metadata, key census, booleans and strings always require exact agreement.
    """
    require(set(actual) == set(expected) == PRIMARY_KEYS, "Primary key census differs")
    differences = []

    def compare(left, right, path):
        if type(right) in (int, float):
            valid = type(left) is type(right) and math.isclose(
                left,
                right,
                rel_tol=reducer.RELATIVE_TOLERANCE,
                abs_tol=reducer.ABSOLUTE_TOLERANCE,
            )
        elif isinstance(right, list):
            valid = isinstance(left, list) and len(left) == len(right)
            if valid:
                values = [
                    compare(x, y, path + "/" + str(i))
                    for i, (x, y) in enumerate(zip(left, right, strict=True))
                ]
                valid = all(values)
        else:
            valid = type(left) is type(right) and left == right
        exact = canonical(left) == canonical(right)
        if not exact:
            differences.append(
                {
                    "path": path,
                    "actual": left,
                    "expected": right,
                    "within_frozen_tolerance": valid,
                }
            )
        return valid

    outcomes = [
        compare(actual[key], expected[key], key) for key in sorted(PRIMARY_KEYS)
    ]
    return {
        "valid": all(outcomes),
        "strict_serialization_equal": canonical(actual) == canonical(expected),
        "differences": differences,
        "relative_tolerance": reducer.RELATIVE_TOLERANCE,
        "absolute_tolerance": reducer.ABSOLUTE_TOLERANCE,
        "complete_key_census": True,
    }


class ReducerFixtures(unittest.TestCase):
    context = None

    def _call(self, records):
        context = self.context
        return context["reducer"].aggregate_timing(
            records, context["method"], context["contract"]["method_plan_sha256"]
        )

    def _negative(self, name, records, expected_message):
        """Retain the exact mutation and observed error before asserting its guard."""
        context = self.context
        directory = context["out"] / name
        directory.mkdir(exist_ok=False)
        context["write"](directory / "input-records.json", records)
        error_type, error_message, returned_interval = None, None, False
        intended_class = False
        try:
            value = self._call(records)
            returned_interval = isinstance(value, dict) and "primary" in value
        except Exception as error:
            error_type, error_message = type(error).__name__, str(error)
            intended_class = type(error) is context["reducer"].TimingAggregationError
        context["write"](
            directory / "outcome.json",
            {
                "fixture_only": True,
                "expected_exception": "TimingAggregationError",
                "expected_message": expected_message,
                "observed_exception": error_type,
                "observed_message": error_message,
                "returned_interval": returned_interval,
                "intended_exception_class": intended_class,
                "scientific_acceptance": False,
            },
        )
        self.assertTrue(intended_class)
        self.assertEqual(error_type, "TimingAggregationError")
        self.assertEqual(error_message, expected_message)
        self.assertFalse(returned_interval)

    def test_positive_same_three_payloads(self):
        context = self.context
        directory = context["out"] / "positive"
        directory.mkdir(exist_ok=False)
        context["write"](directory / "input-records.json", context["records"])
        try:
            value = self._call(copy.deepcopy(context["records"]))
            comparison = compare_primary(
                value["primary"], context["expected_primary"], context["reducer"]
            )
        except Exception as error:
            context["write"](
                directory / "outcome.json",
                {
                    "fixture_only": True,
                    "exception": type(error).__name__,
                    "message": str(error),
                    "scientific_acceptance": False,
                },
            )
            raise
        context["write"](directory / "reproduced-primary.json", value["primary"])
        context["write"](
            directory / "outcome.json",
            {
                "fixture_only": True,
                "comparison": comparison,
                "data_origin": "The same three actual timing payloads; no new independent observation",
                "current_wrapper_job": os.environ["SLURM_JOB_ID"],
                "scientific_acceptance": False,
            },
        )
        self.assertTrue(comparison["valid"])
        self.assertEqual(value["n"], 3)
        self.assertEqual(value["df"], 2)

    def test_missing_record(self):
        records = copy.deepcopy(self.context["records"])
        del records[1]
        self._negative(
            "missing-record", records, "Exactly three complete job records are required"
        )

    def test_duplicate_job_id(self):
        records = copy.deepcopy(self.context["records"])
        records[1]["job_id"] = records[0]["job_id"]
        self._negative(
            "duplicate-job", records, "Three distinct decimal job IDs are required"
        )

    def test_wrong_raw_sha256(self):
        records = copy.deepcopy(self.context["records"])
        old = records[0]["evidence_attestation"]["result_sha256"]
        records[0]["evidence_attestation"]["result_sha256"] = (
            "0" if old[0] != "0" else "1"
        ) + old[1:]
        self._negative(
            "wrong-sha",
            records,
            "Raw result SHA256 differs from its parent attestation",
        )

    def test_mismatched_runtime_binding(self):
        records = copy.deepcopy(self.context["records"])
        old = records[1]["method_binding"]["runtime_sha256"]
        records[1]["method_binding"]["runtime_sha256"] = (
            "0" if old[0] != "0" else "1"
        ) + old[1:]
        self._negative(
            "mismatched-binding",
            records,
            "Numerical method bindings differ across jobs",
        )

    def test_falsely_promoted_current(self):
        records = copy.deepcopy(self.context["records"])
        records[2]["evidence_attestation"]["status"] = "ACTUAL_PROMOTED"
        self._negative(
            "falsely-promoted-current",
            records,
            "The third job must be current and must not fabricate an actual review",
        )


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started_ids, self.finished_ids, self.passed_ids = [], [], []

    def startTest(self, test):
        self.started_ids.append(test.id())
        super().startTest(test)

    def stopTest(self, test):
        self.finished_ids.append(test.id())
        super().stopTest(test)

    def addSuccess(self, test):
        self.passed_ids.append(test.id())
        super().addSuccess(test)


def run_allocated(out, contract, write, sha):
    """Copy authentic inputs, sandbox paths only, then run six fixed fixtures."""
    from e1_diagnostics import _allocated_only

    _allocated_only()
    out.mkdir(exist_ok=False)
    reducer = importlib.import_module("aggregate_timing")
    require(
        Path(reducer.__file__).resolve()
        == Path(__file__).resolve().parents[1] / "aggregate_timing.py",
        "Reducer import differs from pinned source",
    )
    require(contract["fixture_ids"] == FIXTURE_IDS, "Six fixture census differs")
    job = os.environ["SLURM_JOB_ID"]
    sandbox = out / "fixture-only-result-root"
    sandbox.mkdir(exist_ok=False)
    records, copies = [], []
    sources = contract["timing_validation_sources"]
    require(
        len(sources) == 3 and len({row["job_id"] for row in sources} | {job}) == 4,
        "Fixture must retain three distinct actual origins and its real CPU job",
    )
    for index, source in enumerate(sources):
        source_path = Path(source["result_path"])
        require(
            source_path.is_file()
            and not any(
                part.is_symlink() for part in (source_path, *source_path.parents)
            ),
            "Source timing path must be regular without symlink components",
        )
        require(
            source_path.stat().st_size == source["bytes"] <= reducer.MAX_RESULT_BYTES
            and sha(source_path) == source["sha256"],
            "Actual input pin differs",
        )
        fixture_job = source["job_id"] if index < 2 else job
        destination = (
            sandbox / "artifacts" / fixture_job / "bb-selector/timing-study.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source_path, destination)
        require(
            sha(destination) == source["sha256"],
            "Fixture copy differs from actual input",
        )
        records.append(
            {
                "job_id": fixture_job,
                "result_path": str(destination),
                "method_binding": source["method_binding"],
                "evidence_attestation": {
                    "status": "ACTUAL_PROMOTED"
                    if index < 2
                    else "CURRENT_COMPLETE_PENDING_REVIEW",
                    "job_id": fixture_job,
                    "result_sha256": source["sha256"],
                    "custody_sha256": source["custody_sha256"] if index < 2 else None,
                    "review_sha256": source["review_sha256"] if index < 2 else None,
                },
            }
        )
        copies.append(
            {
                "actual_origin_job": source["job_id"],
                "fixture_wrapper_job": fixture_job,
                "source_path": str(source_path),
                "fixture_path": str(destination),
                "sha256": source["sha256"],
                "bytes": source["bytes"],
            }
        )
    expected_input = next(
        row
        for row in contract["inputs"]
        if row["role"] == "actual-third-pooled-analysis"
    )
    expected_path = Path(expected_input["path"])
    require(
        expected_path.stat().st_size == expected_input["bytes"]
        and sha(expected_path) == expected_input["sha256"],
        "Expected pooled analysis pin differs",
    )
    expected_copy = out / "actual-third-pooled-analysis.json"
    shutil.copyfile(expected_path, expected_copy)
    require(
        sha(expected_copy) == expected_input["sha256"], "Expected analysis copy differs"
    )
    expected = json.loads(expected_copy.read_text())["primary"]
    write(
        out / "fixture-input-copies.json",
        {
            "copies": copies,
            "expected_analysis_sha256": expected_input["sha256"],
            "synthetic_fixture_paths_only": True,
            "independent_unit_added": False,
            "allocation_guard_replaced": False,
            "math_or_validation_guard_replaced": False,
            "scientific_acceptance": False,
        },
    )
    method = json.loads(
        (Path(__file__).resolve().parents[1] / "method-plan.json").read_text()
    )
    ReducerFixtures.context = {
        "out": out,
        "records": records,
        "reducer": reducer,
        "method": method,
        "contract": contract,
        "write": write,
        "expected_primary": expected,
    }
    original_root = reducer.RESULT_ROOT
    write(
        out / "sandbox-override.json",
        {
            "only_override": "aggregate_timing.RESULT_ROOT",
            "original": str(original_root),
            "fixture": str(sandbox),
            "fixture_only": True,
            "current_job_id": job,
            "current_attestation_review": None,
            "current_attestation_custody": None,
        },
    )
    holder, normal_exit = [], False

    def make_result(*args, **kwargs):
        value = RecordedResult(*args, **kwargs)
        holder.append(value)
        return value

    try:
        reducer.RESULT_ROOT = sandbox
        suite = unittest.TestLoader().loadTestsFromNames(FIXTURE_IDS)
        require(
            [test.id() for group in suite for test in group] == FIXTURE_IDS,
            "Loaded fixture census differs",
        )
        write(out / "fixture-request.json", {"test_ids": FIXTURE_IDS})
        with (out / "fixtures.log").open("x") as stream:
            result = unittest.TextTestRunner(
                stream=stream, verbosity=2, resultclass=make_result
            ).run(suite)
        normal_exit = True
    finally:
        reducer.RESULT_ROOT = original_root
        ReducerFixtures.context = None
        if holder:
            result = holder[0]
            record = {
                "fixture_only": True,
                "test_ids": FIXTURE_IDS,
                "started_ids": result.started_ids,
                "finished_ids": result.finished_ids,
                "passed_ids": result.passed_ids,
                "tests_run": result.testsRun,
                "failures": [
                    {"id": test.id(), "traceback": message}
                    for test, message in result.failures
                ],
                "errors": [
                    {"id": test.id(), "traceback": message}
                    for test, message in result.errors
                ],
                "skipped": [
                    {"id": test.id(), "reason": reason}
                    for test, reason in result.skipped
                ],
                "normal_exit": normal_exit,
                "root_restored": reducer.RESULT_ROOT == original_root,
                "scientific_acceptance": False,
            }
            write(out / "fixture-result.json", record)
    for item in copies:
        require(
            sha(item["fixture_path"]) == item["sha256"], "Fixture raw bytes changed"
        )
    require(
        normal_exit
        and result.wasSuccessful()
        and not result.skipped
        and result.testsRun == 6
        and result.started_ids
        == result.finished_ids
        == result.passed_ids
        == FIXTURE_IDS,
        "Exact six reducer fixtures did not all pass",
    )
    return record
