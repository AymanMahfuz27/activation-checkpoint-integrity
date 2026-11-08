"""Reduce the frozen three-job BB timing experiment on the admitted third job.

Import performs no I/O or numerical analysis. The parent authenticates actual review
and custody controls; this reducer checks their explicit attestations, reads and
hashes the exact raw result files, then recomputes the preregistered estimator.
It neither writes evidence nor promotes the current job's pending result.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


RESULT_ROOT = Path("/scratch/09999/aymanmahfuz/activation-checkpoint-integrity")
MAX_RESULT_BYTES = 4_194_304
METHOD_FILE_SHA256 = "86614b2a265e10cc2ae54e8a0d3fd49b6eaa83b17e55326eef9f158b2ccdf0bb"
METHOD_CONTENT_SHA256 = "8f6fb9ecc5616e680b4dc36f1666d162777be9704e02b2f673a70f43148b9f70"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
JOB_PATTERN = re.compile(r"[1-9][0-9]*\Z")
BINDING_SHA_FIELDS = (
    "method_plan_sha256", "selector_core_sha256", "selector_integration_sha256",
    "requirements_lock_sha256", "model_graph_sha256", "runtime_sha256",
)
RELATIVE_TOLERANCE = 1e-12
ABSOLUTE_TOLERANCE = 1e-12


class TimingAggregationError(ValueError):
    """A required input failed; no subset interval may replace this experiment."""


def _require(condition, message):
    if not condition:
        raise TimingAggregationError(message)


def _sha(value, label):
    _require(isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
             label + " must be a lowercase SHA256")
    return value


def _finite(value, label):
    _require(type(value) in (int, float), label + " must be a finite number, not a boolean")
    try:
        result = float(value)
    except OverflowError as error:
        raise TimingAggregationError(label + " is outside the finite machine range") from error
    _require(math.isfinite(result), label + " must be finite")
    return result


def _number(value, label, *, positive=True):
    result = _finite(value, label)
    _require(result > 0 if positive else result >= 0,
             label + (" must be finite positive" if positive else " must be finite nonnegative"))
    return result


def _count(value, expected, label):
    _require(type(value) is int and value == expected, label + " has the wrong integer census")


def _close(recorded, recomputed, label):
    _require(math.isclose(_finite(recorded, label), _finite(recomputed, label),
                         rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE),
        label + " disagrees with its raw timing values")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise TimingAggregationError("Nonfinite JSON constant: " + value)


def _read_result(wrapper):
    """Read a bounded regular file and bind its exact bytes before JSON parsing."""
    job = wrapper["job_id"]
    expected = RESULT_ROOT / "artifacts" / job / "bb-selector" / "timing-study.json"
    _require(type(wrapper["result_path"]) is str and wrapper["result_path"] == str(expected),
             "Result path differs from the exact declared job path")
    for path in (expected, *expected.parents):
        _require(not path.is_symlink(), "A result path component is a symlink: " + str(path))
    try:
        descriptor = os.open(expected, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode), "Timing result is not a regular file")
            _require(0 < before.st_size <= MAX_RESULT_BYTES, "Timing result size is outside its bound")
            raw = stream.read(MAX_RESULT_BYTES + 1)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise TimingAggregationError("Cannot read declared timing result: " + str(expected)) from error
    def identity(entry):
        return (entry.st_dev, entry.st_ino, entry.st_size, entry.st_mtime_ns, entry.st_ctime_ns)
    _require(identity(before) == identity(after) and len(raw) == before.st_size,
             "Timing result changed while being read")
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == _sha(wrapper["evidence_attestation"]["result_sha256"], "result_sha256"),
             "Raw result SHA256 differs from its parent attestation")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TimingAggregationError("Timing result is not strict UTF8 JSON") from error
    _require(isinstance(result, dict), "Timing result must be a JSON object")
    return result, {"path": str(expected), "sha256": digest, "bytes": len(raw)}


def _binding(value, recipe, method_sha):
    _require(isinstance(value, dict) and set(value) == set(BINDING_SHA_FIELDS) | {"production_recipe"},
             "Method binding must contain exactly six SHA fields and production_recipe")
    for field in BINDING_SHA_FIELDS:
        _sha(value[field], "method_binding." + field)
    _require(value["method_plan_sha256"] == method_sha and value["production_recipe"] == recipe,
             "Method binding differs from the frozen method or production recipe")
    return value


def _exact_parity(value):
    return (isinstance(value, dict) and value.get("valid") is True
            and value.get("byte_equal") is True and value.get("all_finite") is True
            and value.get("changed_scalar_paths") == [])


def _qualified_counters(value):
    _require(isinstance(value, dict) and value.get("algorithm") == "blake3-256-v1"
             and value.get("backend") == "full" and value.get("include_sketch") is False
             and value.get("canonical_clone_charged") is True
             and value.get("backward") == {"gelu": 384, "softmax": 384, "native_layer_norm": 800},
             "Production observer/fingerprint metadata is unqualified")
    _count(value.get("root_forwards"), 32, "production observer root forwards")
    for field in ("fingerprints", "fingerprint_payload_bytes"):
        _require(type(value.get(field)) is int and value[field] > 0, "Empty production counter: " + field)


def _profile(result):
    """Authenticate the full measured admission cost and separate profile bytes."""
    cost = result["profile_cost"]
    _require(isinstance(cost, dict), "Profile cost is missing")
    call = _number(cost["complete_profile_call_wall_seconds"], "full profile call seconds")
    binding = _number(cost["immutable_identity_binding_setup_seconds"], "binding setup seconds", positive=False)
    census = _number(cost["model_census_admission_seconds"], "census admission seconds", positive=False)
    _close(cost["profile_and_identity_binding_seconds"], call + binding, "profile plus binding cost")
    total = call + binding + census
    _close(cost["complete_profile_admission_seconds"], total, "complete profile admission cost")
    _close(result["production_profile_elapsed_seconds"], total, "reported production profile cost")
    step = _number(result["production_profile_step_elapsed_seconds"], "profile step seconds")
    _require(step <= call or math.isclose(step, call, rel_tol=RELATIVE_TOLERANCE,
                                         abs_tol=ABSOLUTE_TOLERANCE), "Profile step exceeds full profile call")
    for field in ("exported_profile_bytes", "exported_profile_files"):
        _require(type(cost[field]) is int and cost[field] > 0, "Missing profile export census: " + field)
    _require(cost["includes_raw_profile_and_arm_serialization"] is True,
             "Raw profiling export cost was excluded")
    counters = result["production_profile_counters"]
    _qualified_counters(counters)
    return {"full_profile_admission_seconds": total, "profile_step_seconds": step,
            "fingerprint_payload_bytes": counters["fingerprint_payload_bytes"],
            "exported_profile_bytes": cost["exported_profile_bytes"],
            "exported_profile_files": cost["exported_profile_files"],
            "exported_profile_scope": cost["exported_profile_scope"], "profile_updates": 1}


def _validate_result(result, protocol, job_index):
    """Validate all eight pairs, then recompute the direct two-block contrast."""
    _require(result.get("experiment_id") == "BB-DEMAND-20261108-v1"
             and result.get("phase") == "TIMING"
             and result.get("status") == "COMPLETE_PENDING_INDEPENDENT_REVIEW"
             and result.get("execution_complete") is True
             and result.get("independent_unit") == "Slurm job"
             and result.get("scientific_acceptance") is False, "Incomplete or wrong timing result")
    _require(not {"job_id", "result_path", "method_binding", "evidence_attestation"} & set(result),
             "Raw timing payload must not contain wrapper or self-attestation metadata")
    for field, expected_count in (("updates_executed", 128), ("timed_arm_updates", 128),
                                   ("production_profile_updates", 1), ("total_updates_executed", 129)):
        _count(result.get(field), expected_count, field)
    _require(result.get("updates_executed_scope") == "timed arms only", "Timing/profile update scope differs")
    _require(result.get("profile_cost_is_not_free") is True
             and result.get("no_pooled_interval_from_one_job") is True
             and result.get("primary_estimator") == protocol["estimator"], "Timing estimator/cost declaration differs")
    _sha(result["production_plan_sha256"], "production_plan_sha256")
    census = result["model_census"]
    _require(census.get("status") == "PASS" and census.get("validation_scope") == "COMPILED_MODEL_PLAN"
             and census.get("outside_binding_map_validated") is True
             and census.get("recipe") == protocol["production_batch"]
             and census.get("checks") and all(value is True for value in census["checks"].values())
             and len(census["outside_bindings"]) == 96
             and all(row.get("valid") is True for row in census["outside_bindings"]), "Production profile census is unqualified")
    expected = [row for row in protocol["frozen_pair_schedule"] if row["job_index"] == job_index]
    pairs = result["pairs"]
    _require(isinstance(pairs, list) and len(pairs) == len(expected) == 8, "Exactly eight timing pairs are required")
    checked_totals, pair_rows = {}, []
    for position, (pair, scheduled) in enumerate(zip(pairs, expected, strict=True)):
        policy, block = scheduled["policy"], scheduled["block_index"]
        _count(pair.get("schedule_position"), position, "schedule_position")
        _count(pair.get("block_index"), block, "block_index")
        _require(pair.get("policy") == policy and pair.get("arm_order") == scheduled["arm_order"],
                 "Timing pairs do not follow the frozen schedule")
        arms = pair["arms"]
        _require(set(arms) == {"disabled", policy}, "Timing pair arm census differs")
        parity = pair["final_eight_update_parity"]
        _require(set(parity) == {"state", "gradients"} and all(_exact_parity(value) for value in parity.values())
                 and pair.get("final_gradient_phase") == "post-clipping", "Eight-update state/gradient/RNG parity failed")
        totals, arm_rows = {}, {}
        for label in scheduled["arm_order"]:
            arm = arms[label]
            if label == "disabled":
                _require(arm.get("last_observer_counters") is None, "Disabled arm unexpectedly contains observer counters")
            else:
                _qualified_counters(arm.get("last_observer_counters"))
            _require(isinstance(arm["warmup_seconds"], list) and len(arm["warmup_seconds"]) == 2
                     and isinstance(arm["measured_seconds"], list) and len(arm["measured_seconds"]) == 6,
                     "Every timing arm requires two warmups and six measured updates")
            warm = [_number(value, "warmup seconds") for value in arm["warmup_seconds"]]
            measured = [_number(value, "measured seconds") for value in arm["measured_seconds"]]
            total = math.fsum(measured)
            _number(total, "recomputed measured total")
            _close(arm["measured_elapsed_seconds"], total, "recorded measured total")
            totals[label] = total
            arm_rows[label] = {"warmup_seconds": warm, "measured_seconds": measured,
                               "measured_elapsed_seconds": total}
        secondary = totals[policy] / totals["disabled"]
        _close(pair["secondary_checked_disabled_elapsed_ratio"], secondary, "secondary own-disabled ratio")
        checked_totals[(policy, block)] = totals[policy]
        pair_rows.append({"schedule_position": position, "policy": policy, "block_index": block,
                          "arm_order": scheduled["arm_order"], "arms": arm_rows,
                          "secondary_own_disabled_ratio": secondary, "parity_valid": True})
    recorded_blocks = result["primary_direct_blocks"]
    _require(isinstance(recorded_blocks, list) and len(recorded_blocks) == 2,
             "Exactly two recorded direct blocks are required")
    blocks = []
    for block, recorded in enumerate(recorded_blocks):
        consumer = checked_totals[("selected_consumer", block)]
        full_io = checked_totals[("full_forward_io", block)]
        ratio = _number(consumer / full_io, "direct consumer/full-I/O ratio")
        log_ratio = math.log(ratio)
        _count(recorded.get("block_index"), block, "recorded direct block_index")
        for field, recomputed in (("consumer_elapsed_seconds", consumer), ("full_io_elapsed_seconds", full_io),
                                  ("direct_consumer_full_io_ratio", ratio), ("log_ratio", log_ratio)):
            _close(recorded[field], recomputed, "recorded direct block " + field)
        blocks.append({"block_index": block, "consumer_elapsed_seconds": consumer,
                       "full_io_elapsed_seconds": full_io, "direct_ratio": ratio, "log_ratio": log_ratio,
                       "seconds_saved_per_update": (full_io - consumer) / 6})
    mean_log = math.fsum(row["log_ratio"] for row in blocks) / 2
    _close(result["primary_job_mean_log_ratio"], mean_log, "recorded job mean log ratio")
    return {"job_index": job_index, "timing_pairs": pair_rows, "primary_blocks": blocks,
            "mean_log_ratio": mean_log,
            "mean_seconds_saved_per_update": math.fsum(row["seconds_saved_per_update"] for row in blocks) / 2,
            "profile": _profile(result)}


def aggregate_timing(records, method_plan, method_plan_sha256):
    """Return the frozen direct three-job estimator and conditional amortization.

    The parent passes three wrappers naming exact raw results plus separately
    authenticated method/evidence bindings. Validate all inputs before pooling;
    reject the entire reduction on any missing, invalid or substituted job.
    Only job index2 may be current, and its result remains pending actual review.
    """
    from e1_diagnostics import _allocated_only
    _allocated_only()
    _require(_sha(method_plan_sha256, "method_plan_sha256") == METHOD_FILE_SHA256,
             "The reducer requires the exact frozen method file")
    canonical = json.dumps(method_plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    _require(hashlib.sha256(canonical).hexdigest() == METHOD_CONTENT_SHA256, "Method content differs from its frozen identity")
    _require(isinstance(records, list) and len(records) == 3, "Exactly three complete job records are required")
    protocol = method_plan["timing_protocol"]
    jobs, indices, rows, common_binding = set(), set(), [], None
    current_job = os.environ["SLURM_JOB_ID"]
    for expected_index, wrapper in enumerate(records):
        _require(isinstance(wrapper, dict) and set(wrapper) == {
            "job_id", "result_path", "method_binding", "evidence_attestation"}, "Invalid timing input wrapper")
        job = wrapper["job_id"]
        _require(isinstance(job, str) and JOB_PATTERN.fullmatch(job) is not None and job not in jobs,
                 "Three distinct decimal job IDs are required")
        jobs.add(job)
        binding = _binding(wrapper["method_binding"], protocol["production_batch"], method_plan_sha256)
        if common_binding is None:
            common_binding = binding
        _require(binding == common_binding, "Numerical method bindings differ across jobs")
        attestation = wrapper["evidence_attestation"]
        _require(isinstance(attestation, dict) and set(attestation) == {
            "status", "job_id", "result_sha256", "custody_sha256", "review_sha256"}
            and attestation["job_id"] == job, "Evidence attestation is incomplete or names another job")
        result, raw_evidence = _read_result(wrapper)
        index = result.get("independent_job_index")
        _require(type(index) is int and index == expected_index and index not in indices,
                 "Exactly one result for each ordered index0,1,2 is required")
        indices.add(index)
        if index == 2:
            _require(job == current_job and attestation["status"] == "CURRENT_COMPLETE_PENDING_REVIEW"
                     and attestation["custody_sha256"] is None and attestation["review_sha256"] is None,
                     "The third job must be current and must not fabricate an actual review")
        else:
            _require(attestation["status"] == "ACTUAL_PROMOTED", "Earlier jobs require actual promoted attestations")
            _sha(attestation["custody_sha256"], "prior custody_sha256")
            _sha(attestation["review_sha256"], "prior review_sha256")
        row = _validate_result(result, protocol, index)
        row.update(job_id=job, raw_result=raw_evidence, evidence_attestation=dict(attestation))
        rows.append(row)
    _require(indices == {0, 1, 2}, "A required independent job is absent")
    rows.sort(key=lambda row: row["job_index"])
    mean = math.fsum(row["mean_log_ratio"] for row in rows) / 3
    variance = math.fsum((row["mean_log_ratio"] - mean) ** 2 for row in rows) / 2
    _number(variance, "sample variance", positive=False)
    standard_error = math.sqrt(variance / 3)
    # For df2, F(t)=1/2+t/(2*sqrt(t*t+2)); F(t)=.975 gives this exact inverse.
    critical = math.sqrt(2) * 0.95 / math.sqrt(1 - 0.95 ** 2)
    _close(0.5 + critical / (2 * math.sqrt(critical * critical + 2)), 0.975,
           "analytic df2 Student-t quantile CDF check")
    low_log, high_log = mean - critical * standard_error, mean + critical * standard_error
    try:
        point = _number(math.exp(mean), "geometric mean ratio")
        low, high = _number(math.exp(low_log), "lower ratio limit"), _number(math.exp(high_log), "upper ratio limit")
    except OverflowError as error:
        raise TimingAggregationError("The interval exceeds finite machine exponentiation; no interval reported") from error
    direction = "FASTER" if high < 1 else "SLOWER" if low > 1 else "INCONCLUSIVE"
    elapsed_percent = _finite(100 * (point - 1), "elapsed-change percent")
    elapsed_percent_interval = [_finite(100 * (bound - 1), "elapsed-change percent interval")
                               for bound in (low, high)]
    saving_percent = _finite(100 * (1 - point), "time-saving percent")
    saving_percent_interval = [_finite(100 * (1 - bound), "time-saving percent interval")
                              for bound in (high, low)]
    mean_saved = math.fsum(row["mean_seconds_saved_per_update"] for row in rows) / 3
    mean_profile = math.fsum(row["profile"]["full_profile_admission_seconds"] for row in rows) / 3
    amortization = None
    if mean_saved > 0:
        amortization = _number(mean_profile / mean_saved, "conditional point amortization updates")
    return {
        "experiment_id": method_plan["experiment_id"], "phase": "THREE_JOB_TIMING_ANALYSIS",
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW", "execution_complete": True,
        "scientific_acceptance": False,
        "method_plan_sha256": method_plan_sha256, "method_binding": common_binding,
        "current_job_id": current_job, "current_job_actual_review_complete": False,
        "parent_attestations_present": True, "parent_authentication_required": True,
        "parent_authentication_performed_by_reducer": False,
        "parent_authentication_boundary": "Parent authenticates review/custody contents; reducer verifies supplied attestations and exact raw result bytes.",
        "jobs": rows, "independent_unit": "Slurm job", "n": 3, "df": 2,
        "primary": {"contrast": "DIRECT selected_consumer/full_forward_io; no disabled normalization",
                    "job_weighting": "equal; two log block ratios averaged within each job",
                    "mean_log_ratio": mean, "sample_variance_log_ratio": variance,
                    "standard_error_log_ratio": standard_error, "student_t_critical_95_df2": critical,
                    "log_interval_95": [low_log, high_log], "geometric_mean_ratio": point,
                    "ratio_interval_95": [low, high], "direction": direction,
                    "elapsed_change_percent": elapsed_percent,
                    "elapsed_change_percent_interval_95": elapsed_percent_interval,
                    "time_saving_percent": saving_percent,
                    "time_saving_percent_interval_95": saving_percent_interval,
                    "percent_fields": "Monotone presentation transforms of the same primary ratio and interval; no new endpoint.",
                    "conditional_latency_gain_supported": direction == "FASTER"},
        "point_amortization": {
            "full_profile_cost_charged_to": "selected_consumer alone; no baseline profile credit",
            "mean_full_profile_admission_seconds": mean_profile, "mean_seconds_saved_per_update": mean_saved,
            "conditional_updates_to_amortize": amortization,
            "positive_point_savings_required": True, "point_estimate_is_not_a_confidence_bound": True,
            "latency_cost_win_supported_by_primary_interval": direction == "FASTER",
            "no_cost_win_when_primary_interval_inconclusive": True},
        "update_census": {"timed_arm_updates": 384, "profiling_updates": 3, "total_updates": 387},
        "limits": [
            "Exactly three independent jobs and df2 give a wide, strongly model-dependent small-sample interval.",
            "Student-t inference assumes independent approximately normal job mean log ratios; blocks/updates are not extra independent units.",
            "This is machine-program elapsed timing for the frozen model, software, recipe and hardware, not a population or learning guarantee.",
            "Finite Python binary64 arithmetic and the stated consistency tolerance bound the calculation; overflow/underflow fails closed.",
            "Amortization uses all three point savings including negative jobs; it is conditional, charges the entire profile to the candidate and grants no gain when the primary interval is inconclusive.",
            "The current third-job result and this pooled analysis still require independent actual review and custody.",
        ],
        "record_consistency_tolerance": {"relative": RELATIVE_TOLERANCE, "absolute": ABSOLUTE_TOLERANCE},
        "missing_or_invalid_input_policy": "Fail the full reduction; no subset, replacement or complete-case interval.",
    }
