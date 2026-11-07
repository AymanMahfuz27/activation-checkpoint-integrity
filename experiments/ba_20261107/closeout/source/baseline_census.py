"""Allocated-only census of frozen E3 cheap checks and scoped ABFT evidence.

No model runs or products are recomputed. ABFT calibration products are retained
objects, not journaled detector decisions. Native calibration residual maxima
were not exported; the report keeps that limit explicit.
"""

from collections import Counter
import hashlib
import json
import math

from analysis_core import require_allocation
from analysis_original import equal


PREFIX = "E3/baselines/"
PRODUCTS = {
    "forward": [2048, 768, 3072],
    "input_gradient": [2048, 3072, 768],
    "weight_gradient": [3072, 2048, 768],
}
KINDS = {
    "calibration": 30,
    "clean-test": 30,
    "post-product-bit": 30,
    "balanced-cancellation": 1,
    "common-mode-operand": 1,
}
CHEAP_DECISIONS = (
    "missing_gradient_alert",
    "empty_gradient_alert",
    "nonfinite_gradient_alert",
    "nonfinite_loss_alert",
    "norm_alert",
    "layer_norm_alert",
    "loss_spike_alert",
    "clipping_would_activate",
)


def _finite(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _journal(reader, name):
    value = reader.journal(PREFIX + name)
    if value.get("status") == "NOT_RECORDED" or value["partial_final_write"]:
        raise ValueError("Complete baseline journal missing/partial: " + name)
    return value["rows"]


def _cheap_decisions(row):
    """Validate recorded decisions against saved scalar evidence, without rerun."""
    spec = row["spec"]
    bounds = dict(spec["norm_upper_bounds"])
    layers = dict(spec["layer_norm_upper_bounds"])
    groups = dict(spec["layer_groups"])
    complete = set(bounds) == set(spec["expected_names"])
    layer_complete = bool(groups) and set(layers) == set(groups)
    norm_exceeded = [
        name
        for name, bound in spec["norm_upper_bounds"]
        if row["gradient_norms"][name] > bound
    ]
    layer_exceeded = [
        name
        for name, bound in spec["layer_norm_upper_bounds"]
        if row["layer_norms"][name] > bound
    ]
    expected = {
        "missing_gradient_alert": bool(row["missing_gradient_names"]),
        "empty_gradient_alert": bool(row["empty_gradient_names"]),
        "nonfinite_gradient_alert": bool(row["nonfinite_gradient_names"]),
        "nonfinite_loss_alert": not math.isfinite(row["loss"]),
        "norm_alert": bool(norm_exceeded) if complete else None,
        "layer_norm_alert": bool(layer_exceeded) if layer_complete else None,
        "loss_spike_alert": row["loss"] > spec["loss_upper_bound"]
        if spec["loss_upper_bound"] is not None
        else None,
    }
    auxiliary = {
        "norm_upper_bounds": bounds,
        "norm_bounds_cover_all_expected_gradients": complete,
        "norm_limit_exceeded_names": norm_exceeded,
        "layer_groups": {key: list(value) for key, value in groups.items()},
        "layer_norm_upper_bounds": layers,
        "layer_norm_bounds_complete": layer_complete,
        "layer_norm_limit_exceeded": layer_exceeded,
        "loss_upper_bound": spec["loss_upper_bound"],
        "clipping_limit": spec["clipping_limit"],
        "global_norm_covers_all_expected_gradients": True,
    }
    auxiliary_issues = [
        key for key, value in auxiliary.items() if key not in row or row[key] != value
    ]
    norm = row.get("global_gradient_norm")
    clipping_supported = (
        _finite(norm)
        and row.get("present_gradient_norm") == norm
        and row.get("global_norm_finite") is True
    )
    expected["clipping_would_activate"] = (
        norm > spec["clipping_limit"]
        if clipping_supported and spec["clipping_limit"] is not None
        else None
    )
    decisions = {}
    for key, value in expected.items():
        relevant = [
            field
            for field in auxiliary_issues
            if (key == "norm_alert" and field.startswith("norm_"))
            or (key == "layer_norm_alert" and field.startswith("layer_"))
            or (key == "loss_spike_alert" and field == "loss_upper_bound")
            or (
                key == "clipping_would_activate"
                and field
                in ("clipping_limit", "global_norm_covers_all_expected_gradients")
            )
        ]
        unsupported = key == "clipping_would_activate" and not clipping_supported
        decisions[key] = {
            "recorded": row.get(key),
            "expected_from_saved_scalars": value,
            "status": "UNSUPPORTED_MISSING_SCALAR_EVIDENCE"
            if unsupported
            else "MISSING_RECORDED_FIELD"
            if key not in row
            else "CONTRADICTORY_RECORDED_DECISION"
            if row[key] is not value or relevant
            else "PASS",
            "auxiliary_contradictions": relevant,
        }
    return {
        "update": row["update"],
        "phase": row["phase"],
        "decisions": decisions,
        "auxiliary_issues": auxiliary_issues,
    }


def _clean_stage(reader, name, targets, initial, input_sha):
    root = name + "/"
    thresholds = reader.json(PREFIX + root + "thresholds.json")
    completion = reader.json(PREFIX + root + "clean-completion.json")
    if (
        thresholds.get("status") != "FROZEN_CLEAN_CALIBRATION"
        or thresholds.get("stage") != "early-warm10"
        or thresholds.get("source_input_sha256") != input_sha
        or thresholds.get("batch_targets") != targets
        or thresholds.get("calibration_updates") != list(range(1, 31))
        or thresholds.get("test_updates") != list(range(31, 61))
        or completion.get("status") != "COMPLETE_CLEAN_CALIBRATION_AND_DISJOINT_TEST"
        or completion.get("threshold_sha256")
        != reader.members[PREFIX + root + "thresholds.json"]["sha256"]
    ):
        raise ValueError("Clean baseline threshold/completion lineage differs")
    spec = thresholds["spec"]
    names = spec["expected_names"]
    groups = dict(spec["layer_groups"])
    if (
        not names
        or len(set(names)) != len(names)
        or not groups
        or len(groups) != len(spec["layer_groups"])
        or Counter(member for members in groups.values() for member in members)
        != Counter(names)
        or spec.get("clipping_limit") != 1.0
    ):
        raise ValueError("Frozen baseline gradient/layer population invalid")
    updates = _journal(reader, root + "updates.jsonl")
    cheap = _journal(reader, root + "cheap-clean.jsonl")
    if len(updates) != 60 or len(cheap) != 60:
        raise ValueError("Clean baseline requires60 updates and60 observations")
    for index, (row, observed) in enumerate(zip(updates, cheap), 1):
        update = row.get("update", {})
        if (
            row.get("branch_update") != index
            or row.get("status") != "ACCEPTED"
            or row.get("phase") != "ACCEPTED"
            or row.get("policy") != "disabled"
            or row.get("preserve_rng") is not True
            or row.get("spec") is not None
            or row.get("injections") != []
            or update.get("update") != initial["updates"] + index
            or update.get("accepted_tokens") != initial["cursor"] + index * targets
            or update.get("tokens_this_update") != targets
            or observed.get("update") != index
            or observed.get("phase")
            != ("calibration" if index <= 30 else "disjoint-test")
            or observed.get("expected_gradient_count") != len(names)
            or observed.get("present_gradient_count") != len(names)
            or observed.get("missing_gradient_names") != []
            or observed.get("empty_gradient_names") != []
            or observed.get("nonfinite_gradient_names") != []
            or observed.get("global_norm_finite") is not True
            or not _finite(observed.get("loss"))
            or not _finite(observed.get("seconds"), positive=True)
            or observed.get("spec_sha256") != _sha(observed.get("spec"))
            or observed.get("spec", {}).get("expected_names") != names
        ):
            raise ValueError("Invalid clean baseline attempt/observation/counter")
        if index > 30 and observed["spec"] != spec:
            raise ValueError("Disjoint clean test changed frozen thresholds")
        if set(observed.get("gradient_norms", {})) != set(names) or set(
            observed.get("layer_norms", {})
        ) != set(groups):
            raise ValueError("Gradient/layer norm population missing/foreign")
        if any(
            not _finite(value)
            for value in (
                *observed["gradient_norms"].values(),
                *observed["layer_norms"].values(),
            )
        ):
            raise ValueError("Nonfinite clean calibration/test norm")
    eps = 2.0**-23
    expected_parameter = {
        key: 4 * max(row["gradient_norms"][key] for row in cheap[:30]) + eps
        for key in names
    }
    expected_layer = {
        key: 4 * max(row["layer_norms"][key] for row in cheap[:30]) + eps
        for key in groups
    }
    if (
        dict(spec["norm_upper_bounds"]) != expected_parameter
        or dict(spec["layer_norm_upper_bounds"]) != expected_layer
        or spec["loss_upper_bound"] != 4 * max(row["loss"] for row in cheap[:30]) + eps
    ):
        raise ValueError("Cheap threshold values differ from frozen calibration rule")
    test = cheap[30:]
    checks = [_cheap_decisions(row) for row in cheap]
    decisions = {
        key: {
            "true": sum(
                row.get(key) is True and checked["decisions"][key]["status"] == "PASS"
                for row, checked in zip(test, checks[30:])
            ),
            "false": sum(
                row.get(key) is False and checked["decisions"][key]["status"] == "PASS"
                for row, checked in zip(test, checks[30:])
            ),
            "unsupported_or_invalid": sum(
                checked["decisions"][key]["status"] != "PASS"
                or type(row.get(key)) is not bool
                for row, checked in zip(test, checks[30:])
            ),
            "recorded_true": sum(row.get(key) is True for row in test),
            "recorded_false": sum(row.get(key) is False for row in test),
            "contradictions": sum(
                checked["decisions"][key]["status"] == "CONTRADICTORY_RECORDED_DECISION"
                for checked in checks[30:]
            ),
            "denominator": 30,
        }
        for key in CHEAP_DECISIONS
    }
    return {
        "stage": name,
        "targets_per_update": targets,
        "registered_clean_updates": 60,
        "updates": updates,
        "cheap_observations": cheap,
        "cheap_decision_validation": checks,
        "disjoint_clean_test_decisions": decisions,
        "thresholds": thresholds,
        "threshold_rule_validated": True,
        "alert_interpretation": "Observed clean alerts; clipping activation alone is not an incorrect gradient",
    }


def _check(row, tolerance):
    for field in ("finite", "detected"):
        if type(row.get(field)) is not bool:
            raise ValueError("Scoped checksum decision lacks boolean evidence")
    if row.get("atol") != tolerance["atol"] or row.get("rtol") != 0.0:
        raise ValueError("Checksum decision differs from frozen role tolerance")
    finite = row["finite"]
    residuals = [row.get("row_max_residual"), row.get("column_max_residual")]
    if finite:
        if any(not _finite(value) for value in residuals) or row["detected"] != (
            max(residuals) > tolerance["atol"]
        ):
            raise ValueError("Checksum residual/decision contradicts tolerance")
    elif row["detected"] is not True or any(value is not None for value in residuals):
        raise ValueError("Nonfinite checksum outcome mislabeled")


def analyze_baselines(reader, registration, initial, input_sha):
    require_allocation()
    registered = {row["id"]: row for row in registration}
    if (
        len(registered) != 276
        or len(registration) != 276
        or Counter((row["role"], row["kind"]) for row in registration)
        != Counter(
            {(role, kind): count for role in PRODUCTS for kind, count in KINDS.items()}
        )
    ):
        raise ValueError("Scoped ABFT registration differs from fixed276-event census")
    if any(row["shape_mkn"] != PRODUCTS[row["role"]] for row in registration):
        raise ValueError("ABFT registered production shapes differ")
    stages = {
        name: _clean_stage(reader, name, targets, initial, input_sha)
        for name, targets in (
            ("early-warm10", 8192),
            ("production-early-warm10", 65536),
        )
    }
    ready = reader.json(PREFIX + "diagnostic-ready.json")
    if (
        ready.get("status") != "FROZEN_CLEAN_CALIBRATION"
        or ready.get("batch_targets") != 8192
        or ready.get("new_clean_updates") != 60
        or ready.get("by_state", {}).get("early-warm10")
        != stages["early-warm10"]["thresholds"]
        or ready.get("test_receipts", {}).get("early-warm10")
        != reader.members[PREFIX + "early-warm10/clean-completion.json"]["sha256"]
    ):
        raise ValueError("Diagnostic readiness differs from actual clean baseline")
    parity = reader.json(PREFIX + "operand-capture-parity.json")
    parity_valid = (
        parity.get("captured_keys") == ["dy", "weight", "x"]
        and all(
            parity.get(name, {}).get("status") == "ACCEPTED"
            for name in ("plain", "observed")
        )
        and equal(parity.get("full_state", {}))
        and equal(parity.get("gradients", {}))
        and bool(parity.get("full_state", {}).get("tensors"))
        and bool(parity.get("gradients", {}).get("tensors"))
    )
    tolerances = reader.json(PREFIX + "production-early-warm10/abft-tolerances.json")
    if set(tolerances) != set(PRODUCTS) or any(
        not _finite(value.get("atol"), positive=True) or value.get("rtol") != 0.0
        for value in tolerances.values()
    ):
        raise ValueError("Invalid frozen per-role checksum tolerance")
    clean = _journal(reader, "production-early-warm10/abft-clean-tests.jsonl")
    faults = _journal(reader, "production-early-warm10/abft-faults.jsonl")
    observations = {}
    for row in clean:
        identity = f"E3-ABFT-{row['role']}-clean-{row['update']:02}"
        if (
            identity not in registered
            or identity in observations
            or registered[identity]["kind"] != "clean-test"
        ):
            raise ValueError("Foreign/duplicate disjoint ABFT clean test")
        role = row["role"]
        m, k, n = PRODUCTS[role]
        if row.get("shape") != {"left": [m, k], "right": [k, n]} or not _finite(
            row.get("seconds"), positive=True
        ):
            raise ValueError("ABFT clean actual operand shape/cost differs")
        _check(row, tolerances[role])
        observations[identity] = row
    for row in faults:
        case = row["case"]
        identity = case["id"]
        if (
            identity not in registered
            or identity in observations
            or case != registered[identity]
            or case["kind"] in ("clean-test", "calibration")
        ):
            raise ValueError("Foreign/duplicate/mismatched ABFT fault event")
        role, kind = case["role"], case["kind"]
        if row.get("dtype") != "torch.float32" or row.get("tf32") is not False:
            raise ValueError("ABFT production precision metadata differs")
        _check(row["abft"], tolerances[role])
        if any(
            not _finite(row.get(key), positive=True)
            for key in ("abft_seconds", "paired_digest_seconds", "nonfinite_seconds")
        ) or any(
            type(row.get(key)) is not bool
            for key in ("paired_digest_detected", "nonfinite_detected")
        ):
            raise ValueError("ABFT detector cost/decision incomplete")
        if kind == "post-product-bit" and any(
            row.get("fault", {}).get(key) != case[key] for key in ("element", "bit")
        ):
            raise ValueError("ABFT injected post-product bit differs from registration")
        if kind == "common-mode-operand" and any(
            row.get("fault", {}).get(key) != value
            for key, value in {
                "element": 0,
                "bit": 31,
                "phase": "before_both_executions",
                "lifetime": "persistent",
            }.items()
        ):
            raise ValueError("Common-mode operand intervention differs")
        observations[identity] = row
    outputs = []
    for identity, case in registered.items():
        root = PREFIX + "production-early-warm10/"
        update, role = case["production_clean_update"], case["role"]
        required_members = [
            root + f"operands-{update:02}.pt",
            root + f"product-{role}-{update:02}.pt",
        ]
        present = all(name in reader.members for name in required_members)
        observation = observations.get(identity)
        outputs.append(
            {
                "case": case,
                "status": "CALIBRATION_OBJECT_RETAINED_NO_JOURNALED_DECISION"
                if case["kind"] == "calibration" and present
                else "OBSERVED"
                if observation is not None and present
                else "MISSING_REQUIRED_EVIDENCE",
                "observation": observation,
                "operand_product_members": required_members,
                "required_members_present": present,
                "operand_capture_control_valid": parity_valid,
            }
        )
    total = reader.json(PREFIX + "completion.json")
    abft_total = reader.json(PREFIX + "production-early-warm10/abft-completion.json")
    if (
        len(observations) != 186
        or any(row["status"] == "MISSING_REQUIRED_EVIDENCE" for row in outputs)
        or total.get("clean_updates_total") != 120
        or total.get("abft_products_total") != 276
        or abft_total.get("events_total") != 276
        or total.get("status") != "MEASUREMENTS_COMPLETE_PENDING_CUSTODY_AND_REVIEW"
        or abft_total.get("status")
        != "MEASUREMENTS_COMPLETE_PENDING_CUSTODY_AND_REVIEW"
    ):
        raise ValueError("Native baseline completion differs from fixed actual census")
    return {
        "clean_stages": stages,
        "abft_registered_events": 276,
        "abft_journaled_clean_test_and_fault_decisions": 186,
        "abft_calibration_objects_without_native_residual_journal": 90,
        "abft_events": outputs,
        "operand_capture_parity": parity,
        "operand_capture_control_valid": parity_valid,
        "abft_tolerances": tolerances,
        "abft_calibration_rule_numerically_reconstructed": False,
        "limits": [
            "Scoped actual-operand unfused GEMM only; excludes bias/dropout/optimizer/general training",
            "Common-mode operands and balanced cancellation are retained blind-spot cases",
            "90 calibration objects are not90 journaled detector decisions; native residual maxima were not exported",
            "Cheap and checksum timings are unfused references; no optimized-kernel or intrinsic acceleration claim",
            "No independent history replication or complete-case population inference from these products",
        ],
    }
