"""Allocated-only retained E5 state/buffer/RNG comparisons; no model execution."""

import math
import struct

from analysis_core import ARMS, HISTORIES, require_allocation


FIELDS = ("model", "optimizer", "scheduler", "cursor", "updates", "rng")


def compare_cpu_trees(left, right):
    """Compare all leaves; byte-only signed zeros and nonfinite values differ."""
    require_allocation()
    import numpy as np
    import torch

    mismatches, nonfinite, metadata = [], [], []
    byte_equal, value_equal, leaves = True, True, 0

    def walk(a, b, path):
        nonlocal byte_equal, value_equal, leaves
        if type(a) is not type(b):
            metadata.append(path + ":type")
            byte_equal = value_equal = False
            return
        if isinstance(a, dict):
            if {(type(key).__qualname__, repr(key)) for key in a} != {
                (type(key).__qualname__, repr(key)) for key in b
            }:
                metadata.append(path + ":keys")
                byte_equal = value_equal = False
                return
            for key in a:
                walk(a[key], b[key], path + "." + type(key).__name__ + ":" + str(key))
            return
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                metadata.append(path + ":length")
                byte_equal = value_equal = False
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{index}]")
            return
        leaves += 1
        if isinstance(a, torch.Tensor):
            if a.device.type != "cpu" or b.device.type != "cpu":
                raise ValueError("Retained parity must compare trusted CPU loads")
            if (
                a.dtype != b.dtype
                or a.shape != b.shape
                or a.layout != torch.strided
                or b.layout != torch.strided
            ):
                metadata.append(path + ":tensor_metadata")
                byte_equal = value_equal = False
                return
            finite = not (a.is_floating_point() or a.is_complex()) or bool(
                torch.isfinite(a).all() and torch.isfinite(b).all()
            )
            values = finite and torch.equal(a, b)
            raw = torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )
        elif isinstance(a, np.ndarray):
            if a.dtype != b.dtype or a.shape != b.shape:
                metadata.append(path + ":array_metadata")
                byte_equal = value_equal = False
                return
            if a.dtype.hasobject:
                raise ValueError("Object arrays are unsupported retained state")
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            values, raw = finite and np.array_equal(a, b), a.tobytes() == b.tobytes()
        else:
            finite = not isinstance(a, (float, complex)) or (
                math.isfinite(a.real)
                and math.isfinite(a.imag)
                and math.isfinite(b.real)
                and math.isfinite(b.imag)
            )
            values = finite and a == b
            # Preserve signed-zero scalar differences as well as tensor bytes.
            raw = values and (
                a.hex() == b.hex()
                if isinstance(a, float)
                else struct.pack("!dd", a.real, a.imag)
                == struct.pack("!dd", b.real, b.imag)
                if isinstance(a, complex)
                else True
            )
        if not finite:
            nonfinite.append(path)
        if not raw or not values:
            mismatches.append(
                {"path": path, "byte_equal": bool(raw), "value_equal": bool(values)}
            )
        byte_equal = byte_equal and bool(raw)
        value_equal = value_equal and bool(values)

    walk(left, right, "root")
    return {
        "valid": not metadata,
        "all_finite": not nonfinite,
        "byte_equal": byte_equal,
        "value_equal": value_equal and not nonfinite,
        "leaves_compared": leaves,
        "metadata_mismatches": metadata,
        "nonfinite_paths": nonfinite,
        "differences": mismatches,
    }


def retained_state(saved, buffers):
    require_allocation()
    if saved.get("schema") != 1 or any(key not in saved for key in FIELDS):
        raise ValueError("Retained final training state is incomplete/unsupported")
    if not isinstance(buffers, dict) or not buffers:
        raise ValueError(
            "Named nonpersistent/persistent buffers require actual nonempty evidence"
        )
    return {**{key: saved[key] for key in FIELDS}, "named_buffers": buffers}


def _recorded_pass(record):
    return (
        isinstance(record, dict)
        and bool(record.get("tensors"))
        and all(
            record.get(key) is True
            for key in ("valid", "all_finite", "byte_equal", "value_equal")
        )
    )


def evaluate_controls(history, outcomes, parity_record, final_states, initial_counters):
    """Keep actual control evidence and final endpoint applicability separate.

    final_states maps an arm to its already hash-verified saved state and named
    buffers. Missing finals and differing endpoints never become a parity PASS.
    Final comparisons on failed arms remain descriptive, never clean controls.
    """
    require_allocation()
    if (
        history not in HISTORIES
        or len(outcomes) != 4
        or {row.get("arm") for row in outcomes} != set(ARMS)
        or any(row.get("history") != history for row in outcomes)
        or set(final_states) - set(ARMS)
        or initial_counters.get("updates") != 210
        or type(initial_counters.get("cursor")) is not int
    ):
        raise ValueError(
            "Control evidence differs from registered history/arm/input counters"
        )
    summaries = {row["arm"]: row for row in outcomes}
    first_clean = parity_record.get("clean_checking_noninterference")
    clean_first = _recorded_pass(first_clean)
    guarded = summaries.get("buggy-guarded-repaired", {})
    recovery = parity_record.get("first_recovery_equals_clean_full_state_rng")
    first_applicable = (
        guarded.get("first_detection_update") == 1
        and type(guarded.get("accepted_updates")) is int
        and guarded["accepted_updates"] >= 1
        and type(summaries.get("clean-disabled", {}).get("accepted_updates")) is int
        and summaries["clean-disabled"]["accepted_updates"] >= 1
    )
    first_recovery = {
        "applicable": first_applicable,
        "status": "PASS"
        if first_applicable and _recorded_pass(recovery)
        else "MISSING_OR_FAILED_REQUIRED_CONTROL"
        if first_applicable
        else "NOT_APPLICABLE_DELAYED_OR_MISSING_FIRST_ACCEPTED_STATE",
        "raw": recovery,
    }
    comparisons = {}
    for name, a, b in (
        ("clean_checked_vs_clean_disabled", "clean-checked", "clean-disabled"),
        ("guarded_vs_clean_checked", "buggy-guarded-repaired", "clean-checked"),
    ):
        x, y = summaries.get(a), summaries.get(b)
        if not x or not y or a not in final_states or b not in final_states:
            comparisons[name] = {
                "status": "MISSING_FINAL_EVIDENCE",
                "applicable": False,
            }
            continue
        if x.get("accepted_updates") != y.get("accepted_updates"):
            comparisons[name] = {
                "status": "NOT_APPLICABLE_DIFFERENT_ACCEPTED_ENDPOINTS",
                "applicable": False,
                "accepted_updates": [
                    x.get("accepted_updates"),
                    y.get("accepted_updates"),
                ],
            }
            continue
        state_a, buffers_a = final_states[a]
        state_b, buffers_b = final_states[b]
        for summary, saved in ((x, state_a), (y, state_b)):
            if (
                saved["updates"] != 210 + summary["accepted_updates"]
                or saved["cursor"]
                != initial_counters["cursor"] + summary["accepted_updates"] * 65536
            ):
                raise ValueError("Final saved counters differ from matched arm summary")
        result = compare_cpu_trees(
            retained_state(state_a, buffers_a), retained_state(state_b, buffers_b)
        )
        failed = any(
            row.get("status") in ("FAILED_ARM_RETAINED", "MISSING_NOT_RECORDED")
            for row in (x, y)
        )
        passed = all(
            result.get(key) is True
            for key in ("valid", "all_finite", "byte_equal", "value_equal")
        )
        validated = all(row.get("validation_status") == "PASS" for row in (x, y))
        comparisons[name] = {
            "status": "DESCRIPTIVE_FAILED_ARM_COMPARISON"
            if failed
            else "DESCRIPTIVE_UNVALIDATED_ARM_COMPARISON"
            if not validated
            else "PASS"
            if passed
            else "STATE_DIFFERENCE_RETAINED",
            "applicable": True,
            "accepted_updates": x["accepted_updates"],
            "comparison": result,
            "failed_arm_present": failed,
            "arm_validation_passed": validated,
        }
    clean_final = comparisons["clean_checked_vs_clean_disabled"].get("status") == "PASS"
    return {
        "history": history,
        "clean_first_update": {
            "status": "PASS" if clean_first else "MISSING_OR_FAILED_REQUIRED_CONTROL",
            "raw": first_clean,
        },
        "immediate_recovery_first_update": first_recovery,
        "matched_final_state_buffers_rng": comparisons,
        "clean_checking_noninterference_supported": clean_first and clean_final,
        "limit": "Control evidence is required for checking-cost/noninterference claims; delayed/missed alerts and state differences remain actual outcomes",
    }
