"""Additive E1 issue dispositions; never changes strict historical validation."""

from analysis_core import require_allocation


CLIPPING_PREFIX = "RuntimeError: The total norm of order 2.0 for gradients from `parameters` is non-finite, so it cannot be clipped."


def dispositions(e1):
    """Explain each original flagged call using its paired reference and raw state.

    A label here is a description of why a strict gate failed. It cannot turn an
    ERROR into an accepted update or relax a nonfinite gradient equality gate.
    """
    require_allocation()
    references = {
        (item["registration"]["state"], item["registration"].get("case_id")): item[
            "observation"
        ]
        for item in e1["all_calls"]
        if item["registration"]["mode"] == "plain-reconstruction"
    }
    rows = []
    for item in e1["all_calls"]:
        if not item["validation_issues"]:
            continue
        planned = item["registration"]
        observed = item.get("observation") or {}
        reference = references.get((planned["state"], planned.get("case_id"))) or {}
        comparison = observed.get("observer_noninterference") or {}
        label = "UNRESOLVED_REFERENCE_EVENT_OR_PAIRING"
        identity_valid = (
            observed.get("original_reference_valid") is True
            and observed.get("event_bytes_match") is True
            and observed.get("observation_classification", {}).get("status")
            == "SUPPORTED"
            and observed.get("phase") == "BACKWARD_COMPLETE"
            and comparison.get("valid") is True
            and comparison.get("byte_equal") is True
        )
        reference_error = (
            reference.get("status") == "ERROR"
            and reference.get("phase") == "NONFINITE_CLIPPING_ERROR"
            and reference.get("original_event_bytes_match") is True
            and reference.get("original_status_match") is True
        )
        native_error = observed.get("status") == "ERROR" and str(
            observed.get("error", "")
        ).startswith(CLIPPING_PREFIX)
        explicit_nonfinite = planned.get("case_id") == "recompute-middle-nonfinite"
        expected_error_issue = item["validation_issues"] == [
            "fault reference/events/noninterference/pairing not validated"
        ]
        expected_rejection_issue = item["validation_issues"] == [
            "missing or invalid direct observer-gradient comparison"
        ]
        if identity_valid:
            label = "UNRESOLVED_OTHER_STRICT_VALIDATION"
            if native_error and reference_error and expected_error_issue:
                if (
                    comparison.get("all_finite") is True
                    and comparison.get("value_equal") is True
                ):
                    label = "FINITE_GRADIENT_CLIPPING_NORM_ERROR"
                elif (
                    explicit_nonfinite
                    and comparison.get("all_finite") is False
                    and comparison.get("value_equal") is False
                ):
                    label = "EXPLICIT_NONFINITE_REFERENCE_ERROR"
            elif (
                explicit_nonfinite
                and reference_error
                and expected_rejection_issue
                and observed.get("status") == "REJECTED_PRE_OPTIMIZER"
                and item.get("native_pre_optimizer_preservation") is True
                and comparison.get("all_finite") is False
                and comparison.get("value_equal") is False
            ):
                label = "EXPLICIT_NONFINITE_REJECTION_STRICT_COMPARISON_UNMET"
        rows.append(
            {
                "id": planned["id"],
                "state": planned["state"],
                "case_id": planned.get("case_id"),
                "mode": planned["mode"],
                "native_status": observed.get("status"),
                "disposition": label,
                "validation_issues": list(item["validation_issues"]),
                "strict_validation_preserved": True,
                "scientific_acceptance": False,
                "observer_gradient_valid": comparison.get("valid"),
                "observer_gradient_byte_equal": comparison.get("byte_equal"),
                "observer_gradient_all_finite": comparison.get("all_finite"),
                "observer_gradient_value_equal": comparison.get("value_equal"),
                "reference_phase": reference.get("phase"),
                "reference_original_event_bytes_match": reference.get(
                    "original_event_bytes_match"
                ),
                "reference_original_status_match": reference.get(
                    "original_status_match"
                ),
            }
        )
    return {
        "rows": rows,
        "strict_validation_preserved": True,
        "scientific_acceptance": False,
        "interpretation": "Descriptions of retained strict failures, not denominator exclusions or new scientific acceptance",
    }
