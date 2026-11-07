"""Allocated-only fixed original-study census and status/phase validation.

This module reads no files at import and computes no new model endpoints.
The loader must first verify every raw member against frozen custody pins.
"""

from collections import Counter
import math

from analysis_core import require_allocation
from frozen_schedules import CheckingSchedule


def equal(record):
    return all(
        record.get(key) is True
        for key in ("valid", "all_finite", "byte_equal", "value_equal")
    )


def analyze_e1(registration, observations):
    require_allocation()
    expected = {row["id"]: row for row in registration}
    if len(registration) != 638 or len(expected) != 638:
        raise ValueError("E1 requires the frozen distinct638-call registration")
    if sum(row.get("registered_main_attempt") is True for row in registration) != 478:
        raise ValueError("E1 registered/reference role census differs")
    records = {}
    for row in observations:
        identity = row["id"]
        if identity not in expected or identity in records:
            raise ValueError("Foreign or duplicate E1 call")
        records[identity] = row
    output, control_passes = [], []
    for identity, planned in expected.items():
        row = records.get(identity)
        if row is None:
            output.append(
                {
                    "registration": planned,
                    "status": "MISSING_NOT_REACHED",
                    "observation": None,
                    "validation_issues": [],
                }
            )
            if planned["mode"] in (
                "plain-clean",
                "enforce-boundary-off",
                "enforce-all-off",
            ):
                control_passes.append(False)
            continue
        issues = []
        if row.get("stage") != planned["state"]:
            issues.append("state/stage differs from registration")
        mode = planned["mode"]
        scope = (
            "all_operators"
            if mode in ("observe-all", "enforce-all", "enforce-all-off")
            else "checkpoint_boundaries"
        )
        policy = (
            "disabled" if mode in ("plain-clean", "plain-reconstruction") else "every"
        )
        if (
            row.get("scope") != scope
            or row.get("policy") != policy
            or row.get("preserve_rng") is not True
            or row.get("seed") != 1729
        ):
            issues.append("recorded treatment differs from registered call role")
        spec = row.get("spec")
        if "case_id" in planned and (
            not isinstance(spec, dict) or spec.get("case_id") != planned["case_id"]
        ):
            issues.append("fault identity differs from registered call")
        if mode.endswith("-off") and (
            not isinstance(spec, dict) or spec.get("enabled") is not False
        ):
            issues.append("clean trigger-off control has active/missing spec")
        status = row.get("status")
        if status not in ("ACCEPTED", "REJECTED_PRE_OPTIMIZER", "ERROR"):
            issues.append("unknown attempt status")
        if status == "ACCEPTED" and row.get("phase") != "ACCEPTED":
            issues.append("accepted call lacks committed phase")
        rejection = status == "REJECTED_PRE_OPTIMIZER"
        native_preserved = (
            rejection
            and row.get("phase") == "BACKWARD_COMPLETE"
            and equal(row.get("protected_state", {}))
        )
        if rejection and not native_preserved:
            issues.append(
                "rejection lacks valid pre-optimizer protected-state evidence"
            )
        if mode == "observe-all" and rejection:
            issues.append("unenforced observer call unexpectedly rejected")
        if mode in ("observe-all", "enforce-boundary", "enforce-all"):
            if (
                row.get("original_reference_valid") is not True
                or row.get("event_bytes_match") is not True
                or row.get("noninterference_valid") is not True
                or row.get("observation_classification", {}).get("status")
                != "SUPPORTED"
            ):
                issues.append(
                    "fault reference/events/noninterference/pairing not validated"
                )
            if status in ("ACCEPTED", "REJECTED_PRE_OPTIMIZER") and not equal(
                row.get("observer_noninterference", {})
            ):
                issues.append("missing or invalid direct observer-gradient comparison")
            if status == "ACCEPTED" and not equal(
                row.get("accepted_transition_noninterference", {})
            ):
                issues.append(
                    "missing or invalid direct accepted-transition comparison"
                )
        if mode == "plain-clean":
            control_passes.append(status == "ACCEPTED" and not issues)
        elif mode in ("enforce-boundary-off", "enforce-all-off"):
            passed = (
                status == "ACCEPTED"
                and not issues
                and equal(row.get("model_noninterference", {}))
                and equal(row.get("gradient_noninterference", {}))
            )
            control_passes.append(passed)
            if not passed:
                issues.append("failed or incomplete clean noninterference control")
        output.append(
            {
                "registration": planned,
                "status": status,
                "observation": row,
                "native_pre_optimizer_preservation": native_preserved,
                "validated_pre_optimizer_rejection": rejection and not issues,
                "validation_issues": issues,
            }
        )
    return {
        "registered_calls": 638,
        "registered_main_calls": 478,
        "necessary_reference_calls": 160,
        "observed_calls": len(records),
        "missing_calls": 638 - len(records),
        "all_calls": output,
        "census_complete": len(records) == 638,
        "clean_controls_valid": len(control_passes) == 6 and all(control_passes),
        "scientific_acceptance": "PENDING_INDEPENDENT_ACTUAL_REVIEW; incomplete controls/missing calls retained",
        "claim_limit": "Optimizer-step placement confound is source-supported; exact tensor pair and model corruption are not established",
    }


def analyze_e2_not_reached(registered_arms, retained_e2_members):
    require_allocation()
    if len(registered_arms) != 15 or len({row["id"] for row in registered_arms}) != 15:
        raise ValueError("E2 frozen15-arm census differs")
    if retained_e2_members:
        raise ValueError(
            "Unexpected E2 evidence: reconcile original not-reached census before use"
        )
    return {
        "status": "NOT_REACHED",
        "all_arms": [
            dict(registration=row, status="NOT_REACHED") for row in registered_arms
        ],
        "primary_population_intervals": None,
        "recovery_family_intervals": None,
        "reason": "Required history/site endpoints absent; no complete-case CI",
        "frozen_primary_confidence": 0.975,
        "frozen_recovery_confidence": 0.99375,
        "independent_unit": "History mean of two nested sites; three histories",
    }


def active_at(onset, attempt):
    if onset == "immediate":
        return True
    if onset == "delayed":
        return attempt >= 11
    if onset == "intermittent":
        return attempt >= 11 and (attempt - 11) % 4 == 0
    raise ValueError("Unknown frozen onset")


def selected_at(policy, seed, attempt):
    if policy in ("period10", "period100"):
        return CheckingSchedule(
            "periodic", period=int(policy.removeprefix("period")), seed=seed
        ).selected(attempt)
    return CheckingSchedule(policy, seed=seed).selected(attempt)


def _finite_transition(row, accepted, initial, payload_members):
    transition = row.get("transition_difference", {})
    oracle = row.get("corrected_reference", {})
    tape = row.get("forward_tape_receipt", {})
    used = row.get("corrected_tape_use", {})
    tape_path = tape.get("path")
    pinned = (
        payload_members.get("E3/" + tape_path, {}) if isinstance(tape_path, str) else {}
    )
    binding_keys = (
        "path",
        "sha256",
        "microbatches",
        "prestate_cursor",
        "prestate_updates",
    )
    expected_updates = initial["updates"] + accepted
    expected_cursor = initial["cursor"] + accepted * 8192
    return (
        isinstance(oracle, dict)
        and oracle.get("status") == "ACCEPTED"
        and oracle.get("phase") == "ACCEPTED"
        and oracle.get("branch_update") == row.get("branch_update")
        and oracle.get("policy") == "disabled"
        and oracle.get("preserve_rng") is True
        and oracle.get("update", {}).get("update") == expected_updates
        and oracle.get("update", {}).get("accepted_tokens") == expected_cursor
        and oracle.get("update", {}).get("tokens_this_update") == 8192
        and row.get("forward_loss_parity") is True
        and row.get("recorded_forward_microbatches") == 4
        and used.get("same_prestate") is True
        and used.get("preserve_rng") is True
        and all(key in tape and used.get(key) == tape[key] for key in binding_keys)
        and tape.get("microbatches") == 4
        and pinned.get("sha256") == tape.get("sha256")
        and tape.get("prestate_updates") == expected_updates - 1
        and tape.get("prestate_cursor") == expected_cursor - 8192
        and transition.get("valid") is True
        and transition.get("all_finite") is True
        and type(transition.get("value_equal")) is bool
    )


def analyze_e3(schedules, cell_records, initial_counters, payload_members):
    """Validate every retained row; ERROR never acquires a rejection label.

    payload_members is the verified completion-root member mapping. Tape paths
    in row receipts are relative to E3/, and must agree with those actual pins.
    """
    require_allocation()
    if len(schedules) != 39:
        raise ValueError("Frozen E3 registration requires39cells")
    expected = {
        f"E3-{row['policy']}-S{row['seed']}-{row['onset']}": row for row in schedules
    }
    if len(expected) != 39 or any(
        row["max_primary_attempts"] != 300 for row in schedules
    ):
        raise ValueError("Frozen E3 cell/window census differs")
    if (
        initial_counters.get("updates") != 10
        or type(initial_counters.get("cursor")) is not int
    ):
        raise ValueError("Pinned original warm10 counters required")
    observed = {}
    for record in cell_records:
        summary = record.get("summary")
        identity = summary["id"] if summary is not None else record["cell_id"]
        if record.get("cell_id", identity) != identity:
            raise ValueError("E3 registered location differs from summary identity")
        if identity not in expected or identity in observed:
            raise ValueError("Foreign or duplicate E3 cell")
        observed[identity] = record
    results = []
    for identity, cell in expected.items():
        if identity not in observed:
            results.append(
                {
                    "id": identity,
                    "cell": cell,
                    "status": "MISSING_NOT_RECORDED",
                    "validation_issues": ["missing registered cell"],
                    "attempts": [],
                }
            )
            continue
        record = observed[identity]
        summary, rows = record["summary"], record["attempts"]
        issues, classified = [], []
        if not 1 <= len(rows) <= 300 or (
            summary is not None
            and (summary["cell"] != cell or len(rows) != summary["attempts"])
        ):
            issues.append("cell/attempt census differs from registration/summary")
        if record.get("journal_evidence", {}).get("partial_final_write") is not None:
            issues.append("uncommitted final attempt bytes retained")
        accepted, wrong, unsupported, opportunities = 0, 0, 0, 0
        first_wrong, first_alert, onset_accepted = None, None, None
        for index, row in enumerate(rows, start=1):
            row_issues = []
            active = active_at(cell["onset"], index)
            selected = selected_at(cell["policy"], cell["seed"], index)
            if active and onset_accepted is None:
                onset_accepted = accepted
            opportunities += active and selected
            if (
                row.get("id") != f"{identity}-A{index:03}"
                or row.get("branch_update") != index
                or row.get("trigger_active") is not active
                or row.get("selected") is not selected
                or row.get("policy") != cell["policy"]
                or row.get("seed") != cell["seed"]
                or row.get("preserve_rng") is not (not active)
            ):
                row_issues.append(
                    "identity/onset/selection/treatment differs from frozen schedule"
                )
            status, phase = row.get("status"), row.get("phase")
            validated_rejection, actual_wrong = False, None
            if status == "ACCEPTED":
                accepted += 1
                counters = row.get("update", {})
                if (
                    phase != "ACCEPTED"
                    or counters.get("update") != initial_counters["updates"] + accepted
                    or counters.get("accepted_tokens")
                    != initial_counters["cursor"] + accepted * 8192
                    or counters.get("tokens_this_update") != 8192
                ):
                    row_issues.append(
                        "accepted transition phase/update/target counters differ"
                    )
                if (
                    active
                    and not row_issues
                    and _finite_transition(
                        row, accepted, initial_counters, payload_members
                    )
                ):
                    actual_wrong = not row["transition_difference"]["value_equal"]
                    wrong += actual_wrong
                    if actual_wrong and first_wrong is None:
                        first_wrong = index
                    if row.get("actual_wrong_accepted_update") is not actual_wrong:
                        row_issues.append(
                            "saved finite accepted wrong-transition flag differs"
                        )
                elif not active:
                    actual_wrong = False
                    if row.get("actual_wrong_accepted_update") is not False:
                        row_issues.append(
                            "trigger-off accepted update has wrong/unknown label"
                        )
            elif status == "REJECTED_PRE_OPTIMIZER":
                first_alert = index if first_alert is None else first_alert
                validated_rejection = (
                    selected
                    and phase == "BACKWARD_COMPLETE"
                    and equal(row.get("protected_state", {}))
                )
                if not validated_rejection:
                    row_issues.append(
                        "rejection lacks pre-optimizer selection/phase/protected-state proof"
                    )
                if row.get("actual_wrong_accepted_update") is not None:
                    row_issues.append(
                        "uncommitted rejected attempt has an accepted-wrong label"
                    )
            elif status == "ERROR":
                # A convenience reference string is never evidence of a reject.
                if row.get("actual_wrong_accepted_update") is not None:
                    row_issues.append("ERROR carries a finite accepted-wrong label")
            else:
                row_issues.append("unknown primary row status")
            if active:
                tape = row.get("forward_tape_receipt", {})
                tape_path = tape.get("path")
                pinned = (
                    payload_members.get("E3/" + tape_path, {})
                    if isinstance(tape_path, str)
                    else {}
                )
                if (
                    not tape_path
                    or not pinned
                    or pinned.get("sha256") != tape.get("sha256")
                    or tape.get("prestate_updates")
                    != initial_counters["updates"] + accepted - (status == "ACCEPTED")
                    or tape.get("prestate_cursor")
                    != initial_counters["cursor"]
                    + (accepted - (status == "ACCEPTED")) * 8192
                ):
                    row_issues.append(
                        "retained forward tape hash/prestate binding missing or different"
                    )
                if row.get("reference_status") in (
                    "UNSUPPORTED_COUNTERFACTUAL",
                    "NONFINITE_OR_UNSUPPORTED_TRANSITION",
                    "UNSUPPORTED_INCOMPLETE_FORWARD_TAPE",
                ):
                    unsupported += 1
                if (
                    status == "ACCEPTED"
                    and actual_wrong is None
                    and not row.get("reference_status", "").startswith(
                        ("UNSUPPORTED_", "NONFINITE_")
                    )
                ):
                    row_issues.append(
                        "active accepted update lacks a supported finite transition or explicit unsupported outcome"
                    )
            if index < len(rows) and (status != "ACCEPTED" or unsupported):
                row_issues.append("cell continued after native terminal outcome")
            if (
                not math.isfinite(row.get("attempt_total_wall_seconds", float("nan")))
                or row["attempt_total_wall_seconds"] < 0
            ):
                row_issues.append("invalid per-attempt elapsed cost")
            native_preservation = (
                status == "REJECTED_PRE_OPTIMIZER"
                and phase == "BACKWARD_COMPLETE"
                and equal(row.get("protected_state", {}))
            )
            validated_rejection = validated_rejection and not row_issues
            classified.append(
                {
                    "id": row.get("id"),
                    "authoritative_status": status,
                    "phase": phase,
                    "validated_pre_optimizer_rejection": validated_rejection,
                    "native_pre_optimizer_preservation": native_preservation,
                    "finite_actual_wrong_accepted_update": actual_wrong,
                    "byte_only_transition_difference": status == "ACCEPTED"
                    and active
                    and not row_issues
                    and _finite_transition(
                        row, accepted, initial_counters, payload_members
                    )
                    and row["transition_difference"].get("value_equal") is True
                    and row["transition_difference"].get("byte_equal") is False,
                    "validation_issues": row_issues,
                    "raw": row,
                }
            )
            issues.extend(f"A{index:03}: {message}" for message in row_issues)
        if summary is None or not rows:
            issues.append(
                "missing native terminal summary"
                if summary is None
                else "missing retained attempt journal"
            )
            results.append(
                {
                    "id": identity,
                    "cell": cell,
                    "status": "MISSING_TERMINAL_SUMMARY_RETAINED"
                    if summary is None
                    else "MISSING_ATTEMPT_JOURNAL_RETAINED",
                    "summary": summary,
                    "attempts": classified,
                    "validation_issues": issues,
                    "journal_evidence": record.get("journal_evidence", {}),
                    "observed_prefix": {
                        "attempts": len(rows),
                        "accepted_updates": accepted,
                        "accepted_targets": accepted * 8192,
                        "finite_wrong_accepted_updates": None if unsupported else wrong,
                        "selected_active_opportunities": opportunities,
                        "first_alert": first_alert,
                        "first_proven_wrong_transition": first_wrong,
                    },
                    "terminal_duration_or_completed_endpoint": None,
                }
            )
            continue
        if (
            accepted != summary["accepted_updates"]
            or accepted * 8192 != summary["accepted_targets"]
            or opportunities != summary["selected_active_opportunities"]
            or unsupported != summary["unsupported_counterfactuals"]
            or summary["actual_wrong_accepted_updates"]
            != (None if unsupported else wrong)
            or first_wrong != summary["first_proven_wrong_transition"]
            or first_alert != summary["first_alert"]
        ):
            issues.append(
                "summary counters/first events differ from full retained journal"
            )
        if rows:
            last = rows[-1]
            if last.get("status") == "REJECTED_PRE_OPTIMIZER":
                expected_status = (
                    "REJECTED_PRE_OPTIMIZER"
                    if last.get("protected_state", {}).get("byte_equal") is True
                    else "FAILED_PROTECTED_STATE"
                )
            elif last.get("status") == "ERROR" or unsupported:
                expected_status = "INCOMPLETE_ERROR_OR_COUNTERFACTUAL"
            elif last.get("status") == "ACCEPTED" and len(rows) == 300:
                expected_status = "CENSORED_AT300"
            else:
                expected_status = None
            if summary.get("status") != expected_status or expected_status is None:
                issues.append(
                    "summary terminal status differs from final authoritative native outcome"
                )
        if summary.get("independent_history_count") != 1:
            issues.append("E3 history replication metadata differs from one history")
        if summary["status"] == "CENSORED_AT300" and (
            len(rows) != 300 or accepted != 300 or unsupported
        ):
            issues.append(
                "censored window did not retain300supported accepted attempts"
            )
        if summary["status"] == "REJECTED_PRE_OPTIMIZER" and (
            not classified or not classified[-1]["validated_pre_optimizer_rejection"]
        ):
            issues.append(
                "summary rejection does not end in a validated pre-optimizer rejection"
            )
        delay_attempts = (
            first_alert - (1 if cell["onset"] == "immediate" else 11)
            if first_alert is not None
            else None
        )
        delay_updates = (
            accepted - onset_accepted
            if first_alert is not None and onset_accepted is not None
            else None
        )
        if (
            summary["delay_attempts_from_trigger_onset"] != delay_attempts
            or summary["delay_accepted_updates_from_trigger_onset"] != delay_updates
            or summary["delay_accepted_targets_from_trigger_onset"]
            != (delay_updates * 8192 if delay_updates is not None else None)
            or summary["delay_from_first_wrong"]
            != (
                first_alert - first_wrong
                if first_alert is not None and first_wrong is not None
                else None
            )
        ):
            issues.append("summary delay endpoints differ from retained paired cell")
        results.append(
            {
                "id": identity,
                "cell": cell,
                "status": summary["status"],
                "summary": summary,
                "journal_evidence": record.get("journal_evidence", {}),
                "validation_issues": issues,
                "attempts": classified,
            }
        )
    statuses = Counter(row["status"] for row in results)
    return {
        "registered_cells": 39,
        "all_cells": results,
        "cell_status_counts": dict(statuses),
        "attempt_rows_validated": sum(len(row["attempts"]) for row in results),
        "validation_issues": sum(len(row["validation_issues"]) for row in results),
        "independent_training_histories": 1,
        "replication_limit": "Eight randomized seeds reused/paired across onsets; not24independent draws",
        "diagnostic_targets_per_update": 8192,
        "production_targets_per_update": 65536,
        "scientific_acceptance": "PENDING_BASELINE_ABFT_AND_INDEPENDENT_ACTUAL_REVIEW",
    }
