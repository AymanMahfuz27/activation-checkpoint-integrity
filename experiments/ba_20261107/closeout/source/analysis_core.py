"""Draft allocated-only E4/E5 reducers. No import performs data processing.

The execution/admission wrapper and E1/E3 export are still required. These
functions intentionally require a job-bound analysis release even for fixtures.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path
from frozen_schedules import CheckingSchedule


HISTORIES = (1234, 1, 2)
ARMS = ("clean-disabled", "buggy-disabled", "buggy-guarded-repaired", "clean-checked")
POLICIES = ("startup", "period10", "period100", "randomized", "every")
EVALUATIONS = (0, *range(20, 281, 20), 290)
TARGETS = {1234: 3.23264888940468, 1: 3.213664556043455, 2: 3.17776666496699}


def require_allocation():
    """Admission must bind this job, source packet, contract, and frozen inputs."""
    job = os.environ.get("SLURM_JOB_ID")
    release = os.environ.get("ACI_ANALYSIS_RELEASE")
    if not job or not release:
        raise RuntimeError("Numerical work requires an allocated analysis release")
    path = Path(release)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Analysis release must be a saved regular receipt")
    record = json.loads(path.read_text())
    if not (
        record.get("status") == "PASS"
        and record.get("scope") in ("ALLOCATED_FIXTURES_ONLY", "ALLOCATED_ANALYSIS")
        and record.get("job_id") == job
        and record.get("account") == "CCR26063"
        and record.get("source_packet_sha256")
        == os.environ.get("ACI_ANALYSIS_SOURCE_SHA256")
        and record.get("contract_sha256")
        == os.environ.get("ACI_ANALYSIS_CONTRACT_SHA256")
        and record.get("inputs_sha256") == os.environ.get("ACI_ANALYSIS_INPUTS_SHA256")
        and all(
            record.get(key)
            for key in ("source_packet_sha256", "contract_sha256", "inputs_sha256")
        )
    ):
        raise RuntimeError("Analysis release does not bind this allocation")
    return record


def _finite(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _interval(values, confidence):
    """Student-t interval on exactly three independent unit values."""
    if len(values) != 3 or any(not math.isfinite(x) for x in values):
        raise ValueError("Three finite independent units are required")
    from scipy.stats import t

    mean = math.fsum(values) / 3
    variance = math.fsum((x - mean) ** 2 for x in values) / 2
    half = float(t.ppf((1 + confidence) / 2, df=2)) * math.sqrt(variance / 3)
    return {
        "n": 3,
        "df": 2,
        "confidence": confidence,
        "mean": mean,
        "lower": mean - half,
        "upper": mean + half,
        "assumption": "Independent unit means/log ratios are approximately normal; n=3",
    }


def _ratio_interval(values):
    interval = _interval([math.log(x) for x in values], 0.95)
    return {
        "ratio": math.exp(interval["mean"]),
        "lower": math.exp(interval["lower"]),
        "upper": math.exp(interval["upper"]),
        "log_interval": interval,
    }


def analyze_e4(pairs, arms, registered_blocks, initial_counters):
    """Equal-work paired ratios; equal-weight three fresh job means in log space."""
    require_allocation()
    if (
        type(initial_counters.get("updates")) is not int
        or type(initial_counters.get("cursor")) is not int
        or initial_counters["updates"] != 10
        or initial_counters["cursor"] < 0
    ):
        raise ValueError("Pinned E4 early-warm10 counters are required")
    expected = {block["id"]: block for block in registered_blocks}
    if len(expected) != 50 or len(registered_blocks) != 50:
        raise ValueError("Frozen E4 census must contain 50 distinct blocks")
    observed = Counter(row["id"] for row in pairs)
    if set(observed) != set(expected) or any(count != 1 for count in observed.values()):
        raise ValueError("Missing, duplicate, or foreign E4 pair")
    arm_index = {}
    for row in arms:
        key = (row["block"]["id"], row["arm"])
        if key in arm_index or key[0] not in expected:
            raise ValueError("Duplicate or foreign E4 arm")
        arm_index[key] = row
    wanted_arms = {
        (key, arm)
        for key, block in expected.items()
        for arm in ("disabled", block["policy"])
    }
    if set(arm_index) != wanted_arms:
        raise ValueError("All 100 registered E4 arms are required")
    block_results, unit_logs = [], {}
    journaled, warmups = 0, 0
    for row in pairs:
        block = expected[row["id"]]
        if (
            row["policy"] != block["policy"]
            or row["job"] != block["job"]
            or row["order"] != block["order"]
        ):
            raise ValueError("E4 block registration differs")
        disabled = arm_index[(row["id"], "disabled")]
        checked = arm_index[(row["id"], block["policy"])]
        for arm in (disabled, checked):
            if arm["block"] != block:
                raise ValueError("Arm embeds a different frozen block")
            count = block["measured_per_arm"]
            if (
                arm["measured_updates"] != count
                or len(arm["individual_updates"]) != count
                or arm["measured_targets"] != 65536 * count
            ):
                raise ValueError("E4 equal-work update/target census differs")
            warmup = block.get("warmup_per_arm")
            if warmup != 20:
                raise ValueError("Frozen E4 warmup count differs")
            updates = arm["individual_updates"]
            kind = arm["arm"]
            checking = (
                CheckingSchedule(
                    "periodic",
                    period=int(kind.removeprefix("period")),
                    seed=block["schedule_seed"],
                )
                if kind in ("period10", "period100")
                else CheckingSchedule(kind, seed=block["schedule_seed"])
            )
            selected = [checking.selected(21 + offset) for offset in range(count)]
            warm_selected = sum(checking.selected(offset) for offset in range(1, 21))
            if (
                any(
                    update.get("selected") is not wanted
                    for update, wanted in zip(updates, selected)
                )
                or type(arm.get("measured_checked")) is not int
                or arm["measured_checked"] != sum(selected)
                or type(arm.get("warmup_checked")) is not int
                or arm["warmup_checked"] != warm_selected
            ):
                raise ValueError(
                    "E4 selected-check treatment/counters differ from frozen branch schedule"
                )
            for update, wanted in zip(updates, selected):
                comparison = update.get("comparison")
                if not wanted:
                    if comparison is not None:
                        raise ValueError(
                            "Unselected E4 update unexpectedly has a checking comparison"
                        )
                    continue
                if (
                    not isinstance(comparison, dict)
                    or comparison.get("failed") is not False
                    or comparison.get("mode") != "fingerprint"
                    or comparison.get("observation_scope") != "checkpoint_boundaries"
                    or comparison.get("policy") != "bit_exact_fail_closed"
                    or type(comparison.get("eligible_pairs")) is not int
                    or comparison["eligible_pairs"] <= 0
                    or comparison.get("exact_pairs") != comparison["eligible_pairs"]
                    or comparison.get("counts", {}).get("exact_match")
                    != comparison["eligible_pairs"]
                ):
                    raise ValueError(
                        "Selected E4 comparison is unsupported, incomplete or failed"
                    )
            first = initial_counters["updates"] + warmup + 1
            expected_ids = list(range(first, first + count))
            if (
                [update["update"] for update in updates] != expected_ids
                or any(update.get("tokens_this_update") != 65536 for update in updates)
                or [update.get("accepted_tokens") for update in updates]
                != [
                    initial_counters["cursor"] + (warmup + offset + 1) * 65536
                    for offset in range(count)
                ]
            ):
                raise ValueError("E4 measured absolute update/token boundary differs")
            if not _finite(arm["full_loop_seconds"], positive=True):
                raise ValueError("E4 primary elapsed time is not finite positive")
            journaled += count
            warmups += warmup  # Validated source/counter support; no individual warmup journals.
        if block["policy"] == "startup" and checked["measured_checked"] != 0:
            raise ValueError("Frozen startup checks occur outside the primary window")
        if (
            row.get("measured_policy_checks") != checked["measured_checked"]
            or row.get("measured_updates_per_arm") != block["measured_per_arm"]
            or row.get("scope") != "checkpoint_boundaries"
            or row.get("disabled_seconds") != disabled["full_loop_seconds"]
            or row.get("policy_seconds") != checked["full_loop_seconds"]
        ):
            raise ValueError(
                "E4 saved pair treatment/work/time counters differ from both arms"
            )
        parity_rows = [
            arm["paired_full_state_and_rng"]
            for arm in (disabled, checked)
            if "paired_full_state_and_rng" in arm
        ]
        if len(parity_rows) != 1 or not all(
            parity_rows[0].get(key) is True
            for key in ("valid", "all_finite", "byte_equal", "value_equal")
        ):
            raise ValueError(
                "E4 paired final-state/RNG comparison is not a finite equality"
            )
        ratio = checked["full_loop_seconds"] / disabled["full_loop_seconds"]
        if not math.isclose(ratio, row["full_loop_ratio"], rel_tol=1e-12, abs_tol=0):
            raise ValueError("Saved E4 ratio differs from its paired arm times")
        unit_logs.setdefault((block["policy"], block["job"]), []).append(
            math.log(ratio)
        )
        block_results.append({**row, "recomputed_full_loop_ratio": ratio})
    if journaled != 4400 or warmups != 2000:
        raise ValueError("E4 6400-update census differs")
    results = {}
    for policy in POLICIES:
        logs = [unit_logs.get((policy, job), []) for job in (1, 2, 3)]
        if not all(logs) or sum(map(len, logs)) != 10:
            raise ValueError(
                "Each E4 policy requires ten pairs and all three fresh jobs"
            )
        job_means = [math.fsum(values) / len(values) for values in logs]
        interval = _interval(job_means, 0.95)
        results[policy] = {
            "job_mean_log_ratios": dict(zip((1, 2, 3), job_means)),
            "pairs_per_job": list(map(len, logs)),
            "geometric_ratio": math.exp(interval["mean"]),
            "lower": math.exp(interval["lower"]),
            "upper": math.exp(interval["upper"]),
            "log_interval": interval,
        }
    return {
        "independent_unit": "fresh exclusive Slurm job",
        "n": 3,
        "primary": results,
        "all_blocks": block_results,
        "directly_journaled_updates": journaled,
        "source_counter_supported_warmups": warmups,
        "registered_updates": journaled + warmups,
        "startup_limit": "All startup checks are in warmup, zero in primary window",
        "isolation_limit": "Exclusive allocation plus sampled process evidence",
    }


def _equal(record):
    return all(
        record.get(key) is True
        for key in ("valid", "all_finite", "byte_equal", "value_equal")
    )


def _e5_arm(summary, evaluations, attempts, initial, terminal_evidence):
    """Reconstruct committed transitions and the later post-fsync time event."""
    issues, stages = [], []
    history, arm = summary["history"], summary["arm"]
    if history not in HISTORIES or arm not in ARMS:
        raise ValueError("Foreign E5 history/arm")
    if (
        initial.get("updates") != 210
        or type(initial.get("cursor")) is not int
        or initial["cursor"] < 0
    ):
        raise ValueError("Pinned E5 global210 starting counters are required")
    status = summary["status"]
    partial_writes = terminal_evidence.get("journal_partial_writes", {})
    if any(value is not None for value in partial_writes.values()):
        if status == "TARGET_CONFIRMED":
            issues.append(
                "confirmed outcome has uncommitted or incomplete final journal bytes"
            )
        else:
            stages.append(
                {
                    "stage": "UNCOMMITTED_FINAL_JOURNAL_WRITE_RETAINED",
                    "raw": partial_writes,
                }
            )
    terminal = status in ("FAILED_ARM_RETAINED", "CENSORED_WALL_LIMIT")
    error = summary.get("error")
    cutoff = (
        status == "CENSORED_WALL_LIMIT"
        and error == "Science cutoff reached; preserve partial evidence without retry"
    )
    failure = terminal_evidence.get("failure_json", {})
    actual_exception = (
        terminal
        and bool(error)
        and bool(failure.get("traceback"))
        and failure.get("error") == error
    )
    native_timeout = status == "CENSORED_WALL_LIMIT" and bool(error)
    accepted = summary["accepted_updates"]
    if type(accepted) is not int or not 0 <= accepted <= 290:
        raise ValueError("Accepted-update distance outside frozen horizon")
    if summary.get("accepted_targets") != accepted * 65536:
        issues.append("accepted target count differs from accepted updates")
    target = TARGETS[history]
    if summary["target"]["cross_entropy_target"] != target:
        issues.append("target differs from frozen history target")
    scheduled = [row["branch_update"] for row in evaluations]
    if scheduled != sorted(set(scheduled)) or any(
        update not in EVALUATIONS for update in scheduled
    ):
        issues.append("evaluation journal is duplicated, out of order, or off schedule")
    prefix = [update for update in EVALUATIONS if update <= accepted]
    if scheduled != prefix:
        terminal_gap = (
            terminal
            and prefix
            and scheduled == prefix[:-1]
            and (actual_exception or native_timeout)
        )
        if terminal_gap:
            stages.append(
                {
                    "stage": "TERMINAL_EVALUATION_NOT_COMMITTED",
                    "update": prefix[-1],
                    "exception_evidence": "failure.json"
                    if actual_exception
                    else "native TimeoutError summary; no failure.json is emitted by that branch",
                    "location_limit": "Operation location unresolved without a traceback; no completed evaluation or target inferred",
                }
            )
        else:
            issues.append("unexplained or interior evaluation prefix gap")
    first_crossing, qualifying_pair = None, None
    previous, last_seconds = False, -1
    for row in evaluations:
        ce, elapsed = row.get("cross_entropy"), row.get("elapsed_seconds")
        if (
            row.get("status") != "PASS"
            or not _finite(ce)
            or row.get("heldout_tokens") != 1048576
            or not _finite(elapsed)
            or elapsed < last_seconds
            or not _equal(row.get("state_preservation", {}))
        ):
            issues.append(
                "invalid/nonmonotone validation or failed evaluator-state preservation"
            )
            previous = False
            continue
        last_seconds = elapsed
        update = row["branch_update"]
        qualifies = update > 0 and ce <= target
        if update == 0 and ce <= target:
            issues.append("target already met at excluded baseline")
        if qualifies and first_crossing is None:
            first_crossing = update
        if previous and qualifies and qualifying_pair is None:
            qualifying_pair = update
        if update > 0 and (
            row.get("qualifies") is not qualifies
            or row.get("confirmation") is not (previous and qualifies)
        ):
            issues.append(
                "saved qualifying/confirmation flag differs from recomputation"
            )
        previous = qualifies
    confirmation_time = None
    saved = summary.get("confirmed_target")
    if status == "TARGET_CONFIRMED":
        if (
            not saved
            or saved.get("update") != qualifying_pair
            or not qualifying_pair
            or not scheduled
            or qualifying_pair != scheduled[-1]
            or accepted != qualifying_pair
        ):
            issues.append(
                "summary confirmation/accepted endpoint differs from first scheduled pair"
            )
        elif (
            not _finite(saved.get("elapsed_seconds"), positive=True)
            or saved["elapsed_seconds"] < evaluations[-1]["elapsed_seconds"]
            or saved["elapsed_seconds"] > summary["observed_seconds"]
        ):
            issues.append("post-fsync target clock is outside observed bounds")
        else:
            confirmation_time = saved["elapsed_seconds"]
    elif status in (
        "CENSORED_UPDATE_LIMIT",
        "CENSORED_WALL_LIMIT",
        "FAILED_ARM_RETAINED",
    ):
        if saved is not None:
            issues.append("nonconfirmed outcome contains a saved confirmation commit")
        if qualifying_pair is not None:
            # There is no cutoff call between evaluation return, append/fsync,
            # and setting confirmed_target. Deadline passage alone is no proof.
            if actual_exception or native_timeout:
                stages.append(
                    {
                        "stage": "QUALIFYING_PAIR_WITHOUT_OBSERVED_CONFIRMATION_COMMIT",
                        "update": qualifying_pair,
                        "evidence": "retained exception"
                        if actual_exception
                        else "native TimeoutError summary only",
                        "location_limit": "UNRESOLVED_TIMEOUT_LOCATION"
                        if not actual_exception
                        else "retained traceback",
                        "no_post_fsync_time_inferred": True,
                    }
                )
            else:
                issues.append(
                    "qualifying pair lacks a commit and actual interruption evidence"
                )
        if status == "CENSORED_UPDATE_LIMIT" and accepted != 290:
            issues.append("update censoring before fixed update limit")
    else:
        issues.append("unrecognized E5 outcome status")
    if not _finite(summary.get("observed_seconds"), positive=True):
        issues.append("observed elapsed bound is not finite positive")
    committed, repaired, retries, rejects = 0, False, [], []
    recovery_selected = False
    recovery_stage = "NO_ALERT_REPAIR_NOT_OBSERVED"
    for index, row in enumerate(attempts):
        update, ordinal = row.get("branch_update"), row.get("attempt_ordinal")
        if row.get("history") != history or row.get("arm") != arm:
            issues.append("attempt history/arm identity differs")
        retry = row.get("is_single_safe_retry") is True
        if update != committed + 1 or type(update) is not int or not 1 <= update <= 290:
            issues.append(
                "branch advanced without exactly one preceding accepted transition"
            )
        if retry:
            retries.append(row)
            previous_row = attempts[index - 1] if index else {}
            if (
                arm != "buggy-guarded-repaired"
                or ordinal != 1
                or len(retries) != 1
                or previous_row.get("status") != "REJECTED_PRE_OPTIMIZER"
                or previous_row.get("branch_update") != update
                or previous_row.get("attempt_ordinal") != 0
                or previous_row.get("permanent_correction_selected") is not True
                or not _equal(previous_row.get("full_rollback_parity", {}))
            ):
                issues.append(
                    "safe retry violates reject/rollback/same-update sequence"
                )
        elif ordinal != 0:
            issues.append("only single safe retry may use nonzero attempt ordinal")
        expected_policy = (
            "every"
            if arm in ("clean-checked", "buggy-guarded-repaired")
            else "disabled"
        )
        expected_rng = arm != "buggy-disabled" and (
            arm != "buggy-guarded-repaired" or repaired or retry
        )
        if (
            row.get("policy") != expected_policy
            or row.get("preserve_rng") is not expected_rng
        ):
            issues.append("attempt checking/RNG treatment differs from frozen arm")
        outcome = row.get("status")
        if outcome == "ACCEPTED":
            committed += 1
            counters = row.get("update", {})
            if (
                row.get("phase") != "ACCEPTED"
                or counters.get("update") != initial["updates"] + committed
                or counters.get("accepted_tokens")
                != initial["cursor"] + committed * 65536
                or counters.get("tokens_this_update") != 65536
            ):
                issues.append(
                    "accepted phase or committed update/token counters differ"
                )
            if retry:
                repaired = True
                recovery_stage = "SAFE_RETRY_ACCEPTED_PERMANENT_CORRECTION_SELECTED"
        elif outcome == "REJECTED_PRE_OPTIMIZER":
            rejects.append(row)
            if row.get("phase") != "BACKWARD_COMPLETE":
                issues.append("rejection was not at the pre-optimizer phase")
            if arm == "buggy-guarded-repaired" and retry:
                # The one safe retry may itself reject. The worker exits this
                # arm as failed and never performs a second rollback/retry.
                recovery_stage = "SAFE_RETRY_REJECTED_RETAINED"
                if status != "FAILED_ARM_RETAINED" or index != len(attempts) - 1:
                    issues.append("rejected safe retry was not a terminal failed arm")
                stages.append({"stage": recovery_stage, "update": update})
            elif arm == "buggy-guarded-repaired" and repaired:
                # A later corrected/checking attempt can still reject. Native
                # code retains a terminal failure and does not start recovery
                # again; initial rollback fields do not belong on this row.
                recovery_stage = "POST_REPAIR_REJECTION_RETAINED"
                if status != "FAILED_ARM_RETAINED" or index != len(attempts) - 1:
                    issues.append("post-repair rejection was not terminal failed arm")
                stages.append({"stage": recovery_stage, "update": update})
            elif arm == "buggy-guarded-repaired":
                if len(rejects) != 1 or repaired:
                    issues.append("multiple/post-repair rejections")
                protected = _equal(row.get("protected_state_preservation", {}))
                rollback = _equal(row.get("full_rollback_parity", {}))
                recovery_selected = row.get("permanent_correction_selected") is True
                # Native failed gates are experimental failures, not silently
                # promoted successes or automatically malformed data.
                if not protected or not rollback:
                    recovery_stage = (
                        "FAILED_OR_INTERRUPTED_PROTECTED_STATE"
                        if not protected
                        else "FAILED_OR_INTERRUPTED_FULL_ROLLBACK"
                    )
                    if not (status == "FAILED_ARM_RETAINED" and actual_exception):
                        issues.append(
                            "recovery preservation gate absent/failed without failure evidence"
                        )
                    stages.append({"stage": recovery_stage, "update": update})
                elif index + 1 == len(attempts):
                    recovery_stage = "ROLLBACK_COMPLETE_RETRY_NOT_REACHED"
                    if not (
                        terminal and recovery_selected and (cutoff or actual_exception)
                    ):
                        issues.append("unexplained missing safe retry")
                    stages.append({"stage": recovery_stage, "update": update})
        elif outcome == "ERROR":
            if status != "FAILED_ARM_RETAINED" or index != len(attempts) - 1:
                issues.append("runtime ERROR not retained as terminal failed arm")
            if retry:
                recovery_stage = "SAFE_RETRY_FAILED_RETAINED"
        else:
            issues.append("unknown attempted-update outcome")
        if outcome != "ACCEPTED" and index + 1 < len(attempts):
            following = attempts[index + 1]
            if not (
                outcome == "REJECTED_PRE_OPTIMIZER"
                and following.get("is_single_safe_retry") is True
            ):
                issues.append(
                    "continued after a failed/rejected attempt outside safe retry"
                )
    if committed != accepted:
        issues.append("attempt/summary accepted census differs")
    if len(rejects) != summary.get("rejection_count"):
        issues.append("rejection count differs")
    first_alert = rejects[0]["branch_update"] if rejects else None
    if first_alert != summary.get("first_detection_update"):
        issues.append("first alert differs from retained attempts")
    if (
        arm == "buggy-guarded-repaired"
        and summary.get("permanent_correction_applied") is not recovery_selected
    ):
        issues.append(
            "legacy correction-selected summary flag differs from native attempt flag"
        )
    if status == "TARGET_CONFIRMED" and (
        not attempts
        or attempts[-1].get("status") != "ACCEPTED"
        or attempts[-1].get("branch_update") != accepted
    ):
        issues.append("confirmed arm has no matching final accepted transition")
    cost_keys = (
        "observed_seconds",
        "evaluation_seconds",
        "attempt_loop_seconds",
        "rejected_attempt_compute_seconds",
        "rollback_seconds",
        "safe_retry_seconds",
        "experimental_restore_seconds",
        "final_retention_seconds",
    )
    cost = {key: summary.get(key) for key in cost_keys}
    unknown = []
    tolerance = 1e-6
    observed = summary.get("observed_seconds")
    for label, records in (("attempt", attempts), ("evaluation", evaluations)):
        previous_elapsed = -1.0
        for record in records:
            elapsed = record.get("elapsed_seconds")
            if (
                not _finite(elapsed)
                or elapsed + tolerance < previous_elapsed
                or (_finite(observed) and elapsed > observed + tolerance)
            ):
                issues.append(
                    label + " clock is invalid, nonmonotone, or after observed bound"
                )
            if _finite(elapsed):
                previous_elapsed = elapsed
    # Worker order is E0, attempts through E20, E20, attempts through E40,
    # E40, etc. Separate monotonic journals cannot establish that ordering.
    for evaluation in evaluations:
        ev_time, ev_update = (
            evaluation.get("elapsed_seconds"),
            evaluation.get("branch_update"),
        )
        if not _finite(ev_time) or type(ev_update) is not int:
            continue
        for attempt in attempts:
            attempt_time, attempt_update = (
                attempt.get("elapsed_seconds"),
                attempt.get("branch_update"),
            )
            if not _finite(attempt_time) or type(attempt_update) is not int:
                continue
            if (attempt_update <= ev_update and attempt_time > ev_time + tolerance) or (
                attempt_update > ev_update and attempt_time + tolerance < ev_time
            ):
                issues.append(
                    "cross-journal attempt/evaluation chronology differs from worker order"
                )
    attempt_durations = []
    for row in attempts:
        duration = row.get("wall_seconds")
        if duration is None and terminal:
            unknown.append("interrupted_attempt_wall_seconds")
        elif not _finite(duration):
            issues.append("invalid accepted/failed attempt wall duration")
        else:
            attempt_durations.append(duration)
    rollback_durations = [row.get("rollback_seconds", 0.0) for row in attempts]
    if (
        _finite(cost.get("attempt_loop_seconds"))
        and all(_finite(value) for value in rollback_durations)
        and math.fsum(attempt_durations + rollback_durations)
        > cost["attempt_loop_seconds"] + tolerance
    ):
        issues.append(
            "retained attempt/rollback durations exceed aggregate attempt loop"
        )
    for key, value in cost.items():
        if value is None and terminal:
            unknown.append(key)
        elif not _finite(value):
            issues.append("invalid cost annotation: " + key)
    for key, selected, field in (
        ("rejected_attempt_compute_seconds", rejects, "wall_seconds"),
        ("safe_retry_seconds", retries, "wall_seconds"),
        ("rollback_seconds", attempts, "rollback_seconds"),
    ):
        values = [row.get(field, 0.0) for row in selected]
        if any(not _finite(value) for value in values):
            issues.append("invalid retained attempt cost: " + field)
        elif _finite(cost[key]) and not math.isclose(
            cost[key], math.fsum(values), rel_tol=1e-9, abs_tol=1e-8
        ):
            issues.append("summary/attempt nested cost differs: " + key)
    top = ("observed_seconds", "evaluation_seconds", "attempt_loop_seconds")
    if all(_finite(cost.get(key)) for key in top):
        residual = (
            cost["observed_seconds"]
            - cost["evaluation_seconds"]
            - cost["attempt_loop_seconds"]
        )
        if residual < -1e-6:
            issues.append("exclusive elapsed components exceed observed clock")
        cost["unallocated_in_clock_seconds"] = residual
    cost["unknown_components"] = unknown
    cost["residual_scope"] = (
        "Unallocated elapsed time; may include interrupted work, not attributed overhead"
    )
    if issues:
        confirmation_time = None
    return {
        "history": history,
        "arm": arm,
        "status": status,
        "validation_status": "FAILED_VALIDATION_RETAINED"
        if issues
        else "PARTIAL_TERMINAL_STAGE_RETAINED"
        if stages
        else "PASS",
        "issues": sorted(set(issues)),
        "terminal_stages": stages,
        "accepted_updates": accepted,
        "first_detection_update": first_alert,
        "observed_bound_seconds": summary.get("observed_seconds"),
        "numerically_qualifying_pair_update": qualifying_pair,
        "confirmed_update": saved.get("update")
        if confirmation_time is not None
        else None,
        "confirmed_seconds": confirmation_time,
        "first_crossing_update_secondary": first_crossing,
        "recovery_stage": recovery_stage,
        "legacy_correction_selected": summary.get("permanent_correction_applied"),
        "safe_retry_observed": bool(retries),
        "safe_retry_accepted": repaired,
        "cost": cost,
        "evaluations": evaluations,
        "attempts": attempts,
    }


def _missing_e5(history, arm, record=None):
    record = record or {}
    evaluations, attempts = record.get("evaluations", []), record.get("attempts", [])
    clocks = [
        row.get("elapsed_seconds")
        for row in evaluations + attempts
        if _finite(row.get("elapsed_seconds"))
    ]
    return {
        "history": history,
        "arm": arm,
        "status": "MISSING_ARM_SUMMARY_RETAINED",
        "validation_status": "MISSING_TERMINAL_OBSERVATION",
        "issues": ["missing native terminal arm summary"],
        "accepted_updates": None,
        "first_detection_update": None,
        "observed_bound_seconds": None,
        "last_journal_elapsed_seconds": max(clocks) if clocks else None,
        "bound_limit": "Last journal time is only a lower bound; terminal measured duration is unknown",
        "confirmed_seconds": None,
        "confirmed_update": None,
        "cost": {},
        "evaluations": evaluations,
        "attempts": attempts,
        "journal_evidence": record.get("journal_evidence", {}),
    }


def analyze_e5(records, initial_counters):
    require_allocation()
    expected = {(history, arm) for history in HISTORIES for arm in ARMS}
    outcomes = {}
    for record in records:
        summary = record.get("summary")
        key = (
            (summary["history"], summary["arm"])
            if summary is not None
            else (record["history"], record["arm"])
        )
        if key not in expected or key in outcomes:
            raise ValueError("Duplicate or foreign E5 arm")
        outcomes[key] = (
            _missing_e5(*key, record)
            if summary is None
            else _e5_arm(
                summary,
                record["evaluations"],
                record["attempts"],
                initial_counters[summary["history"]],
                record.get("terminal_evidence", {}),
            )
        )
    for history, arm in expected - set(outcomes):
        outcomes[(history, arm)] = _missing_e5(history, arm)
    contrasts = {}
    for name, numerator, denominator in (
        ("recovery_policy_vs_buggy", "buggy-guarded-repaired", "buggy-disabled"),
        ("checking_cost", "clean-checked", "clean-disabled"),
        ("recovery_residual", "buggy-guarded-repaired", "clean-checked"),
    ):
        pairs = []
        for history in HISTORIES:
            left, right = (
                outcomes[(history, numerator)],
                outcomes[(history, denominator)],
            )
            a, b = left["confirmed_seconds"], right["confirmed_seconds"]
            pairs.append(
                {
                    "history": history,
                    "numerator_status": left["status"],
                    "denominator_status": right["status"],
                    "numerator_seconds": a,
                    "denominator_seconds": b,
                    "ratio": a / b if a is not None and b is not None else None,
                }
            )
        values = [pair["ratio"] for pair in pairs]
        complete = all(value is not None for value in values)
        contrasts[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "histories": pairs,
            "population_interval": _ratio_interval(values) if complete else None,
            "interval_status": "COMPLETE_THREE_HISTORY_PAIRS"
            if complete
            else "NOT_ESTIMABLE_NO_COMPLETE_CASE_POPULATION_INTERVAL",
        }
    return {
        "independent_unit": "history/job",
        "n": 3,
        "outcomes": list(outcomes.values()),
        "contrasts": contrasts,
        "known_validation_target": True,
        "claim_scope": "Detection plus rollback, safe retry and permanent correction",
    }
