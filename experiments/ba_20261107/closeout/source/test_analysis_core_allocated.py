"""Worker-schema and method fixtures; execute only in reviewed allocation.

Reference df=2 intervals use the closed-form Student-t CDF inversion rather
than the reducer's scipy.ppf implementation. No fixtures have run locally.
"""

import copy
import hashlib
import math
import unittest

from analysis_core import (
    ARMS,
    EVALUATIONS,
    HISTORIES,
    POLICIES,
    TARGETS,
    analyze_e4,
    analyze_e5,
    require_allocation,
)

E4_INITIAL = dict(updates=10, cursor=10 * 65536)
E5_INITIAL = {history: dict(updates=210, cursor=210 * 65536) for history in HISTORIES}
EQUAL = dict(valid=True, all_finite=True, byte_equal=True, value_equal=True)
CUTOFF = "Science cutoff reached; preserve partial evidence without retry"


def fixture_selected(policy, seed, update):
    if policy == "disabled":
        return False
    if policy == "startup":
        return update <= 5
    if policy == "every":
        return True
    if policy.startswith("period"):
        return update % int(policy[6:]) == 0
    key = f"aci-checking-v1:{seed}:{update}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / (2**64) < 0.1


def e4_fixture():
    blocks, pairs, arms = [], [], []
    for policy in POLICIES:
        measured = 100 if policy == "period100" else 30
        for index in range(10):
            job = 1 if index < 4 else 2 if index < 7 else 3
            block = dict(
                id=f"E4-{policy}-B{index + 1:02}",
                policy=policy,
                job=job,
                order=["disabled", policy],
                measured_per_arm=measured,
                warmup_per_arm=20,
                schedule_seed=91000 + index,
            )
            ratio = {1: 1.0, 2: 2.0, 3: 4.0}[job]
            blocks.append(block)
            pairs.append(
                dict(
                    id=block["id"],
                    policy=policy,
                    job=job,
                    order=block["order"],
                    full_loop_ratio=ratio,
                    disabled_seconds=10.0,
                    policy_seconds=ratio * 10.0,
                    measured_updates_per_arm=measured,
                    measured_policy_checks=sum(
                        fixture_selected(policy, block["schedule_seed"], 21 + offset)
                        for offset in range(measured)
                    ),
                    scope="checkpoint_boundaries",
                )
            )
            for arm, seconds in (("disabled", 10.0), (policy, ratio * 10.0)):
                value = dict(
                    block=block,
                    arm=arm,
                    full_loop_seconds=seconds,
                    measured_updates=measured,
                    measured_targets=65536 * measured,
                    measured_checked=sum(
                        fixture_selected(arm, block["schedule_seed"], 21 + offset)
                        for offset in range(measured)
                    ),
                    warmup_checked=sum(
                        fixture_selected(arm, block["schedule_seed"], update)
                        for update in range(1, 21)
                    ),
                    individual_updates=[
                        dict(
                            update=31 + offset,
                            accepted_tokens=(31 + offset) * 65536,
                            tokens_this_update=65536,
                            selected=fixture_selected(
                                arm, block["schedule_seed"], 21 + offset
                            ),
                            comparison=dict(
                                failed=False,
                                mode="fingerprint",
                                observation_scope="checkpoint_boundaries",
                                policy="bit_exact_fail_closed",
                                eligible_pairs=4,
                                exact_pairs=4,
                                counts={"exact_match": 4},
                            )
                            if fixture_selected(
                                arm, block["schedule_seed"], 21 + offset
                            )
                            else None,
                        )
                        for offset in range(measured)
                    ],
                )
                if arm != "disabled":
                    value["paired_full_state_and_rng"] = dict(EQUAL)
                arms.append(value)
    return pairs, arms, blocks, E4_INITIAL


def e5_fixture(history, arm, *, status="TARGET_CONFIRMED", accepted=None):
    limit = (
        accepted
        if accepted is not None
        else 40
        if status == "TARGET_CONFIRMED"
        else 290
    )
    evaluations = []
    for update in EVALUATIONS:
        if update > limit:
            break
        qualifies = status == "TARGET_CONFIRMED" and update > 0
        ce = TARGETS[history] - 0.1 if qualifies else TARGETS[history] + 0.2
        evaluations.append(
            dict(
                branch_update=update,
                elapsed_seconds=float(update + 1),
                cross_entropy=ce,
                heldout_tokens=1048576,
                status="PASS",
                state_preservation=dict(EQUAL),
            )
        )
        if update:
            evaluations[-1].update(
                qualifies=qualifies, confirmation=qualifies and update >= 40
            )
    summary = dict(
        history=history,
        arm=arm,
        status=status,
        accepted_updates=limit,
        accepted_targets=65536 * limit,
        target={"cross_entropy_target": TARGETS[history]},
        confirmed_target=dict(update=limit, elapsed_seconds=float(limit + 2))
        if status == "TARGET_CONFIRMED"
        else None,
        observed_seconds=float(limit + 3),
        evaluation_seconds=1.0,
        attempt_loop_seconds=float(limit) + 1.0,
        rejected_attempt_compute_seconds=0.0,
        rollback_seconds=0.0,
        safe_retry_seconds=0.0,
        experimental_restore_seconds=5.0,
        final_retention_seconds=2.0,
        rejection_count=0,
        first_detection_update=None,
        permanent_correction_applied=False,
        error=None,
    )
    attempts = [
        dict(
            history=history,
            arm=arm,
            branch_update=update,
            attempt_ordinal=0,
            status="ACCEPTED",
            phase="ACCEPTED",
            wall_seconds=0.01,
            elapsed_seconds=float(update) + 0.5,
            policy="every"
            if arm in ("clean-checked", "buggy-guarded-repaired")
            else "disabled",
            preserve_rng=arm not in ("buggy-disabled", "buggy-guarded-repaired"),
            update=dict(
                update=210 + update,
                accepted_tokens=(210 + update) * 65536,
                tokens_this_update=65536,
            ),
        )
        for update in range(1, limit + 1)
    ]
    return dict(summary=summary, evaluations=evaluations, attempts=attempts)


def recovery_fixture(history=1234):
    record = e5_fixture(history, "buggy-guarded-repaired")
    rejected = dict(
        history=history,
        arm="buggy-guarded-repaired",
        branch_update=1,
        attempt_ordinal=0,
        status="REJECTED_PRE_OPTIMIZER",
        phase="BACKWARD_COMPLETE",
        policy="every",
        preserve_rng=False,
        wall_seconds=2.0,
        elapsed_seconds=7.0,
        rollback_seconds=1.0,
        protected_state_preservation=dict(EQUAL),
        full_rollback_parity=dict(EQUAL),
        permanent_correction_selected=True,
    )
    retry = record["attempts"][0]
    retry.update(attempt_ordinal=1, is_single_safe_retry=True, wall_seconds=3.0)
    for attempt in record["attempts"]:
        attempt["preserve_rng"] = True
        attempt["elapsed_seconds"] = (
            7.1 + (attempt["branch_update"] - 1) / 10
            if attempt["branch_update"] <= 20
            else 21.1 + (attempt["branch_update"] - 21) / 10
        )
    record["attempts"].insert(0, rejected)
    record["summary"].update(
        rejection_count=1,
        first_detection_update=1,
        permanent_correction_applied=True,
        rejected_attempt_compute_seconds=2.0,
        rollback_seconds=1.0,
        safe_retry_seconds=3.0,
        attempt_loop_seconds=10.0,
    )
    return record


def all_records():
    return [e5_fixture(history, arm) for history in HISTORIES for arm in ARMS]


def replace_record(rows, record):
    summary = record["summary"]
    for index, row in enumerate(rows):
        if (row["summary"]["history"], row["summary"]["arm"]) == (
            summary["history"],
            summary["arm"],
        ):
            rows[index] = record
            return
    raise AssertionError("fixture outcome not registered")


class MethodFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_e4_measured_selection_pair_counts_and_selected_comparison_are_bound(self):
        for change in (
            "selection",
            "disabled_count",
            "warmup_count",
            "comparison",
            "pair_count",
        ):
            pairs, arms, blocks, initial = e4_fixture()
            checked = next(row for row in arms if row["arm"] == "every")
            if change == "selection":
                checked["individual_updates"][0]["selected"] = False
            elif change == "disabled_count":
                next(row for row in arms if row["arm"] == "disabled")[
                    "measured_checked"
                ] = 1
            elif change == "warmup_count":
                checked["warmup_checked"] = 0
            elif change == "comparison":
                checked["individual_updates"][0]["comparison"]["failed"] = True
            else:
                next(row for row in pairs if row["id"] == checked["block"]["id"])[
                    "measured_policy_checks"
                ] = 0
            with self.assertRaises(ValueError):
                analyze_e4(pairs, arms, blocks, initial)

    def test_e4_equal_job_weight_and_independent_nontrivial_interval(self):
        result = analyze_e4(*e4_fixture())
        # For df2: F(t)=1/2+t/(2*sqrt(t*t+2)). At upper p=.975,
        # u=2p-1=.95 and q=sqrt(2*u*u/(1-u*u)). No scipy reference.
        q = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
        half = q * math.log(2) / math.sqrt(3)
        for policy in POLICIES:
            interval = result["primary"][policy]
            self.assertAlmostEqual(interval["geometric_ratio"], 2.0)
            self.assertAlmostEqual(interval["lower"], math.exp(math.log(2) - half))
            self.assertAlmostEqual(interval["upper"], math.exp(math.log(2) + half))
            self.assertEqual(interval["pairs_per_job"], [4, 3, 3])
            self.assertEqual(interval["log_interval"]["df"], 2)
        self.assertEqual(result["registered_updates"], 6400)

    def test_e4_missing_or_duplicate_pair_rejected(self):
        pairs, arms, blocks, counters = e4_fixture()
        pairs[-1] = copy.deepcopy(pairs[0])
        with self.assertRaisesRegex(ValueError, "Missing, duplicate"):
            analyze_e4(pairs, arms, blocks, counters)

    def test_e4_startup_measured_check_rejected(self):
        pairs, arms, blocks, counters = e4_fixture()
        arms[1]["measured_checked"] = 1
        with self.assertRaisesRegex(
            ValueError, "selected-check treatment/counters differ"
        ):
            analyze_e4(pairs, arms, blocks, counters)

    def test_e4_startup_journal_selection_cannot_move_into_primary(self):
        pairs, arms, blocks, counters = e4_fixture()
        arm = next(row for row in arms if row["arm"] == "startup")
        arm["individual_updates"][0]["selected"] = True
        arm["measured_checked"] = 1
        with self.assertRaisesRegex(
            ValueError, "selected-check treatment/counters differ"
        ):
            analyze_e4(pairs, arms, blocks, counters)

    def test_e4_shifted_contiguous_counters_cannot_claim_warmup(self):
        pairs, arms, blocks, counters = e4_fixture()
        for update in arms[0]["individual_updates"]:
            update["update"] -= 20
            update["accepted_tokens"] -= 20 * 65536
        with self.assertRaisesRegex(ValueError, "absolute update/token"):
            analyze_e4(pairs, arms, blocks, counters)

    def test_e5_second_postbaseline_uses_post_fsync_summary_time(self):
        result = analyze_e5(all_records(), E5_INITIAL)
        self.assertTrue(
            all(row["validation_status"] == "PASS" for row in result["outcomes"])
        )
        self.assertTrue(
            all(row["confirmed_seconds"] == 42.0 for row in result["outcomes"])
        )
        self.assertEqual(
            result["contrasts"]["recovery_policy_vs_buggy"]["population_interval"][
                "ratio"
            ],
            1.0,
        )

    def test_e5_wrong_treatment_identity_and_committed_counters_rejected(self):
        for mutate in (
            lambda row: row.update(preserve_rng=True),
            lambda row: row.update(history=99),
            lambda row: row.update(policy="every"),
            lambda row: row.update(phase="OPTIMIZER_ENTERED"),
            lambda row: row["update"].update(update=999),
            lambda row: row["update"].update(accepted_tokens=999),
        ):
            rows = all_records()
            mutate(rows[1]["attempts"][0])  # buggy-disabled
            result = analyze_e5(rows, E5_INITIAL)
            self.assertIsNone(
                result["contrasts"]["recovery_policy_vs_buggy"]["population_interval"]
            )

    def test_e5_successful_guarded_reject_rollback_retry_permanent_correction(self):
        rows = all_records()
        replace_record(rows, recovery_fixture())
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][2]
        self.assertEqual(outcome["validation_status"], "PASS")
        self.assertTrue(outcome["safe_retry_accepted"])
        self.assertEqual(outcome["first_detection_update"], 1)
        self.assertEqual(outcome["cost"]["unallocated_in_clock_seconds"], 32.0)
        self.assertEqual(outcome["cost"]["rollback_seconds"], 1.0)
        self.assertEqual(outcome["cost"]["safe_retry_seconds"], 3.0)

    def test_e5_no_reversion_to_buggy_rng_after_retry(self):
        rows = all_records()
        record = recovery_fixture()
        record["attempts"][-1]["preserve_rng"] = False
        replace_record(rows, record)
        self.assertEqual(
            analyze_e5(rows, E5_INITIAL)["outcomes"][2]["validation_status"],
            "FAILED_VALIDATION_RETAINED",
        )

    def test_e5_missing_interior_evaluation_cannot_bridge_confirmation(self):
        rows = all_records()
        rows[1]["evaluations"].pop(1)
        result = analyze_e5(rows, E5_INITIAL)
        self.assertIsNone(result["outcomes"][1]["confirmed_seconds"])
        self.assertIsNone(
            result["contrasts"]["recovery_policy_vs_buggy"]["population_interval"]
        )

    def test_e5_ordinary_unreached_target_censor_has_no_speedup_or_ci(self):
        rows = all_records()
        replace_record(
            rows, e5_fixture(1234, "buggy-disabled", status="CENSORED_UPDATE_LIMIT")
        )
        result = analyze_e5(rows, E5_INITIAL)
        self.assertEqual(result["outcomes"][1]["validation_status"], "PASS")
        contrast = result["contrasts"]["recovery_policy_vs_buggy"]
        self.assertIsNone(contrast["histories"][0]["ratio"])
        self.assertIsNone(contrast["population_interval"])
        self.assertIsNotNone(contrast["histories"][1]["ratio"])

    def test_e5_exception_bound_missing_final_evaluation_retains_failed_bound(self):
        rows = all_records()
        record = e5_fixture(
            1234, "buggy-disabled", status="FAILED_ARM_RETAINED", accepted=20
        )
        record["evaluations"].pop()
        record["summary"]["error"] = "evaluation IO failed"
        record["terminal_evidence"] = dict(
            failure_json=dict(
                error="evaluation IO failed",
                traceback="evaluate(...) raised OSError: evaluation IO failed",
            )
        )
        replace_record(rows, record)
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][1]
        self.assertEqual(outcome["status"], "FAILED_ARM_RETAINED")
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )
        self.assertEqual(outcome["observed_bound_seconds"], 23.0)
        self.assertIsNone(outcome["confirmed_seconds"])

    def test_e5_cutoff_before_baseline_retains_no_e0_no_target(self):
        rows = all_records()
        record = e5_fixture(
            1234, "buggy-disabled", status="CENSORED_WALL_LIMIT", accepted=0
        )
        record["evaluations"] = []
        record["summary"]["error"] = CUTOFF
        replace_record(rows, record)
        self.assertEqual(
            analyze_e5(rows, E5_INITIAL)["outcomes"][1]["validation_status"],
            "PARTIAL_TERMINAL_STAGE_RETAINED",
        )

    def test_e5_qualifying_bytes_without_fsync_commit_not_confirmation(self):
        rows = all_records()
        record = rows[1]
        record["summary"].update(
            status="FAILED_ARM_RETAINED", confirmed_target=None, error="fsync failed"
        )
        record["terminal_evidence"] = dict(
            failure_json=dict(
                error="fsync failed",
                traceback="append -> os.fsync -> OSError: fsync failed",
            )
        )
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][1]
        self.assertEqual(outcome["numerically_qualifying_pair_update"], 40)
        self.assertIsNone(outcome["confirmed_seconds"])
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )

    def test_e5_native_timeout_does_not_invent_post_evaluation_operation(self):
        rows = all_records()
        rows[1]["summary"].update(
            status="CENSORED_WALL_LIMIT", confirmed_target=None, error=CUTOFF
        )
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][1]
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )
        self.assertEqual(
            outcome["terminal_stages"][0]["location_limit"],
            "UNRESOLVED_TIMEOUT_LOCATION",
        )
        self.assertIsNone(outcome["confirmed_seconds"])

    def test_e5_cutoff_after_rollback_before_retry_is_partial_not_corruption(self):
        rows = all_records()
        record = recovery_fixture()
        record["attempts"] = record["attempts"][:1]
        record["attempts"][0]["elapsed_seconds"] = 4.0
        record["evaluations"] = record["evaluations"][:1]
        record["summary"].update(
            status="CENSORED_WALL_LIMIT",
            accepted_updates=0,
            accepted_targets=0,
            confirmed_target=None,
            error=CUTOFF,
            safe_retry_seconds=0.0,
            observed_seconds=5.0,
            attempt_loop_seconds=3.5,
        )
        replace_record(rows, record)
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][2]
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )
        self.assertEqual(
            outcome["recovery_stage"], "ROLLBACK_COMPLETE_RETRY_NOT_REACHED"
        )
        self.assertFalse(outcome["safe_retry_observed"])
        self.assertFalse(outcome["safe_retry_accepted"])
        self.assertIsNone(outcome["confirmed_seconds"])

    def test_e5_failed_rollback_retains_experimental_gate_failure(self):
        rows = all_records()
        record = recovery_fixture()
        record["attempts"] = record["attempts"][:1]
        record["attempts"][0]["elapsed_seconds"] = 4.0
        record["attempts"][0].update(
            full_rollback_parity={"valid": False}, permanent_correction_selected=False
        )
        record["evaluations"] = record["evaluations"][:1]
        record["summary"].update(
            status="FAILED_ARM_RETAINED",
            accepted_updates=0,
            accepted_targets=0,
            confirmed_target=None,
            error="Full state/RNG rollback failed",
            permanent_correction_applied=False,
            safe_retry_seconds=0.0,
            observed_seconds=5.0,
            attempt_loop_seconds=3.5,
        )
        record["terminal_evidence"] = dict(
            failure_json=dict(
                error=record["summary"]["error"],
                traceback="guarded_attempt raised RuntimeError: Full state/RNG rollback failed",
            )
        )
        replace_record(rows, record)
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][2]
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )
        self.assertFalse(outcome["safe_retry_accepted"])

    def test_e5_nonzero_nested_costs_cannot_be_added_twice_or_go_unreconciled(self):
        rows = all_records()
        replace_record(rows, recovery_fixture())
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][2]
        cost = outcome["cost"]
        self.assertEqual(cost["observed_seconds"], 43.0)
        self.assertEqual(cost["unallocated_in_clock_seconds"], 32.0)
        rows[2]["summary"]["rollback_seconds"] = -1.0
        self.assertEqual(
            analyze_e5(rows, E5_INITIAL)["outcomes"][2]["validation_status"],
            "FAILED_VALIDATION_RETAINED",
        )

    def test_e5_single_safe_retry_can_reject_without_second_repair(self):
        rows = all_records()
        record = recovery_fixture()
        record["attempts"] = record["attempts"][:2]
        retry = record["attempts"][1]
        retry.pop("update")
        retry.update(
            status="REJECTED_PRE_OPTIMIZER",
            phase="BACKWARD_COMPLETE",
            wall_seconds=2.0,
            elapsed_seconds=7.2,
        )
        record["evaluations"] = record["evaluations"][:1]
        record["summary"].update(
            status="FAILED_ARM_RETAINED",
            accepted_updates=0,
            accepted_targets=0,
            confirmed_target=None,
            error="safe retry rejected",
            rejection_count=2,
            rejected_attempt_compute_seconds=4.0,
            safe_retry_seconds=2.0,
            attempt_loop_seconds=7.0,
            observed_seconds=8.0,
        )
        replace_record(rows, record)
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][2]
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )
        self.assertEqual(outcome["recovery_stage"], "SAFE_RETRY_REJECTED_RETAINED")
        self.assertTrue(outcome["legacy_correction_selected"])
        self.assertFalse(outcome["safe_retry_accepted"])
        self.assertIsNone(outcome["confirmed_seconds"])

    def test_e5_journal_clocks_must_fit_observed_bound_and_monotone_order(self):
        for mutate in (
            lambda record: record["evaluations"][-1].update(elapsed_seconds=999.0),
            lambda record: record["attempts"][-1].update(elapsed_seconds=999.0),
            lambda record: record["attempts"][-1].update(elapsed_seconds=0.0),
            lambda record: record["attempts"][0].update(wall_seconds=-1.0),
        ):
            rows = all_records()
            mutate(rows[1])
            self.assertEqual(
                analyze_e5(rows, E5_INITIAL)["outcomes"][1]["validation_status"],
                "FAILED_VALIDATION_RETAINED",
            )

    def test_e5_native_timeout_missing_evaluation_without_failure_json(self):
        rows = all_records()
        record = e5_fixture(
            1234, "buggy-disabled", status="CENSORED_WALL_LIMIT", accepted=20
        )
        record["evaluations"].pop()
        record["summary"]["error"] = (
            "TimeoutError while operating; native branch has no traceback"
        )
        replace_record(rows, record)
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][1]
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )
        self.assertIsNone(outcome["confirmed_seconds"])
        self.assertIn("unresolved", outcome["terminal_stages"][0]["location_limit"])

    def test_e5_later_corrected_attempt_rejection_is_retained_failure(self):
        rows = all_records()
        record = recovery_fixture()
        record["attempts"] = record["attempts"][
            :6
        ]  # first reject + five accepted transitions
        record["attempts"].append(
            dict(
                history=1234,
                arm="buggy-guarded-repaired",
                branch_update=6,
                attempt_ordinal=0,
                status="REJECTED_PRE_OPTIMIZER",
                phase="BACKWARD_COMPLETE",
                policy="every",
                preserve_rng=True,
                wall_seconds=0.25,
                elapsed_seconds=9.0,
            )
        )
        record["evaluations"] = record["evaluations"][:1]
        record["summary"].update(
            status="FAILED_ARM_RETAINED",
            accepted_updates=5,
            accepted_targets=5 * 65536,
            confirmed_target=None,
            error="later corrected attempt rejected",
            rejection_count=2,
            rejected_attempt_compute_seconds=2.25,
            attempt_loop_seconds=9.0,
            observed_seconds=10.0,
        )
        replace_record(rows, record)
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][2]
        self.assertEqual(
            outcome["validation_status"], "PARTIAL_TERMINAL_STAGE_RETAINED"
        )
        self.assertEqual(outcome["recovery_stage"], "POST_REPAIR_REJECTION_RETAINED")
        self.assertTrue(outcome["legacy_correction_selected"])
        self.assertTrue(
            outcome["safe_retry_accepted"]
        )  # historical accepted retry, not final success
        self.assertIsNone(outcome["confirmed_seconds"])

    def test_e5_cross_journal_clock_order_rejects_individually_monotone_streams(self):
        rows = all_records()
        record = recovery_fixture()
        for attempt in record["attempts"]:
            if attempt["branch_update"] >= 21:
                attempt["elapsed_seconds"] = 16.5 + (attempt["branch_update"] - 21) / 10
        replace_record(rows, record)
        outcome = analyze_e5(rows, E5_INITIAL)["outcomes"][2]
        self.assertEqual(outcome["validation_status"], "FAILED_VALIDATION_RETAINED")
        self.assertIn(
            "cross-journal attempt/evaluation chronology differs from worker order",
            outcome["issues"],
        )


if __name__ == "__main__":
    require_allocation()
    unittest.main()
