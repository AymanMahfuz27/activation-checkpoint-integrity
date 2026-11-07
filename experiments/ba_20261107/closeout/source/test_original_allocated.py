"""Original-study census/ERROR semantics fixtures, allocated only."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from analysis_core import require_allocation
from analysis_original import (
    active_at,
    analyze_e1,
    analyze_e2_not_reached,
    analyze_e3,
    selected_at,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
INITIAL = dict(updates=10, cursor=655360)
EQUAL = dict(valid=True, all_finite=True, byte_equal=True, value_equal=True)


def frozen(name):
    return json.loads((CONFIGS / name).read_text())


def error_cells():
    schedules = frozen("e3-schedules.json")
    records, members = [], {}
    for cell in schedules:
        identity = f"E3-{cell['policy']}-S{cell['seed']}-{cell['onset']}"
        active = active_at(cell["onset"], 1)
        selected = selected_at(cell["policy"], cell["seed"], 1)
        row = dict(
            id=identity + "-A001",
            branch_update=1,
            trigger_active=active,
            selected=selected,
            policy=cell["policy"],
            seed=cell["seed"],
            preserve_rng=not active,
            status="ERROR",
            phase="OPTIMIZER_ENTERED",
            actual_wrong_accepted_update=None,
            attempt_total_wall_seconds=1.0,
            reference_status="REJECTED_NO_COMMITTED_TRANSITION"
            if active
            else "NOT_REQUIRED_TRIGGER_OFF",
        )
        if active:
            relative = identity + "/forward-tape-001.pt"
            row["forward_tape_receipt"] = dict(
                path=relative,
                sha256="0" * 64,
                prestate_cursor=INITIAL["cursor"],
                prestate_updates=INITIAL["updates"],
            )
            members["E3/" + relative] = dict(sha256="0" * 64, bytes=0)
        summary = dict(
            id=identity,
            cell=cell,
            status="INCOMPLETE_ERROR_OR_COUNTERFACTUAL",
            attempts=1,
            accepted_updates=0,
            accepted_targets=0,
            selected_active_opportunities=int(active and selected),
            independent_history_count=1,
            unsupported_counterfactuals=0,
            actual_wrong_accepted_updates=0,
            first_proven_wrong_transition=None,
            first_alert=None,
            delay_attempts_from_trigger_onset=None,
            delay_accepted_updates_from_trigger_onset=None,
            delay_accepted_targets_from_trigger_onset=None,
            delay_from_first_wrong=None,
        )
        records.append(dict(summary=summary, attempts=[row]))
    return schedules, records, INITIAL, members


def finite_cell(kind="wrong"):
    schedules, records, initial, members = error_cells()
    record = next(
        row
        for row in records
        if row["summary"]["cell"]["policy"] == "every"
        and row["summary"]["cell"]["onset"] == "immediate"
    )
    first = record["attempts"][0]
    first["forward_tape_receipt"]["microbatches"] = 4
    first.update(
        status="ACCEPTED",
        phase="ACCEPTED",
        update=dict(
            update=11, accepted_tokens=initial["cursor"] + 8192, tokens_this_update=8192
        ),
        forward_loss_parity=True,
        recorded_forward_microbatches=4,
        corrected_reference=dict(
            status="ACCEPTED",
            phase="ACCEPTED",
            branch_update=1,
            policy="disabled",
            preserve_rng=True,
            update=dict(
                update=11,
                accepted_tokens=initial["cursor"] + 8192,
                tokens_this_update=8192,
            ),
        ),
        corrected_tape_use=dict(
            first["forward_tape_receipt"], same_prestate=True, preserve_rng=True
        ),
        transition_difference=dict(
            valid=True,
            all_finite=kind != "nonfinite",
            value_equal=kind in ("equal", "byte-only"),
            byte_equal=kind == "equal",
        ),
        reference_status="NONFINITE_OR_UNSUPPORTED_TRANSITION"
        if kind == "nonfinite"
        else "SAME_PRESTATE_AND_ACTUAL_FORWARD_TAPE",
        actual_wrong_accepted_update=None if kind == "nonfinite" else kind == "wrong",
    )
    record["summary"].update(
        accepted_updates=1,
        accepted_targets=8192,
        actual_wrong_accepted_updates=None
        if kind == "nonfinite"
        else int(kind == "wrong"),
        first_proven_wrong_transition=1 if kind == "wrong" else None,
    )
    if kind == "nonfinite":
        record["summary"]["unsupported_counterfactuals"] = 1
    else:
        second = copy.deepcopy(first)
        second.update(
            id=record["summary"]["id"] + "-A002",
            branch_update=2,
            status="REJECTED_PRE_OPTIMIZER",
            phase="BACKWARD_COMPLETE",
            actual_wrong_accepted_update=None,
            protected_state=dict(EQUAL),
            reference_status="REJECTED_NO_COMMITTED_TRANSITION",
        )
        second.pop("update")
        second["forward_tape_receipt"].update(
            path=record["summary"]["id"] + "/forward-tape-002.pt",
            prestate_updates=11,
            prestate_cursor=initial["cursor"] + 8192,
        )
        members["E3/" + second["forward_tape_receipt"]["path"]] = dict(
            sha256="0" * 64, bytes=0
        )
        record["attempts"].append(second)
        record["summary"].update(
            status="REJECTED_PRE_OPTIMIZER",
            attempts=2,
            first_alert=2,
            selected_active_opportunities=2,
            delay_attempts_from_trigger_onset=1,
            delay_accepted_updates_from_trigger_onset=1,
            delay_accepted_targets_from_trigger_onset=8192,
            delay_from_first_wrong=1 if kind == "wrong" else None,
        )
    return schedules, records, initial, members, record


def full_unchecked_cells():
    """Independent300-update prefixes for all3 onset patterns, no selected checks."""
    require_allocation()
    schedules, records, initial, members = error_cells()
    _, _, _, _, prototype = finite_cell("wrong")
    template = prototype["attempts"][0]
    for record in records:
        cell = record["summary"]["cell"]
        if cell["policy"] != "disabled":
            continue
        identity = record["summary"]["id"]
        rows = []
        wrong = 0
        first = None
        for index in range(1, 301):
            active = cell["onset"] == "immediate" or (
                index >= 11 and (cell["onset"] == "delayed" or (index - 11) % 4 == 0)
            )
            row = copy.deepcopy(template)
            row.update(
                id=f"{identity}-A{index:03}",
                branch_update=index,
                policy="disabled",
                seed=cell["seed"],
                selected=False,
                trigger_active=active,
                preserve_rng=not active,
                actual_wrong_accepted_update=active,
            )
            row["update"] = dict(
                update=10 + index,
                accepted_tokens=initial["cursor"] + index * 8192,
                tokens_this_update=8192,
            )
            if active:
                tape = dict(
                    path=f"{identity}/forward-tape-{index:03}.pt",
                    sha256="0" * 64,
                    prestate_updates=9 + index,
                    prestate_cursor=initial["cursor"] + (index - 1) * 8192,
                    microbatches=4,
                )
                row["forward_tape_receipt"] = tape
                row["corrected_tape_use"] = {
                    **tape,
                    "same_prestate": True,
                    "preserve_rng": True,
                }
                row["corrected_reference"].update(
                    branch_update=index, update=copy.deepcopy(row["update"])
                )
                members["E3/" + tape["path"]] = dict(sha256="0" * 64, bytes=0)
                wrong += 1
                if first is None:
                    first = index
            else:
                for key in (
                    "forward_tape_receipt",
                    "corrected_tape_use",
                    "corrected_reference",
                    "transition_difference",
                    "forward_loss_parity",
                    "recorded_forward_microbatches",
                ):
                    row.pop(key, None)
                row["reference_status"] = "NOT_REQUIRED_TRIGGER_OFF"
            rows.append(row)
        record["attempts"] = rows
        record["summary"].update(
            status="CENSORED_AT300",
            attempts=300,
            accepted_updates=300,
            accepted_targets=300 * 8192,
            actual_wrong_accepted_updates=wrong,
            first_proven_wrong_transition=first,
            first_alert=None,
            selected_active_opportunities=0,
            unsupported_counterfactuals=0,
        )
    return schedules, records, initial, members


class OriginalStudyFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_full300_unchecked_onsets_retain_every_accepted_wrong_or_clean_update(self):
        result = analyze_e3(*full_unchecked_cells())
        unchecked = [
            row for row in result["all_cells"] if row["cell"]["policy"] == "disabled"
        ]
        self.assertEqual(len(unchecked), 3)
        for row in unchecked:
            self.assertEqual(row["status"], "CENSORED_AT300")
            self.assertEqual(row["validation_issues"], [])
            self.assertEqual(len(row["attempts"]), 300)
            onset = row["cell"]["onset"]
            self.assertEqual(
                row["summary"]["actual_wrong_accepted_updates"],
                300 if onset == "immediate" else 290 if onset == "delayed" else 73,
            )
            self.assertEqual(
                row["summary"]["first_proven_wrong_transition"],
                1 if onset == "immediate" else 11,
            )
            self.assertIsNone(row["summary"]["first_alert"])
            self.assertFalse(
                any(
                    item["validated_pre_optimizer_rejection"]
                    for item in row["attempts"]
                )
            )

    def test_randomized_seed_selection_is_independent_reference_and_paired_across_onsets(
        self,
    ):
        cells = [
            row for row in frozen("e3-schedules.json") if row["policy"] == "randomized"
        ]
        seeds = {row["seed"] for row in cells}
        self.assertEqual(len(seeds), 8)
        self.assertEqual(len(cells), 24)
        for seed in seeds:
            paired = [row for row in cells if row["seed"] == seed]
            self.assertEqual(
                {row["onset"] for row in paired},
                {"immediate", "delayed", "intermittent"},
            )
            wanted = [
                int.from_bytes(
                    hashlib.sha256(f"aci-checking-v1:{seed}:{index}".encode()).digest()[
                        :8
                    ],
                    "big",
                )
                / (2**64)
                < 0.1
                for index in range(1, 301)
            ]
            for row in paired:
                self.assertEqual(
                    [
                        selected_at(row["policy"], row["seed"], index)
                        for index in range(1, 301)
                    ],
                    wanted,
                )

    def test_rejection_with_contradictory_identity_selection_or_tape_is_unvalidated(
        self,
    ):
        for change in ("identity", "selection", "tape"):
            schedules, records, initial, members, record = finite_cell()
            rejected = record["attempts"][-1]
            if change == "identity":
                rejected["id"] = "foreign-A002"
            elif change == "selection":
                rejected["selected"] = False
            else:
                rejected["forward_tape_receipt"]["sha256"] = "f" * 64
            result = analyze_e3(schedules, records, initial, members)
            cell = next(
                row
                for row in result["all_cells"]
                if row["id"] == record["summary"]["id"]
            )
            last = cell["attempts"][-1]
            self.assertEqual(last["authoritative_status"], "REJECTED_PRE_OPTIMIZER")
            self.assertTrue(last["native_pre_optimizer_preservation"])
            self.assertFalse(last["validated_pre_optimizer_rejection"])
            self.assertTrue(last["validation_issues"])

    def test_e1_all_registered_missing_calls_are_retained(self):
        result = analyze_e1(frozen("e1-attempts-v2.json"), [])
        self.assertEqual(len(result["all_calls"]), 638)
        self.assertEqual(result["missing_calls"], 638)
        self.assertFalse(result["clean_controls_valid"])

    def test_e1_failed_clean_metadata_does_not_become_corruption_claim(self):
        registration = frozen("e1-attempts-v2.json")
        observations = [
            dict(
                id="E1-early-warm10-plain-clean",
                stage="early-warm10",
                status="ACCEPTED",
            ),
            dict(
                id="E1-early-warm10-enforce-boundary-off",
                stage="early-warm10",
                status="ACCEPTED",
                model_noninterference={"valid": False, "reason": "tensor_metadata"},
                gradient_noninterference=dict(EQUAL),
            ),
        ]
        result = analyze_e1(registration, observations)
        self.assertEqual(result["observed_calls"], 2)
        self.assertEqual(result["missing_calls"], 636)
        self.assertFalse(result["clean_controls_valid"])

    def test_e1_duplicate_or_foreign_call_rejected(self):
        for rows in (
            [dict(id="foreign")],
            [
                dict(
                    id="E1-early-warm10-plain-clean",
                    stage="early-warm10",
                    status="ACCEPTED",
                )
            ]
            * 2,
        ):
            with self.assertRaises(ValueError):
                analyze_e1(frozen("e1-attempts-v2.json"), rows)

    def test_e2_not_reached_preserves15arms_and_no_population_ci(self):
        arms = frozen("original-contract.json")["e2"]["arms"]
        result = analyze_e2_not_reached(arms, [])
        self.assertEqual(len(result["all_arms"]), 15)
        self.assertIsNone(result["primary_population_intervals"])
        with self.assertRaisesRegex(ValueError, "Unexpected E2"):
            analyze_e2_not_reached(arms, ["unexpected.json"])

    def test_e3_error_convenience_string_never_counts_as_rejection(self):
        result = analyze_e3(*error_cells())
        self.assertEqual(result["attempt_rows_validated"], 39)
        self.assertTrue(
            all(
                not row["attempts"][0]["validated_pre_optimizer_rejection"]
                for row in result["all_cells"]
            )
        )
        self.assertTrue(
            all(
                row["attempts"][0]["authoritative_status"] == "ERROR"
                for row in result["all_cells"]
            )
        )

    def test_e3_valid_rejection_requires_authoritative_phase_and_preservation(self):
        for valid_phase in (True, False):
            schedules, records, initial, members = error_cells()
            record = next(
                row
                for row in records
                if row["summary"]["cell"]["policy"] == "every"
                and row["summary"]["cell"]["onset"] == "immediate"
            )
            record["attempts"][0].update(
                status="REJECTED_PRE_OPTIMIZER",
                phase="BACKWARD_COMPLETE" if valid_phase else "OPTIMIZER_ENTERED",
                protected_state=dict(EQUAL),
            )
            record["summary"].update(
                status="REJECTED_PRE_OPTIMIZER",
                first_alert=1,
                delay_attempts_from_trigger_onset=0,
                delay_accepted_updates_from_trigger_onset=0,
                delay_accepted_targets_from_trigger_onset=0,
            )
            result = analyze_e3(schedules, records, initial, members)
            actual = next(
                row
                for row in result["all_cells"]
                if row["id"] == record["summary"]["id"]
            )
            self.assertEqual(
                actual["attempts"][0]["validated_pre_optimizer_rejection"], valid_phase
            )

    def test_e3_missing_journal_and_duplicate_cells_remain_visible_or_reject(self):
        schedules, records, initial, members = error_cells()
        records[0]["attempts"] = []
        result = analyze_e3(schedules, records, initial, members)
        self.assertGreater(result["validation_issues"], 0)
        records[-1] = copy.deepcopy(records[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_e3(schedules, records, initial, members)

    def test_e1_clean_validity_depends_on_registered_state_and_treatment(self):
        registration = frozen("e1-attempts-v2.json")
        controls = []
        for planned in registration:
            mode = planned["mode"]
            if mode not in ("plain-clean", "enforce-boundary-off", "enforce-all-off"):
                continue
            controls.append(
                dict(
                    id=planned["id"],
                    stage=planned["state"],
                    status="ACCEPTED",
                    phase="ACCEPTED",
                    policy="disabled" if mode == "plain-clean" else "every",
                    scope="all_operators"
                    if mode == "enforce-all-off"
                    else "checkpoint_boundaries",
                    preserve_rng=True,
                    seed=1729,
                    spec=None if mode == "plain-clean" else dict(enabled=False),
                    model_noninterference=dict(EQUAL),
                    gradient_noninterference=dict(EQUAL),
                )
            )
        self.assertTrue(analyze_e1(registration, controls)["clean_controls_valid"])
        controls[0]["stage"] = "wrong-state"
        self.assertFalse(analyze_e1(registration, controls)["clean_controls_valid"])

    def test_e1_native_rejection_is_distinct_from_reference_validated_claim(self):
        registration = frozen("e1-attempts-v2.json")
        planned = next(row for row in registration if row["mode"] == "enforce-boundary")
        row = dict(
            id=planned["id"],
            stage=planned["state"],
            policy="every",
            scope="checkpoint_boundaries",
            seed=1729,
            preserve_rng=True,
            spec=dict(case_id=planned["case_id"]),
            status="REJECTED_PRE_OPTIMIZER",
            phase="BACKWARD_COMPLETE",
            protected_state=dict(EQUAL),
            original_reference_valid=True,
            event_bytes_match=True,
            noninterference_valid=True,
            observer_noninterference=dict(EQUAL),
            observation_classification=dict(status="SUPPORTED"),
        )
        result = analyze_e1(registration, [row])
        actual = next(
            r for r in result["all_calls"] if r["registration"]["id"] == row["id"]
        )
        self.assertTrue(actual["validated_pre_optimizer_rejection"])
        row["event_bytes_match"] = False
        actual = next(
            r
            for r in analyze_e1(registration, [row])["all_calls"]
            if r["registration"]["id"] == row["id"]
        )
        self.assertTrue(actual["native_pre_optimizer_preservation"])
        self.assertFalse(actual["validated_pre_optimizer_rejection"])

    def test_e3_finite_wrong_equal_byte_only_and_nonfinite_are_separate(self):
        for kind in ("wrong", "equal", "byte-only", "nonfinite"):
            schedules, records, initial, members, record = finite_cell(kind)
            result = analyze_e3(schedules, records, initial, members)
            actual = next(
                row
                for row in result["all_cells"]
                if row["id"] == record["summary"]["id"]
            )
            row = actual["attempts"][0]
            self.assertEqual(
                row["finite_actual_wrong_accepted_update"],
                None if kind == "nonfinite" else kind == "wrong",
            )
            self.assertEqual(
                row["byte_only_transition_difference"], kind == "byte-only"
            )
            self.assertEqual(actual["validation_issues"], [])
            if kind != "nonfinite":
                self.assertEqual(
                    actual["summary"]["delay_from_first_wrong"],
                    1 if kind == "wrong" else None,
                )

    def test_e3_wrong_labels_require_exact_tape_and_primary_counter_bindings(self):
        for mutate in (
            lambda row: row["corrected_tape_use"].update(sha256="f" * 64),
            lambda row: row["corrected_tape_use"].update(path="foreign.pt"),
            lambda row: row["corrected_tape_use"].update(microbatches=3),
            lambda row: row["corrected_reference"]["update"].update(update=999),
            lambda row: row["update"].update(accepted_tokens=999),
            lambda row: row.update(phase="OPTIMIZER_ENTERED"),
        ):
            schedules, records, initial, members, record = finite_cell()
            mutate(record["attempts"][0])
            actual = next(
                row
                for row in analyze_e3(schedules, records, initial, members)["all_cells"]
                if row["id"] == record["summary"]["id"]
            )
            self.assertIsNone(
                actual["attempts"][0]["finite_actual_wrong_accepted_update"]
            )
            self.assertGreater(len(actual["validation_issues"]), 0)

    def test_e3_unknown_or_inconsistent_terminal_summary_is_not_accepted(self):
        for status in ("MADE_UP", "CENSORED_AT300", "FAILED_PROTECTED_STATE"):
            schedules, records, initial, members = error_cells()
            records[0]["summary"]["status"] = status
            result = analyze_e3(schedules, records, initial, members)
            self.assertGreater(len(result["all_cells"][0]["validation_issues"]), 0)


if __name__ == "__main__":
    require_allocation()
    unittest.main()
