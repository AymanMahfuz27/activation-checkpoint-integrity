"""Allocated synthetic controls ensure additive labels never erase strict issues."""

import copy
import unittest

from analysis_core import require_allocation
from e1_disposition import CLIPPING_PREFIX, dispositions


def fixture(*, nonfinite=False, rejected=False):
    case = (
        "recompute-middle-nonfinite" if nonfinite else "gradient-L6-exponent-high-bit30"
    )
    planned = {
        "id": "synthetic-call",
        "state": "early-warm10",
        "case_id": case,
        "mode": "enforce-all" if rejected else "observe-all",
    }
    ref = {
        "status": "ERROR",
        "phase": "NONFINITE_CLIPPING_ERROR",
        "original_event_bytes_match": True,
        "original_status_match": True,
    }
    observed = {
        "status": "REJECTED_PRE_OPTIMIZER" if rejected else "ERROR",
        "phase": "BACKWARD_COMPLETE",
        "error": CLIPPING_PREFIX,
        "original_reference_valid": True,
        "event_bytes_match": True,
        "observation_classification": {"status": "SUPPORTED"},
        "observer_noninterference": {
            "valid": True,
            "byte_equal": True,
            "all_finite": not nonfinite,
            "value_equal": not nonfinite,
        },
    }
    issue = (
        "missing or invalid direct observer-gradient comparison"
        if rejected
        else "fault reference/events/noninterference/pairing not validated"
    )
    item = {
        "registration": planned,
        "status": observed["status"],
        "observation": observed,
        "native_pre_optimizer_preservation": rejected,
        "validation_issues": [issue],
    }
    return {
        "all_calls": [
            {
                "registration": {
                    **planned,
                    "id": "reference",
                    "mode": "plain-reconstruction",
                },
                "observation": ref,
                "validation_issues": [],
            },
            item,
        ]
    }


class E1DispositionFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_error_and_nonfinite_categories_leave_original_strict_state_unchanged(self):
        for nonfinite, rejected, expected in (
            (False, False, "FINITE_GRADIENT_CLIPPING_NORM_ERROR"),
            (True, False, "EXPLICIT_NONFINITE_REFERENCE_ERROR"),
            (True, True, "EXPLICIT_NONFINITE_REJECTION_STRICT_COMPARISON_UNMET"),
        ):
            with self.subTest(expected=expected):
                e1 = fixture(nonfinite=nonfinite, rejected=rejected)
                before = copy.deepcopy(e1)
                result = dispositions(e1)
                self.assertEqual(result["rows"][0]["disposition"], expected)
                self.assertEqual(e1, before)
                self.assertFalse(result["scientific_acceptance"])
                self.assertEqual(
                    result["rows"][0]["validation_issues"],
                    before["all_calls"][1]["validation_issues"],
                )

    def test_invalid_reference_or_events_cannot_be_explained_by_case_identity(self):
        for key in ("original_reference_valid", "event_bytes_match"):
            e1 = fixture(nonfinite=True)
            e1["all_calls"][1]["observation"][key] = False
            self.assertEqual(
                dispositions(e1)["rows"][0]["disposition"],
                "UNRESOLVED_REFERENCE_EVENT_OR_PAIRING",
            )

    def test_missing_paired_reference_error_stays_unresolved(self):
        e1 = fixture()
        e1["all_calls"][0]["observation"]["original_status_match"] = False
        self.assertEqual(
            dispositions(e1)["rows"][0]["disposition"],
            "UNRESOLVED_OTHER_STRICT_VALIDATION",
        )

    def test_finite_byte_difference_and_unpreserved_rejection_stay_unresolved(self):
        e1 = fixture()
        e1["all_calls"][1]["observation"]["observer_noninterference"]["byte_equal"] = (
            False
        )
        self.assertEqual(
            dispositions(e1)["rows"][0]["disposition"],
            "UNRESOLVED_REFERENCE_EVENT_OR_PAIRING",
        )
        e1 = fixture(nonfinite=True, rejected=True)
        e1["all_calls"][1]["native_pre_optimizer_preservation"] = False
        self.assertEqual(
            dispositions(e1)["rows"][0]["disposition"],
            "UNRESOLVED_OTHER_STRICT_VALIDATION",
        )

    def test_extra_issue_or_wrong_nonfinite_flag_cannot_receive_expected_label(self):
        e1 = fixture()
        e1["all_calls"][1]["validation_issues"].append(
            "recorded treatment differs from registered call role"
        )
        self.assertEqual(
            dispositions(e1)["rows"][0]["disposition"],
            "UNRESOLVED_OTHER_STRICT_VALIDATION",
        )
        e1 = fixture(nonfinite=True, rejected=True)
        e1["all_calls"][0]["observation"]["original_status_match"] = False
        self.assertEqual(
            dispositions(e1)["rows"][0]["disposition"],
            "UNRESOLVED_OTHER_STRICT_VALIDATION",
        )
        e1 = fixture(nonfinite=True)
        e1["all_calls"][1]["observation"]["observer_noninterference"]["all_finite"] = (
            True
        )
        self.assertEqual(
            dispositions(e1)["rows"][0]["disposition"],
            "UNRESOLVED_OTHER_STRICT_VALIDATION",
        )


if __name__ == "__main__":
    unittest.main()
