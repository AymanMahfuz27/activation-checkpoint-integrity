"""Tensor/state/control fixtures; imports and execution require allocation."""

import copy
import unittest

from analysis_core import ARMS, require_allocation
from state_parity import compare_cpu_trees, evaluate_controls, retained_state


def fixture():
    require_allocation()
    import torch

    initial = dict(updates=210, cursor=210 * 65536)
    outcomes = [
        dict(
            history=1234,
            arm=arm,
            status="CENSORED_WALL_LIMIT",
            validation_status="PASS",
            accepted_updates=1,
            first_detection_update=1 if arm == "buggy-guarded-repaired" else None,
        )
        for arm in ARMS
    ]
    state = dict(
        schema=1,
        model={"weight": torch.tensor([1.0, -0.0])},
        optimizer={0: {"step": torch.tensor(1.0)}},
        scheduler={"last_epoch": 1},
        cursor=211 * 65536,
        updates=211,
        rng={"cpu": torch.tensor([1, 2], dtype=torch.uint8), "cuda": []},
    )
    buffers = {"rotary.inv_freq": torch.tensor([0.5])}
    states = {arm: (copy.deepcopy(state), copy.deepcopy(buffers)) for arm in ARMS}
    proof = dict(
        valid=True,
        all_finite=True,
        byte_equal=True,
        value_equal=True,
        tensors=[{"name": "full-model-and-buffers"}],
    )
    parity = dict(
        clean_checking_noninterference=dict(proof),
        first_recovery_equals_clean_full_state_rng=dict(proof),
    )
    return initial, outcomes, states, parity


class StateParityFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def test_all_saved_fields_and_named_buffers_compare(self):
        initial, outcomes, states, parity = fixture()
        result = evaluate_controls(1234, outcomes, parity, states, initial)
        self.assertTrue(result["clean_checking_noninterference_supported"])
        self.assertEqual(result["immediate_recovery_first_update"]["status"], "PASS")

    def test_empty_or_missing_recorded_parity_is_not_pass(self):
        initial, outcomes, states, _ = fixture()
        for parity in (
            {},
            {
                "clean_checking_noninterference": {
                    "valid": True,
                    "all_finite": True,
                    "byte_equal": True,
                    "value_equal": True,
                    "tensors": [],
                }
            },
        ):
            self.assertFalse(
                evaluate_controls(1234, outcomes, parity, states, initial)[
                    "clean_checking_noninterference_supported"
                ]
            )

    def test_named_nonpersistent_buffer_difference_is_not_hidden(self):
        initial, outcomes, states, parity = fixture()
        states["clean-checked"][1]["rotary.inv_freq"].add_(1.0)
        result = evaluate_controls(1234, outcomes, parity, states, initial)
        self.assertFalse(result["clean_checking_noninterference_supported"])
        self.assertEqual(
            result["matched_final_state_buffers_rng"][
                "clean_checked_vs_clean_disabled"
            ]["status"],
            "STATE_DIFFERENCE_RETAINED",
        )

    def test_different_endpoints_and_delayed_detection_are_explicit_not_applicable(
        self,
    ):
        initial, outcomes, states, parity = fixture()
        next(row for row in outcomes if row["arm"] == "clean-checked")[
            "accepted_updates"
        ] = 2
        next(row for row in outcomes if row["arm"] == "buggy-guarded-repaired")[
            "first_detection_update"
        ] = 2
        result = evaluate_controls(1234, outcomes, parity, states, initial)
        self.assertFalse(result["clean_checking_noninterference_supported"])
        self.assertFalse(result["immediate_recovery_first_update"]["applicable"])
        self.assertEqual(
            result["matched_final_state_buffers_rng"][
                "clean_checked_vs_clean_disabled"
            ]["status"],
            "NOT_APPLICABLE_DIFFERENT_ACCEPTED_ENDPOINTS",
        )

    def test_failed_arm_equal_state_remains_descriptive(self):
        initial, outcomes, states, parity = fixture()
        next(row for row in outcomes if row["arm"] == "clean-checked")["status"] = (
            "FAILED_ARM_RETAINED"
        )
        result = evaluate_controls(1234, outcomes, parity, states, initial)
        self.assertFalse(result["clean_checking_noninterference_supported"])
        self.assertEqual(
            result["matched_final_state_buffers_rng"][
                "clean_checked_vs_clean_disabled"
            ]["status"],
            "DESCRIPTIVE_FAILED_ARM_COMPARISON",
        )

    def test_rng_and_optimizer_and_counter_differences_are_visible(self):
        initial, outcomes, states, parity = fixture()
        for field in ("rng", "optimizer", "scheduler"):
            left = retained_state(*states["clean-disabled"])
            right = copy.deepcopy(left)
            right[field] = {"changed": True}
            self.assertFalse(compare_cpu_trees(left, right)["byte_equal"])
        states["clean-checked"][0]["cursor"] += 1
        with self.assertRaisesRegex(ValueError, "counters"):
            evaluate_controls(1234, outcomes, parity, states, initial)

    def test_invalid_trajectory_cannot_establish_clean_noninterference(self):
        initial, outcomes, states, parity = fixture()
        next(row for row in outcomes if row["arm"] == "clean-checked")[
            "validation_status"
        ] = "FAILED_VALIDATION_RETAINED"
        result = evaluate_controls(1234, outcomes, parity, states, initial)
        self.assertFalse(result["clean_checking_noninterference_supported"])
        self.assertEqual(
            result["matched_final_state_buffers_rng"][
                "clean_checked_vs_clean_disabled"
            ]["status"],
            "DESCRIPTIVE_UNVALIDATED_ARM_COMPARISON",
        )

    def test_signed_zero_nonfinite_shape_and_key_types_are_separate(self):
        import torch

        self.assertTrue(
            compare_cpu_trees(torch.tensor([-0.0]), torch.tensor([0.0]))["value_equal"]
        )
        self.assertFalse(
            compare_cpu_trees(torch.tensor([-0.0]), torch.tensor([0.0]))["byte_equal"]
        )
        self.assertFalse(
            compare_cpu_trees(
                torch.tensor([float("nan")]), torch.tensor([float("nan")])
            )["all_finite"]
        )
        self.assertFalse(
            compare_cpu_trees(torch.tensor([1.0]), torch.tensor([[1.0]]))["valid"]
        )
        self.assertFalse(compare_cpu_trees({1: "same"}, {1.0: "same"})["valid"])

    def test_scalar_nonfinite_classification_is_symmetric(self):
        for finite, nonfinite in (
            (1.0, float("inf")),
            (1.0, float("nan")),
            (complex(1, 2), complex(1, float("inf"))),
            (complex(1, 2), complex(float("nan"), 2)),
        ):
            for a, b in ((finite, nonfinite), (nonfinite, finite)):
                result = compare_cpu_trees(a, b)
                self.assertFalse(result["all_finite"])
                self.assertFalse(result["value_equal"])
                self.assertEqual(result["nonfinite_paths"], ["root"])


if __name__ == "__main__":
    require_allocation()
    unittest.main()
