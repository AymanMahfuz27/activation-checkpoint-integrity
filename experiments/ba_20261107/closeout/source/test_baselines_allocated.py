"""Frozen baseline census and calibration fixtures; run only on allocation."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from analysis_core import require_allocation
from baseline_census import PREFIX, analyze_baselines


class SyntheticReader:
    def __init__(self):
        self.members, self.objects, self.journals = {}, {}, {}

    def store(self, name, value):
        name = PREFIX + name
        self.objects[name] = value
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.members[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }

    def json(self, name):
        return self.objects[name]

    def journal(self, name):
        return {
            "rows": self.journals[name],
            "partial_final_write": None,
            "status": "RETAINED",
        }


def fixture():
    require_allocation()
    reader = SyntheticReader()
    registration = json.loads(
        (Path(__file__).parent.parent / "configs/e3-abft-products.json").read_text()
    )
    initial = {"updates": 10, "cursor": 10 * 65536}
    eps = 2.0**-23
    spec = {
        "expected_names": ["w"],
        "norm_upper_bounds": [["w", 8.0 + eps]],
        "layer_norm_upper_bounds": [["layer", 8.0 + eps]],
        "layer_groups": [["layer", ["w"]]],
        "loss_upper_bound": 8.0 + eps,
        "clipping_limit": 1.0,
    }
    stages = {}
    for stage, targets in (("early-warm10", 8192), ("production-early-warm10", 65536)):
        threshold = {
            "status": "FROZEN_CLEAN_CALIBRATION",
            "stage": "early-warm10",
            "source_input_sha256": "a" * 64,
            "batch_targets": targets,
            "calibration_updates": list(range(1, 31)),
            "test_updates": list(range(31, 61)),
            "spec": copy.deepcopy(spec),
        }
        stages[stage] = threshold
        reader.store(stage + "/thresholds.json", threshold)
        reader.store(
            stage + "/clean-completion.json",
            {
                "status": "COMPLETE_CLEAN_CALIBRATION_AND_DISJOINT_TEST",
                "threshold_sha256": reader.members[PREFIX + stage + "/thresholds.json"][
                    "sha256"
                ],
            },
        )
        updates, cheap = [], []
        for index in range(1, 61):
            updates.append(
                {
                    "branch_update": index,
                    "status": "ACCEPTED",
                    "phase": "ACCEPTED",
                    "policy": "disabled",
                    "preserve_rng": True,
                    "spec": None,
                    "injections": [],
                    "update": {
                        "update": 10 + index,
                        "accepted_tokens": initial["cursor"] + index * targets,
                        "tokens_this_update": targets,
                    },
                }
            )
            observed_spec = copy.deepcopy(spec)
            if index <= 30:
                observed_spec.update(
                    norm_upper_bounds=[],
                    layer_norm_upper_bounds=[],
                    loss_upper_bound=None,
                )
            spec_hash = hashlib.sha256(
                json.dumps(
                    observed_spec, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            cheap.append(
                {
                    "update": index,
                    "phase": "calibration" if index <= 30 else "disjoint-test",
                    "expected_gradient_count": 1,
                    "present_gradient_count": 1,
                    "missing_gradient_names": [],
                    "empty_gradient_names": [],
                    "nonfinite_gradient_names": [],
                    "global_norm_finite": True,
                    "nonfinite_loss_alert": False,
                    "loss": 2.0,
                    "seconds": 0.01,
                    "spec_sha256": spec_hash,
                    "spec": observed_spec,
                    "gradient_norms": {"w": 2.0},
                    "layer_norms": {"layer": 2.0},
                    "present_gradient_norm": 2.0,
                    "global_gradient_norm": 2.0,
                    "global_norm_covers_all_expected_gradients": True,
                    "norm_upper_bounds": dict(observed_spec["norm_upper_bounds"]),
                    "norm_bounds_cover_all_expected_gradients": index > 30,
                    "norm_limit_exceeded_names": [],
                    "layer_groups": {"layer": ["w"]},
                    "layer_norm_upper_bounds": dict(
                        observed_spec["layer_norm_upper_bounds"]
                    ),
                    "layer_norm_bounds_complete": index > 30,
                    "layer_norm_limit_exceeded": [],
                    "loss_upper_bound": observed_spec["loss_upper_bound"],
                    "clipping_limit": 1.0,
                    "missing_gradient_alert": False,
                    "empty_gradient_alert": False,
                    "nonfinite_gradient_alert": False,
                    "norm_alert": False if index > 30 else None,
                    "layer_norm_alert": False if index > 30 else None,
                    "loss_spike_alert": False if index > 30 else None,
                    "clipping_would_activate": True,
                }
            )
        reader.journals[PREFIX + stage + "/updates.jsonl"] = updates
        reader.journals[PREFIX + stage + "/cheap-clean.jsonl"] = cheap
    reader.store(
        "diagnostic-ready.json",
        {
            "status": "FROZEN_CLEAN_CALIBRATION",
            "batch_targets": 8192,
            "new_clean_updates": 60,
            "by_state": {"early-warm10": stages["early-warm10"]},
            "test_receipts": {
                "early-warm10": reader.members[
                    PREFIX + "early-warm10/clean-completion.json"
                ]["sha256"]
            },
        },
    )
    equal = dict(
        valid=True,
        all_finite=True,
        byte_equal=True,
        value_equal=True,
        tensors=[{"name": "w"}],
    )
    reader.store(
        "operand-capture-parity.json",
        {
            "plain": {"status": "ACCEPTED"},
            "observed": {"status": "ACCEPTED"},
            "captured_keys": ["dy", "weight", "x"],
            "full_state": equal,
            "gradients": copy.deepcopy(equal),
        },
    )
    roles = ("forward", "input_gradient", "weight_gradient")
    reader.store(
        "production-early-warm10/abft-tolerances.json",
        {role: {"atol": 0.25, "rtol": 0.0} for role in roles},
    )
    clean, faults = [], []
    for case in registration:
        role, update, kind = case["role"], case["production_clean_update"], case["kind"]
        root = PREFIX + "production-early-warm10/"
        for name in (f"operands-{update:02}.pt", f"product-{role}-{update:02}.pt"):
            reader.members[root + name] = {"sha256": "b" * 64, "bytes": 1}
        check = dict(
            finite=True,
            detected=False,
            row_max_residual=0.1,
            column_max_residual=0.2,
            atol=0.25,
            rtol=0.0,
        )
        if kind == "clean-test":
            m, k, n = case["shape_mkn"]
            clean.append(
                {
                    "role": role,
                    "update": update,
                    "shape": {"left": [m, k], "right": [k, n]},
                    "seconds": 0.01,
                    **check,
                }
            )
        elif kind != "calibration":
            fault = (
                {"element": case["element"], "bit": case["bit"]}
                if kind == "post-product-bit"
                else {
                    "element": 0,
                    "bit": 31,
                    "phase": "before_both_executions",
                    "lifetime": "persistent",
                }
                if kind == "common-mode-operand"
                else None
            )
            faults.append(
                {
                    "case": copy.deepcopy(case),
                    "dtype": "torch.float32",
                    "tf32": False,
                    "abft": check,
                    "abft_seconds": 0.01,
                    "paired_digest_seconds": 0.01,
                    "nonfinite_seconds": 0.01,
                    "paired_digest_detected": False,
                    "nonfinite_detected": False,
                    "fault": fault,
                }
            )
    reader.journals[PREFIX + "production-early-warm10/abft-clean-tests.jsonl"] = clean
    reader.journals[PREFIX + "production-early-warm10/abft-faults.jsonl"] = faults
    reader.store(
        "completion.json",
        {
            "status": "MEASUREMENTS_COMPLETE_PENDING_CUSTODY_AND_REVIEW",
            "clean_updates_total": 120,
            "abft_products_total": 276,
        },
    )
    reader.store(
        "production-early-warm10/abft-completion.json",
        {
            "status": "MEASUREMENTS_COMPLETE_PENDING_CUSTODY_AND_REVIEW",
            "events_total": 276,
        },
    )
    return reader, registration, initial


class BaselineFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_allocation()

    def analyze(self, reader, registration, initial):
        return analyze_baselines(reader, registration, initial, "a" * 64)

    def test_complete_census_keeps_calibration_objects_distinct_from_decisions(self):
        result = self.analyze(*fixture())
        self.assertEqual(result["abft_registered_events"], 276)
        self.assertEqual(result["abft_journaled_clean_test_and_fault_decisions"], 186)
        self.assertEqual(
            result["abft_calibration_objects_without_native_residual_journal"], 90
        )
        self.assertFalse(result["abft_calibration_rule_numerically_reconstructed"])
        self.assertTrue(result["operand_capture_control_valid"])
        decision = result["clean_stages"]["early-warm10"][
            "disjoint_clean_test_decisions"
        ]["clipping_would_activate"]
        self.assertEqual(decision["true"], 30)
        self.assertEqual(decision["recorded_true"], 30)
        self.assertEqual(decision["false"], 0)
        self.assertEqual(decision["contradictions"], 0)
        self.assertEqual(decision["unsupported_or_invalid"], 0)
        self.assertEqual(decision["denominator"], 30)

    def test_cheap_flags_are_validated_without_erasing_raw_contradictions(self):
        for key in (
            "norm_alert",
            "layer_norm_alert",
            "loss_spike_alert",
            "missing_gradient_alert",
            "empty_gradient_alert",
            "nonfinite_gradient_alert",
            "nonfinite_loss_alert",
            "clipping_would_activate",
        ):
            reader, registration, initial = fixture()
            raw = reader.journals[PREFIX + "early-warm10/cheap-clean.jsonl"][30]
            raw[key] = not raw[key]
            result = self.analyze(reader, registration, initial)
            stage = result["clean_stages"]["early-warm10"]
            check = stage["cheap_decision_validation"][30]["decisions"][key]
            self.assertEqual(check["status"], "CONTRADICTORY_RECORDED_DECISION")
            self.assertIs(stage["cheap_observations"][30][key], raw[key])
            self.assertEqual(
                stage["disjoint_clean_test_decisions"][key]["contradictions"], 1
            )

    def test_missing_clipping_scalar_is_unsupported_not_a_false_or_true_count(self):
        reader, registration, initial = fixture()
        del reader.journals[PREFIX + "early-warm10/cheap-clean.jsonl"][30][
            "global_gradient_norm"
        ]
        stage = self.analyze(reader, registration, initial)["clean_stages"][
            "early-warm10"
        ]
        self.assertEqual(
            stage["cheap_decision_validation"][30]["decisions"][
                "clipping_would_activate"
            ]["status"],
            "UNSUPPORTED_MISSING_SCALAR_EVIDENCE",
        )
        self.assertEqual(
            stage["disjoint_clean_test_decisions"]["clipping_would_activate"]["true"],
            29,
        )

    def test_actual_nonfinite_clean_loss_still_fails_population_gate(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for index in (0, 30):
                reader, registration, initial = fixture()
                raw = reader.journals[PREFIX + "early-warm10/cheap-clean.jsonl"][index]
                raw["loss"] = value
                raw["nonfinite_loss_alert"] = True
                with self.assertRaisesRegex(ValueError, "observation/counter"):
                    self.analyze(reader, registration, initial)

    def test_wrong_batch_or_endpoint_counters_fail(self):
        reader, registration, initial = fixture()
        reader.journals[PREFIX + "early-warm10/updates.jsonl"][0]["update"][
            "tokens_this_update"
        ] = 65536
        with self.assertRaisesRegex(ValueError, "counter"):
            self.analyze(reader, registration, initial)

    def test_fitted_threshold_and_heldout_spec_changes_fail(self):
        reader, registration, initial = fixture()
        reader.objects[PREFIX + "early-warm10/thresholds.json"]["spec"][
            "loss_upper_bound"
        ] = 9.0
        with self.assertRaises(ValueError):
            self.analyze(reader, registration, initial)

    def test_duplicate_missing_or_false_tolerance_decision_fail(self):
        for change in ("duplicate", "missing", "decision"):
            reader, registration, initial = fixture()
            rows = reader.journals[
                PREFIX + "production-early-warm10/abft-clean-tests.jsonl"
            ]
            if change == "duplicate":
                rows.append(copy.deepcopy(rows[0]))
            elif change == "missing":
                rows.pop()
            else:
                rows[0]["detected"] = True
            with self.assertRaises(ValueError):
                self.analyze(reader, registration, initial)

    def test_empty_capture_parity_is_not_supported_control(self):
        reader, registration, initial = fixture()
        reader.objects[PREFIX + "operand-capture-parity.json"]["full_state"][
            "tensors"
        ] = []
        result = self.analyze(reader, registration, initial)
        self.assertFalse(result["operand_capture_control_valid"])

    def test_missing_calibration_product_is_not_invented_observation(self):
        reader, registration, initial = fixture()
        del reader.members[PREFIX + "production-early-warm10/product-forward-01.pt"]
        with self.assertRaisesRegex(ValueError, "census"):
            self.analyze(reader, registration, initial)


if __name__ == "__main__":
    require_allocation()
    unittest.main()
