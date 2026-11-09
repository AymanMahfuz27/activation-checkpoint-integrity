"""Meaningful CUDA qualification of clean parity, fault effects and recovery.

The parent admits this source, input corpus and complete runtime before loading
the module. These small fixtures do not estimate a learning benefit.
"""

import json
import copy
from pathlib import Path
import unittest

from . import runner as training


class TrainingFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path(__file__).with_name("pilot-config.json").read_text())
        cls.config["max_updates"] = 3
        cls.config["model"]["width"] = 32
        cls.config["model"]["sequence_length"] = 16
        cls.config["batch_size"] = 2
        cls.config["eval_tokens"] = 64
        cls.train, cls.heldout, cls.data_record = training.load_data(cls.config)

    def engine(self):
        return training.make_engine(self.config, 11)

    def step(self, model, optimizer, controller, policy, fault=None, update=0):
        x, y = training.batch_for(self.train, self.config, update, seed=11)
        return training.backward_attempt(model, optimizer, controller, x, y, self.config,
                                         policy, fault, capture_gradients=True)

    def fault(self, *, sham=False):
        return dict(self.config["fault"], seed=991, sham=sham, exact_audit=True)

    def test_clean_checked_matches_unchecked_optimizer_and_rng(self):
        model, optimizer, controller = self.engine()
        for update in range(2):
            row, _ = self.step(model, optimizer, controller, "disabled", update=update)
            self.assertTrue(row["optimizer_entered"])
        reference = training.snapshot(model, optimizer, 2)
        for policy in ("saved_slot", "stock_metadata", "simple_input", "full_forward_io"):
            with self.subTest(policy=policy):
                model, optimizer, controller = self.engine()
                for update in range(2):
                    row, _ = self.step(model, optimizer, controller, policy, update=update)
                    self.assertFalse(row["rejected"])
                    self.assertEqual(row["fault_events"], [])
                self.assertTrue(training.equal_tree(training.snapshot(model, optimizer, 2), reference))

    def test_fault_changes_actual_backward_gradient_and_update(self):
        model, optimizer, controller = self.engine()
        _, clean_gradients = self.step(model, optimizer, controller, "disabled")
        clean = training.snapshot(model, optimizer, 1)
        model, optimizer, controller = self.engine()
        row, faulty_gradients = self.step(model, optimizer, controller, "disabled", self.fault())
        self.assertTrue(row["optimizer_entered"])
        self.assertTrue(row["finite_gradients"])
        self.assertEqual(len(row["fault_events"]), 1)
        self.assertTrue(row["fault_events"][0]["actual_backward_equals_injected"])
        self.assertTrue(row["fault_events"][0]["actual_backward_exact_bytes_equal"])
        self.assertFalse(training.equal_tree(clean_gradients, faulty_gradients))
        self.assertFalse(training.equal_tree(clean["model"], training.snapshot(model, optimizer, 1)["model"]))

    def test_rejected_attempt_restores_and_retry_matches_clean(self):
        model, optimizer, controller = self.engine()
        self.step(model, optimizer, controller, "disabled")
        reference = training.snapshot(model, optimizer, 1)
        model, optimizer, controller = self.engine()
        initial = training.snapshot(model, optimizer, 0)
        row, _ = self.step(model, optimizer, controller, "saved_slot", self.fault())
        self.assertTrue(row["rejected"])
        self.assertFalse(row["clipping_entered"])
        self.assertFalse(row["optimizer_entered"])
        training.restore(model, optimizer, initial)
        self.assertTrue(training.equal_tree(training.snapshot(model, optimizer, 0), initial))
        retry, _ = self.step(model, optimizer, controller, "saved_slot")
        self.assertFalse(retry["rejected"])
        self.assertTrue(retry["optimizer_entered"])
        self.assertTrue(training.equal_tree(training.snapshot(model, optimizer, 1), reference))

    def test_sham_preserves_clean_state_and_is_not_detection(self):
        model, optimizer, controller = self.engine()
        self.step(model, optimizer, controller, "disabled")
        reference = training.snapshot(model, optimizer, 1)
        model, optimizer, controller = self.engine()
        row, _ = self.step(model, optimizer, controller, "saved_slot", self.fault(sham=True))
        self.assertFalse(row["rejected"])
        self.assertTrue(row["optimizer_entered"])
        self.assertEqual(row["fault_events"][0]["before_signature"], row["fault_events"][0]["after_signature"])
        self.assertTrue(training.equal_tree(training.snapshot(model, optimizer, 1), reference))

    def test_simple_input_and_full_io_see_early_replay_fault(self):
        for policy in ("simple_input", "full_forward_io"):
            with self.subTest(policy=policy):
                model, optimizer, controller = self.engine()
                row, _ = self.step(model, optimizer, controller, policy, self.fault())
                self.assertTrue(row["rejected"])
                self.assertFalse(row["optimizer_entered"])

    def test_calibration_metric_is_weighted_finite_and_nonmutating(self):
        model, optimizer, controller = self.engine()
        initial = training.snapshot(model, optimizer, 0)
        first = training.evaluate(model, self.heldout, self.config)
        second = training.evaluate(model, self.heldout, self.config)
        self.assertEqual(first, second)
        self.assertTrue(first["finite"])
        self.assertEqual(first["evaluated_bytes"], 64)
        self.assertFalse(self.data_record["test_evaluated"])
        self.assertTrue(training.equal_tree(training.snapshot(model, optimizer, 0), initial))


class EndpointFixtures(unittest.TestCase):
    """Pure control-clock fixtures; loaded only after actual job admission."""

    def config(self):
        return {"target_perplexity": 10.0, "administrative_wall_horizon_seconds": 100.0,
                "endpoint_clock": "checker_cold_elapsed_seconds",
                "target_confirmation_evaluations": 2}

    def metric(self, elapsed, perplexity):
        return {"byte_perplexity": perplexity, "elapsed_seconds": elapsed - 5.0,
                "checker_cold_elapsed_seconds": elapsed,
                "instrumented_cold_elapsed_seconds": elapsed,
                "endpoint_clock_values": {"elapsed_seconds": elapsed - 5.0,
                    "checker_cold_elapsed_seconds": elapsed, "instrumented_cold_elapsed_seconds": elapsed}}

    def test_two_scheduled_evaluations_reset_and_charge_selected_clock(self):
        endpoint = training.TargetWallEndpoint(self.config())
        endpoint.observe(self.metric(10.0, 12.0))
        endpoint.observe(self.metric(20.0, 9.0))
        self.assertIsNone(endpoint.first_confirmed)
        endpoint.observe(self.metric(30.0, 9.0), scheduled=False)
        self.assertIsNone(endpoint.first_confirmed)
        endpoint.observe(self.metric(40.0, 11.0))
        endpoint.observe(self.metric(60.0, 9.0))
        self.assertIsNone(endpoint.first_confirmed)
        endpoint.observe(self.metric(80.0, 9.0))
        self.assertEqual(endpoint.stop_reason(81.0), "TARGET_CONFIRMED")
        record = endpoint.result(81.0, stop_reason="TARGET_CONFIRMED", failure=None)
        self.assertTrue(record["valid"])
        self.assertFalse(record["common_horizon_right_censored"])
        self.assertEqual(record["restricted_seconds"], 80.0)
        self.assertEqual(record["confirmed_target_elapsed_seconds"], 80.0)

    def test_late_confirmation_and_early_failure_never_invent_common_followup(self):
        endpoint = training.TargetWallEndpoint(self.config())
        endpoint.observe(self.metric(90.0, 9.0))
        endpoint.observe(self.metric(102.0, 9.0))
        self.assertIsNone(endpoint.first_confirmed)
        self.assertIsNotNone(endpoint.first_confirmed_after_horizon)
        self.assertEqual(endpoint.stop_reason(103.0), "COMMON_WALL_HORIZON")
        record = endpoint.result(110.0, stop_reason="COMMON_WALL_HORIZON", failure=None)
        self.assertTrue(record["valid"])
        self.assertTrue(record["common_horizon_right_censored"])
        self.assertEqual(record["administrative_censor_elapsed_seconds"], 100.0)
        self.assertEqual(record["restricted_seconds"], 100.0)
        self.assertEqual(record["observed_stop_elapsed_seconds"], 110.0)
        self.assertEqual(record["observed_horizon_overshoot_seconds"], 10.0)
        for reason in ("ACCEPTED_UPDATE_SAFETY_LIMIT_BEFORE_COMMON_HORIZON", "RuntimeError: injected failure"):
            early = training.TargetWallEndpoint(self.config()).result(30.0, stop_reason=None, failure=reason)
            self.assertFalse(early["valid"])
            self.assertFalse(early["common_horizon_right_censored"])
            self.assertIsNone(early["restricted_seconds"])
            self.assertIsNone(early["administrative_censor_elapsed_seconds"])

    def test_arm_fault_overrides_preserve_schedule_and_reject_unregistered_fields(self):
        config = json.loads(Path(__file__).with_name("pilot-config.json").read_text())
        arm = {"name": "scoped-control", "policy": "disabled", "fault_probability": 0.05,
               "fault": {"kind": "bit_flip", "severity": 22, "window": "consumer_entry", "layer": 1}}
        before = copy.deepcopy(config)
        schedule = training.fault_schedule(config, 101)
        spec = training.declared_fault(config, arm)
        self.assertEqual(spec["severity"], 22)
        self.assertEqual(spec["window"], "consumer_entry")
        self.assertEqual(spec["seed_offset"], config["fault"]["seed_offset"])
        self.assertEqual(training.fault_schedule(config, 101), schedule)
        self.assertEqual(config, before)
        for bad in ({"seed_offset": 1}, {"seed": 1}, {"severity": float("nan")}, {"layer": 99}):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                training.declared_fault(config, dict(arm, fault=bad))

    def test_qualification_uses_calibration_without_test_tensors_and_rejects_unknown_phase(self):
        config = json.loads(Path(__file__).with_name("pilot-config.json").read_text())
        config.update(phase="qualification", stopping_mode="target_or_wall",
                      target_perplexity=10.0, administrative_wall_horizon_seconds=1.0,
                      endpoint_clock="checker_cold_elapsed_seconds", target_preregistered=False)
        training.validate_config(config)
        train, calibration, record = training.load_data(config)
        self.assertEqual(record["evaluation_split"], "calibration")
        self.assertIs(record["test_evaluated"], False)
        self.assertEqual(record["constructed_tensor_splits"], ["train", "calibration"])
        self.assertEqual(train.numel(), record["split_boundaries"][1])
        self.assertEqual(calibration.numel(), record["split_boundaries"][2] - record["split_boundaries"][1])
        self.assertEqual(config["data"]["evaluation_split"], "calibration")
        unknown = copy.deepcopy(config)
        unknown["phase"] = "unknown-phase"
        with self.assertRaises(ValueError):
            training.validate_config(unknown)
        with self.assertRaises(RuntimeError):
            training.load_data(unknown)
        invalid_test = copy.deepcopy(config)
        invalid_test["data"]["evaluation_split"] = "test"
        with self.assertRaises(RuntimeError):
            training.load_data(invalid_test)


def run_smoke(config=None, output=None):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrainingFixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skips": len(result.skipped), "successful": result.wasSuccessful()}
    if output is not None:
        training.write(Path(output) / "training-smoke.json", record)
    if not result.wasSuccessful():
        raise RuntimeError("Training smoke qualification failed")
    return record
