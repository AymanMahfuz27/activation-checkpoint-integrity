"""Meaningful CUDA qualification of clean parity, fault effects and recovery.

The parent admits this source, input corpus and complete runtime before loading
the module. These small fixtures do not estimate a learning benefit.
"""

import json
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
