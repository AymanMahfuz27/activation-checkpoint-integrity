"""Minimal saved-slot numerical qualification; run only after allocated admission.

Each fixture uses real autograd and an optimizer. No test has been executed by
this source creation; the admitted parent chooses BB_TEST_DEVICE explicitly.
"""

from contextlib import nullcontext
import gc
import os
import unittest

import selector as core
from saved_slots import SavedSlotCheck, POLICIES, qualify_checkpoint_source

import torch
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


class _Model(torch.nn.Module):
    def __init__(self, *, family="gelu", early_stop=True, recompute_fault=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 64).reshape(8, 8))
        self.bias = torch.nn.Parameter(torch.linspace(-0.05, 0.05, 8))
        self.family, self.early_stop, self.recompute_fault = family, early_stop, recompute_fault
        self.calls = {"original": 0, "recompute": 0}
        self.gap_injector = None

    def body(self, value):
        role = core.REGION.get()[1]
        self.calls[role] += 1
        hidden = F.linear(value, self.weight, self.bias)
        hidden = hidden * (0.5 + torch.rand_like(hidden))
        if self.recompute_fault and role == "recompute":
            # A simple deterministic software-level replay fault changes the
            # saved activation before derivative saving/observer recording.
            hidden = hidden + 0.75
        if self.family == "gelu":
            result = F.gelu(hidden)
        elif self.family == "linear":
            result = hidden @ self.weight
        elif self.family == "softmax":
            result = F.softmax(hidden, dim=-1)
        elif self.family == "views":
            result = hidden[:, 1:7] @ self.weight[1:7, :]
        elif self.family == "two_gelu":
            first = F.gelu(hidden)
            if role == "original" and self.gap_injector is not None:
                first.register_hook(self.gap_injector.fire)
            result = F.gelu(first * 1.1)
        else:
            raise ValueError(self.family)
        return result

    def forward(self, value):
        def contexts():
            return core.region_context("block", "original"), core.region_context("block", "recompute")
        with cp.set_checkpoint_early_stop(self.early_stop):
            return cp.checkpoint(self.body, value, use_reentrant=False, context_fn=contexts,
                                 determinism_check="default", preserve_rng_state=True)


class _LateConsumerFault(TorchDispatchMode):
    """Mutate after saved unpack and checker return, before native GELU backward."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) == "aten.gelu_backward.default":
            with torch.no_grad(), _disable_current_modes():
                args[1].add_(0.75)
            self.calls += 1
        return func(*args, **(kwargs or {}))


class _SavedGapInjector(TorchDispatchMode):
    """Equal-instrumentation control for mutation after stock frame checking.

    The second GELU's backward triggers native recomputation and initial all-live
    metadata validation. The original first GELU output's gradient hook then
    mutates the retained replay input before that first GELU's saved slot is
    unpacked. Native consumer bytes are independently checked at dispatch.
    """

    def __init__(self, enabled):
        super().__init__()
        self.enabled = enabled
        self.target = None
        self.expected = None
        self.replay_gelu_calls = 0
        self.backward_gelu_calls = 0
        self.hook_calls = 0
        self.native_consumed_mutation = False

    def fire(self, gradient):
        self.hook_calls += 1
        if self.target is None or self.hook_calls != 1:
            raise AssertionError("Gap fixture target/hook census failed")
        with torch.no_grad(), _disable_current_modes():
            if self.enabled:
                self.target.add_(0.75)
            self.expected = self.target.detach().clone()
        return gradient

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        context = core.REGION.get()
        if name == "aten.gelu.default" and context is not None and context[1] == "recompute":
            if self.replay_gelu_calls == 0:
                self.target = args[0]
            self.replay_gelu_calls += 1
        if name == "aten.gelu_backward.default":
            if self.backward_gelu_calls == 1:
                if self.expected is None or self.hook_calls != 1:
                    raise AssertionError("Gap fixture mutation did not precede target native consumption")
                with torch.no_grad(), _disable_current_modes():
                    self.native_consumed_mutation = bool(torch.equal(args[1], self.expected))
            self.backward_gelu_calls += 1
        return func(*args, **(kwargs or {}))


def _run(device, *, policy=None, family="gelu", early_stop=True, replay_fault=False, late_fault=False, gap_fault=False):
    torch.manual_seed(43)
    model = _Model(family=family, early_stop=early_stop, recompute_fault=replay_fault).to(device)
    source = torch.linspace(-0.7, 0.8, 32, device=device).reshape(4, 8).requires_grad_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    checker = SavedSlotCheck(policy) if policy is not None else None
    injector = (_SavedGapInjector(gap_fault) if family == "two_gelu"
                else _LateConsumerFault() if late_fault else None)
    if family == "two_gelu":
        model.gap_injector = injector
    status, result = "ACCEPTED", None
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    try:
        with (checker.track_model(model) if checker else nullcontext()), \
             (checker if checker else nullcontext()), \
             (injector if injector else nullcontext()):
            loss = model(source).square().mean()
            loss.backward()
            if checker:
                result = checker.finish()
                if result["failed"]:
                    status = "REJECTED"
            if status == "ACCEPTED":
                optimizer.step()
    except cp.CheckpointError:
        if policy != "stock_metadata_sync":
            raise
        result = checker.summary()
        if checker.unsupported or not checker.native_value_mismatch or result["status"] != "VALUE_MISMATCH":
            raise
        status = "REJECTED_NATIVE_VALUE_METADATA"
    gradients = {name: None if value.grad is None else value.grad.detach().clone()
                 for name, value in model.named_parameters()}
    final = {name: value.detach().clone() for name, value in model.named_parameters()}
    return {"status": status, "checker": checker, "result": result, "initial": initial,
            "gradients": gradients, "final": final, "calls": model.calls,
            "rng": torch.get_rng_state().clone(),
            "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()] if device.type == "cuda" else [],
            "late_calls": injector.calls if isinstance(injector, _LateConsumerFault) else 0,
            "gap_hook_calls": injector.hook_calls if isinstance(injector, _SavedGapInjector) else 0,
            "native_consumed_gap_mutation": injector.native_consumed_mutation if isinstance(injector, _SavedGapInjector) else None}


def _equal(first, second):
    if first.keys() != second.keys():
        return False
    for key in first:
        a, b = first[key], second[key]
        if a is None or b is None:
            if a is not None or b is not None:
                return False
        elif not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            return False
    return True


class SavedSlotFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        name = os.environ.get("BB_TEST_DEVICE")
        if name not in ("cpu", "cuda:0"):
            raise RuntimeError("Allocated parent must set BB_TEST_DEVICE")
        cls.device = torch.device(name)
        qualify_checkpoint_source()

    def test_clean_gradients_updates_rng_and_replay_are_unchanged(self):
        for early_stop in (True, False):
            baseline = _run(self.device, early_stop=early_stop)
            for policy in POLICIES:
                with self.subTest(policy=policy, early_stop=early_stop):
                    checked = _run(self.device, policy=policy, early_stop=early_stop)
                    self.assertEqual(checked["status"], "ACCEPTED")
                    self.assertEqual(checked["result"]["status"], "PASS")
                    self.assertEqual(checked["calls"], baseline["calls"])
                    self.assertEqual(checked["calls"], {"original": 1, "recompute": 1})
                    self.assertTrue(_equal(checked["gradients"], baseline["gradients"]))
                    self.assertTrue(_equal(checked["final"], baseline["final"]))
                    self.assertTrue(torch.equal(checked["rng"], baseline["rng"]))
                    self.assertEqual(len(checked["cuda_rng"]), len(baseline["cuda_rng"]))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(checked["cuda_rng"], baseline["cuda_rng"])))

    def test_replay_fault_is_seen_by_candidate_and_stock_metadata_baselines(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                checked = _run(self.device, policy=policy, replay_fault=True)
                self.assertIn(checked["status"], ("REJECTED", "REJECTED_NATIVE_VALUE_METADATA"))
                self.assertTrue(_equal(checked["initial"], checked["final"]))

    def test_linear_saved_activations_are_covered_without_nonlinear_registry(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                clean = _run(self.device, policy=policy, family="linear")
                self.assertEqual(clean["result"]["status"], "PASS")
                fault = _run(self.device, policy=policy, family="linear", replay_fault=True)
                self.assertIn(fault["status"], ("REJECTED", "REJECTED_NATIVE_VALUE_METADATA"))

    def test_unpack_to_kernel_fault_is_an_explicit_remaining_window(self):
        baseline = _run(self.device)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                fault = _run(self.device, policy=policy, late_fault=True)
                self.assertEqual(fault["status"], "ACCEPTED")
                self.assertEqual(fault["result"]["status"], "PASS")
                self.assertEqual(fault["late_calls"], 1)
                self.assertFalse(_equal(fault["gradients"], baseline["gradients"]))

    def test_later_unpack_fault_distinguishes_native_all_live_from_per_unpack(self):
        clean = _run(self.device, family="two_gelu")
        faulty = _run(self.device, family="two_gelu", gap_fault=True)
        self.assertEqual(faulty["status"], "ACCEPTED")
        self.assertEqual(faulty["gap_hook_calls"], 1)
        self.assertTrue(faulty["native_consumed_gap_mutation"])
        self.assertFalse(_equal(clean["gradients"], faulty["gradients"]))
        for policy in POLICIES:
            with self.subTest(policy=policy):
                checked = _run(self.device, policy=policy, family="two_gelu", gap_fault=True)
                self.assertEqual(checked["calls"], faulty["calls"])
                self.assertEqual(checked["gap_hook_calls"], 1)
                self.assertTrue(checked["native_consumed_gap_mutation"])
                if policy in ("per_unpack", "full_copy"):
                    self.assertEqual(checked["status"], "REJECTED")
                    self.assertTrue(_equal(checked["initial"], checked["final"]))
                else:
                    self.assertEqual(checked["status"], "ACCEPTED")
                    self.assertEqual(checked["result"]["status"], "PASS")
                    self.assertTrue(_equal(checked["gradients"], faulty["gradients"]))

    def test_noncontiguous_offset_views_preserve_native_behavior(self):
        baseline = _run(self.device, family="views")
        for policy in POLICIES:
            with self.subTest(policy=policy):
                checked = _run(self.device, family="views", policy=policy)
                self.assertEqual(checked["result"]["status"], "PASS")
                self.assertEqual(checked["calls"], baseline["calls"])
                self.assertTrue(_equal(checked["gradients"], baseline["gradients"]))
                self.assertTrue(_equal(checked["final"], baseline["final"]))

    def test_native_hook_and_metadata_callback_are_restored_after_failure(self):
        hook = cp._checkpoint_hook
        metadata = cp._allowed_determinism_checks_to_fns["default"]
        for policy in POLICIES:
            _run(self.device, policy=policy, replay_fault=True)
            self.assertIs(cp._checkpoint_hook, hook)
            self.assertIs(cp._allowed_determinism_checks_to_fns["default"], metadata)
        gc.collect()

    def test_no_backward_cannot_be_reported_as_verified(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                model = _Model().to(self.device)
                checker = SavedSlotCheck(policy)
                with checker:
                    model(torch.ones(4, 8, device=self.device, requires_grad=True))
                    with self.assertRaises(core.UnsupportedPlan):
                        checker.finish()

    def test_wrong_source_and_nested_adaptation_fail_closed(self):
        with self.assertRaises(core.UnsupportedPlan):
            SavedSlotCheck(expected_checkpoint_sha256="0" * 64)
        with SavedSlotCheck("per_unpack"):
            with self.assertRaises(core.UnsupportedPlan):
                with SavedSlotCheck("per_unpack"):
                    pass


if __name__ == "__main__":
    unittest.main()
