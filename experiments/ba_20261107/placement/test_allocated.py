"""Actual CUDA mode-stacking, delivery, autograd and canonical-byte controls."""

import hashlib
import os
import struct
import sys
import unittest

import torch

from ac_integrity.capture.dispatch import CaptureMode
from ac_integrity.research import completion
from ac_integrity.research.m4m5 import FaultSpec, Intervention
from placement.injector import EarlyResultFault, IDENTITY, Site, TARGET
from placement.run_placement import leaf_manifest


def tiny_site(**changes):
    values = dict(
        input_shape=(2, 2),
        weight_shape=(2, 2),
        bias_shape=(2,),
        output_shape=(2, 2),
        numel=4,
        element=1,
        old_bits=1065353216,
        new_bits=1065353217,
    )
    values.update(changes)
    return Site(**values)


def fault(enabled=True):
    item = Intervention(
        FaultSpec(
            "recompute-L6-fraction-low-bit0",
            "recompute",
            6,
            "bit",
            bit=0,
            seed=1729,
            enabled=enabled,
        )
    )
    item.before_forward(None, None, 1)
    return item


def operands(*, gradients=False):
    return (
        torch.zeros(2, dtype=torch.float32, device="cuda"),
        torch.ones((2, 2), dtype=torch.float32, device="cuda", requires_grad=gradients),
        torch.eye(2, dtype=torch.float32, device="cuda", requires_grad=gradients),
    )


class RecordedOutputs:
    lightweight = True

    def __init__(self, order):
        self.order, self.events, self.unsupported = order, [], []

    def should_observe_current_operator(self):
        return True

    def record(self, func, inputs, leaves, spec, args, kwargs):
        self.order.append("upper_record")
        values = [value for _, value in leaves if isinstance(value, torch.Tensor)]
        self.events.append(
            {
                "operator": str(func),
                "bits": values[0]
                .detach()
                .clone(memory_format=torch.contiguous_format)
                .reshape(-1)
                .view(torch.int32)
                .cpu()
                .tolist(),
            }
        )


class PlacementFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            sys.platform != "linux"
            or not os.environ.get("SLURM_JOB_ID", "").isdecimal()
            or not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "Real allocated CUDA placement fixtures required; no skips"
            )
        completion.source_allocation_guard()

    def test_lower_mutation_precedes_real_upper_record_without_helper_events(self):
        instance = fault()
        mode = EarlyResultFault({"instance": instance}, tiny_site())
        upper = RecordedOutputs(mode.order)
        args = operands()
        with instance.context(TARGET, "recompute"), mode, CaptureMode(upper):
            result = torch.addmm(*args)
        self.assertEqual(
            mode.order, ["native_func_return", "lower_mutation", "upper_record"]
        )
        self.assertEqual(len(upper.events), 1)
        self.assertEqual(upper.events[0]["operator"], "aten.addmm.default")
        self.assertEqual(
            upper.events[0]["bits"], [1065353216, 1065353217, 1065353216, 1065353216]
        )
        self.assertEqual(
            int(result.reshape(-1).view(torch.int32)[1].item()), 1065353217
        )
        self.assertEqual(len(instance.events), 1)
        self.assertEqual(instance.events[0]["tensor"], IDENTITY)

    def test_disabled_and_original_phase_leave_same_values_and_pair_ordinals(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                instance = fault(enabled)
                mode = EarlyResultFault({"instance": instance}, tiny_site())
                upper = RecordedOutputs(mode.order)
                args = operands()
                with mode, CaptureMode(upper):
                    with instance.context(TARGET, "original"):
                        torch.addmm(*args)
                    if not enabled:
                        with instance.context(TARGET, "recompute"):
                            torch.addmm(*args)
                self.assertEqual(len(instance.events), 0)
                self.assertEqual(len(upper.events), 2 if not enabled else 1)
                self.assertTrue(
                    all(row["bits"] == [1065353216] * 4 for row in upper.events)
                )
                if not enabled:
                    mode.validate(False, expected_per_role=1)

    def test_only_first_recompute_is_changed_and_duplicate_candidates_are_not_hidden(
        self,
    ):
        instance = fault()
        mode = EarlyResultFault({"instance": instance}, tiny_site())
        upper = RecordedOutputs(mode.order)
        args = operands()
        with instance.context(TARGET, "recompute"), mode, CaptureMode(upper):
            torch.addmm(*args)
            torch.addmm(*args)
        self.assertEqual(len(instance.events), 1)
        self.assertEqual(mode.seen["recompute"], 2)
        self.assertEqual(upper.events[0]["bits"][1], 1065353217)
        self.assertEqual(upper.events[1]["bits"][1], 1065353216)
        with self.assertRaisesRegex(RuntimeError, "Missing or ambiguous"):
            mode.validate(True, expected_per_role=1)

    def test_wrong_old_word_or_output_shape_fails_without_site_adaptation(self):
        for site, message in (
            (tiny_site(old_bits=0), "old word"),
            (tiny_site(output_shape=(4, 1)), "shape/dtype/layout"),
        ):
            with self.subTest(message=message):
                instance = fault()
                mode = EarlyResultFault({"instance": instance}, site)
                args = operands()
                with self.assertRaisesRegex(RuntimeError, message):
                    with instance.context(TARGET, "recompute"), mode:
                        torch.addmm(*args)
                self.assertEqual(instance.events, [])

    def test_missing_or_different_operand_schema_cannot_claim_supported_delivery(self):
        instance = fault()
        mode = EarlyResultFault({"instance": instance}, tiny_site(input_shape=(4, 1)))
        args = operands()
        with instance.context(TARGET, "recompute"), mode:
            torch.addmm(*args)
        self.assertEqual(mode.candidates, [])
        self.assertEqual(instance.events, [])
        with self.assertRaisesRegex(RuntimeError, "Missing or ambiguous"):
            mode.validate(True, expected_per_role=1)

    def test_cloned_native_result_preserves_matched_late_autograd_gradients(self):
        gradients, values = [], []
        for early in (False, True):
            instance = fault()
            mode = EarlyResultFault({"instance": instance}, tiny_site())
            bias, inputs, weight = operands(gradients=True)
            with instance.context(TARGET, "recompute"):
                if early:
                    with mode:
                        result = torch.addmm(bias, inputs, weight)
                else:
                    result = torch.addmm(bias, inputs, weight).clone()
                    instance.mutate(result, IDENTITY, event_key=[1, "recompute"])
                torch.nn.functional.gelu(result).sum().backward()
            values.append(result.detach())
            gradients.append((inputs.grad.detach(), weight.grad.detach()))
        self.assertTrue(torch.equal(values[0], values[1]))
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(gradients[0], gradients[1]))
        )

    def test_logical_byte_manifest_handles_singleton_expanded_zero_stride(self):
        for size in (1, 4):
            value = torch.ones((), device="cuda").expand(size)
            self.assertEqual(value.stride(), (0,))
            row = next(iter(leaf_manifest({"test": value}).values()))
            self.assertEqual(
                row["sha256"], hashlib.sha256(struct.pack("<f", 1.0) * size).hexdigest()
            )
            self.assertEqual(row["bytes"], 4 * size)
            self.assertEqual(value.stride(), (0,))
            self.assertEqual(value.detach().cpu().tolist(), [1.0] * size)
            self.assertEqual(row["shape"], [size])
            self.assertEqual(row["stride"], [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
