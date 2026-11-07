"""Focused actual-CUDA sanity checks; parent runs only after Slurm admission."""

from contextlib import contextmanager
import os
import unittest

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from ac_integrity.capture.dispatch import exclude_capture
from provenance.provenance import GeluTrace, PRIMARY_EVENT, REGION, tensor_record


class ProvenanceFixtures(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
            raise RuntimeError("Real allocated CUDA fixtures required; do not skip")

    @contextmanager
    def region(self, role):
        token = REGION.set(("gpt_neox.layers.6", role))
        try:
            yield
        finally:
            REGION.reset(token)

    def test_logical_byte_snapshot_and_exact_selected_bits(self):
        value = torch.tensor([PRIMARY_EVENT["old_bits_u32"] - 2**32], dtype=torch.int32, device="cuda").view(torch.float32)
        before = tensor_record(value, 0)
        value.view(torch.int32).bitwise_xor_(1)
        after = tensor_record(value, 0)
        self.assertEqual(before["selected_bits_u32"], PRIMARY_EVENT["old_bits_u32"])
        self.assertEqual(after["selected_bits_u32"], PRIMARY_EVENT["new_bits_u32"])
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_shared_capture_exclusion_does_not_suppress_natural_gelu(self):
        value = torch.ones(1, device="cuda", requires_grad=True)
        tracer = GeluTrace(6, 0, expected_shape=(1,))
        with tracer, self.region("original"), exclude_capture():
            torch.nn.functional.gelu(value)
        self.assertEqual(len(tracer.forward), 1)

    def test_native_checkpoint_replays_and_actual_backward_arguments_visible(self):
        tracer = GeluTrace(6, 0, expected_shape=(1,))
        with tracer:
            for index in range(4):
                # Distinct inputs make consumer-to-microbatch joins identifiable.
                value = torch.tensor([0.2 + index], device="cuda", requires_grad=True)
                with set_checkpoint_early_stop(False):
                    output = checkpoint(torch.nn.functional.gelu, value, use_reentrant=False,
                                        context_fn=lambda: (self.region("original"), self.region("recompute")))
                output.sum().backward()
        result = tracer.summary()
        self.assertEqual(len(result["forward"]), 8)
        self.assertEqual(len(result["backward"]), 4)
        self.assertTrue(all(pair["input_bytes_equal"] and pair["output_bytes_equal"]
                            and pair["backward_uses_replay_input"] for pair in result["pairs"]))
        self.assertTrue(result["same_cotangent_local_vjp"]["byte_equal"])

    def test_wrong_target_shape_fails_instead_of_claiming_empty_coverage(self):
        tracer = GeluTrace(6, 0, expected_shape=(2,))
        with self.assertRaisesRegex(RuntimeError, "Target GELU shape changed"):
            with tracer, self.region("original"):
                torch.nn.functional.gelu(torch.ones(1, device="cuda"))

    def test_empty_consumer_trace_cannot_claim_supported_coverage(self):
        tracer = GeluTrace(6, 0, expected_shape=(1,))
        with self.assertRaisesRegex(RuntimeError, "all four"):
            tracer.summary()


if __name__ == "__main__":
    unittest.main(verbosity=2)
