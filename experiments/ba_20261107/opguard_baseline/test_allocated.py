"""Allocated native XOR/wrapper and retained-trace rejection controls."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import struct
import tempfile
import unittest

from opguard_baseline.aggregate import (
    aggregate,
    EXPECTED_RANGES,
    NUMEL,
    NBYTES,
    SHAPE,
    TAG,
)


def fixture(root):
    calls = [
        {"role": role, "ordinal": ordinal, "region": "gpt_neox.layers.6", "layer": 6}
        for ordinal in range(4)
        for role in ("original", "recompute")
    ]
    arms = {}
    for treatment in ("clean", "fault"):
        for observer in ("native", "official"):
            name = treatment + "-" + observer
            folder = root / name
            folder.mkdir()
            arm = {
                "status": "ARM_COMPLETE_AWAITING_NORMAL_PROCESS_EXIT",
                "experiment_id": "BA-OPGUARD-L6-20261107-v1",
                "fault": treatment == "fault",
                "official": observer == "official",
                "calls": calls,
                "contexts": [
                    {"region": "gpt_neox.layers.6", "role": role, "count": 4}
                    for role in ("original", "recompute")
                ],
                "initial_state": {"root": {"type": "int", "value": 1}},
                "post_state": {
                    "root": {"type": "int", "value": 2 if treatment == "clean" else 3}
                },
                "gradients": {"root": {"type": "tensor", "sha256": "a" * 64}},
                "row": {
                    "injections": [
                        {
                            "element": 6154933,
                            "old_bits_u32": 3198802410,
                            "new_bits_u32": 3198802411,
                            "role": "recompute",
                        }
                    ]
                    if treatment == "fault"
                    else []
                },
            }
            if observer == "official":
                arm["official_forward_receipts"] = [
                    {**call, "original_calls": 1, "same_result_object": True}
                    for call in calls
                ]
                arm["native_backend"] = {
                    "fallback_reported": False,
                    "native_sha256": "b" * 64,
                }
                trace = (
                    folder
                    / "official-trace"
                    / "exec_trace"
                    / "gpu_000"
                    / "fixture.jsonl"
                )
                trace.parent.mkdir(parents=True)
                records = []
                for index, call in enumerate(calls):
                    values = [10, 20, 30, 40]
                    if (
                        treatment == "fault"
                        and call["role"] == "recompute"
                        and call["ordinal"] == 0
                    ):
                        values[-1] ^= 1

                    def summary(signature):
                        return {
                            "shape": SHAPE,
                            "stats": {
                                "dtype": "torch.float32",
                                "device": "cuda:0",
                                "numel": NUMEL,
                                "xor_signature": signature,
                                "xor_sig_mode": "bytes",
                                "xor_sig_layout": {
                                    "scheme": "flat-u32-le-v1",
                                    "regions": 4,
                                    "logical_nbytes": NBYTES,
                                    "byte_ranges": EXPECTED_RANGES,
                                },
                            },
                        }

                    records.append(
                        {
                            "counter": index + 1,
                            "tag_prefix": TAG,
                            "gpu_rank": 0,
                            "dump_mode": "summary",
                            "is_async": False,
                            "meta": {
                                "ba_role": call["role"],
                                "ba_ordinal": call["ordinal"],
                                "ba_region": "gpt_neox.layers.6",
                                "ba_layer": 6,
                            },
                            "inputs": {
                                "args": ["synthetic-original-callable", summary(values)]
                            },
                            "outputs": summary([1, 2, 3, 4]),
                        }
                    )
                trace.write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
            (folder / "arm-result.json").write_text(json.dumps(arm))
            (folder / "child-outcome.json").write_text(
                json.dumps({"returncode": 0, "normal_exit": True})
            )
            arms[name] = folder
    qualification = {
        "status": "PASS",
        "opguard_revision": "146d6bb523cf8364bbf2788cb319aef1a46f966b",
        "doctor": {"native_ready": True},
        "native_smoke": {"verified": True},
    }
    return arms, qualification


class OfficialBaselineFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            sys.platform != "linux"
            or not os.environ.get("SLURM_JOB_ID", "").isdecimal()
        ):
            raise RuntimeError("Actual allocated fixture context required; do not skip")

    def test_registered_single_bit_input_delta_with_equal_outputs_qualifies(self):
        with tempfile.TemporaryDirectory() as folder:
            arms, qualification = fixture(Path(folder))
            result = aggregate(arms, qualification)
            self.assertTrue(result["qualified"])
            self.assertTrue(result["primary_input_signature_changed"])
            self.assertTrue(result["all_output_signatures_equal"])
            self.assertFalse(result["scientific_acceptance"])

    def test_uncommitted_raw_trace_and_abnormal_child_exit_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            arms, qualification = fixture(Path(folder))
            trace = next((arms["fault-official"] / "official-trace").rglob("*.jsonl"))
            original = trace.read_bytes()
            trace.write_bytes(original[:-1])
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                aggregate(arms, qualification)
            trace.write_bytes(original)
            (arms["fault-official"] / "child-outcome.json").write_text(
                json.dumps({"returncode": 0, "normal_exit": False})
            )
            with self.assertRaisesRegex(ValueError, "finish normally"):
                aggregate(arms, qualification)

    def test_placeholder_and_wrong_regional_layout_are_not_value_coverage(self):
        for field, value in (
            ("xor_sig_mode", "placeholder-meta"),
            ("xor_sig_layout", {}),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                arms, qualification = fixture(Path(folder))
                trace = next(
                    (arms["fault-official"] / "official-trace").rglob("*.jsonl")
                )
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                rows[0]["inputs"]["args"][1]["stats"][field] = value
                trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    aggregate(arms, qualification)

    def test_gradient_state_or_reference_change_disqualifies_observer(self):
        for field in ("initial_state", "post_state", "gradients"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                arms, qualification = fixture(Path(folder))
                path = arms["fault-official"] / "arm-result.json"
                arm = json.loads(path.read_text())
                arm[field] = {"wrong": 1}
                path.write_text(json.dumps(arm))
                with self.assertRaisesRegex(ValueError, "changes or omits"):
                    aggregate(arms, qualification)

    def test_native_smoke_and_signature_control_cannot_be_invented(self):
        with tempfile.TemporaryDirectory() as folder:
            arms, qualification = fixture(Path(folder))
            wrong = copy.deepcopy(qualification)
            wrong["native_smoke"]["verified"] = False
            with self.assertRaisesRegex(ValueError, "qualification"):
                aggregate(arms, wrong)
            trace = next((arms["fault-official"] / "official-trace").rglob("*.jsonl"))
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            for row in rows:
                row["inputs"]["args"][1]["stats"]["xor_signature"] = [10, 20, 30, 40]
            trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = aggregate(arms, qualification)
            self.assertFalse(result["qualified"])
            self.assertEqual(result["status"], "UNSUPPORTED_SIGNATURE_CONTROL")

    def test_installed_native_regional_xor_exposes_the_selected_single_bit(self):
        import torch
        import xor_reduce_cuda as native

        self.assertTrue(torch.cuda.is_available())
        data = torch.zeros(64, dtype=torch.uint8, device="cuda")
        before = (
            native.xor_reduce_regional_cuda(data, 4).view(torch.uint32).cpu().tolist()
        )
        data[63] = 1
        after = (
            native.xor_reduce_regional_cuda(data, 4).view(torch.uint32).cpu().tolist()
        )
        self.assertEqual([a ^ b for a, b in zip(before, after)], [0, 0, 0, 1 << 24])

    def test_official_wrapper_executes_original_once_and_returns_same_object(self):
        import torch
        from opguard.wrapped_ops import make_wrapper

        calls = []

        def original(value):
            output = torch.nn.functional.gelu(value)
            calls.append(output)
            return output

        result = make_wrapper(
            tag_prefix="ba-native-fixture", dump_tensors=False, global_sync=False
        )(1)(original, torch.ones(4, device="cuda"))
        self.assertEqual(len(calls), 1)
        self.assertIs(result, calls[0])

    def test_manifest_canonicalizes_singleton_and_expanded_zero_stride(self):
        import torch
        from ac_integrity.research import completion
        from opguard_baseline.run_arm import manifest

        for size in (1, 4):
            value = torch.ones((), device="cuda").expand(size)
            self.assertEqual(value.stride(), (0,))
            actual = manifest({"zero_stride": value}, completion)
            record = next(iter(actual.values()))
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(struct.pack("<f", 1.0) * size).hexdigest(),
            )
            self.assertEqual(record["bytes"], size * 4)
            self.assertEqual(value.stride(), (0,))
            self.assertEqual(value.detach().cpu().tolist(), [1.0] * size)
            self.assertEqual(record["shape"], [size])
            self.assertEqual(record["stride"], [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
