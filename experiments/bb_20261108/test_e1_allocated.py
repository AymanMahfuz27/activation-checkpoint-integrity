"""Allocated E1 diagnostic fixtures; never run numerical cases on the laptop."""

import json
import math
import os
import unittest

from e1_diagnostics import (
    FP32_RANGE_MARGIN,
    _allocated_only,
    classify_norms,
    diagnose_gradients,
)


class E1DiagnosticFixtures(unittest.TestCase):
    def setUp(self):
        _allocated_only()
        import torch

        self.torch = torch
        device = os.environ.get("BB_TEST_DEVICE")
        if device not in ("cpu", "cuda:0"):
            raise RuntimeError("An explicit registered fixture device is required")
        if device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("The CUDA fixture stage cannot fall back to CPU")
        self.device = device

    def test_finite_intermediate_overflow(self):
        torch = self.torch
        value = torch.tensor([1e20, 1e20], dtype=torch.float32, device=self.device)
        # This explicit squared-sum control deliberately overflows FP32. The
        # live vector_norm backend is measured independently and may be stable.
        overflowed = torch.sqrt(torch.sum(value * value)).item()
        reference = torch.linalg.vector_norm(value.to(torch.float64)).item()
        maximum = float(torch.finfo(torch.float32).max)
        self.assertTrue(bool(torch.isfinite(value).all().item()))
        self.assertTrue(math.isinf(overflowed))
        self.assertTrue(math.isfinite(reference) and reference < maximum)
        self.assertEqual(
            classify_norms(True, overflowed, reference, maximum),
            "INTERMEDIATE_FP32_REDUCTION_OVERFLOW",
        )
        live = diagnose_gradients([("control", value)])
        self.assertEqual(live["fp64_reference_total_norm"]["kind"], "finite")
        expected = (
            "FINITE_FP32_AND_FP64_NORMS"
            if live["recomputed_fp32_total_norm"]["kind"] == "finite"
            else "INTERMEDIATE_FP32_REDUCTION_OVERFLOW"
        )
        self.assertEqual(live["category"], expected)

    def test_true_norm_outside_fp32_range(self):
        torch = self.torch
        value = torch.tensor([3e38, 3e38], dtype=torch.float32, device=self.device)
        result = diagnose_gradients([("outside_range", value)])
        self.assertTrue(result["all_gradient_elements_finite"])
        self.assertEqual(result["category"], "REFERENCE_NORM_OUTSIDE_FP32_RANGE")
        self.assertGreater(
            result["fp64_reference_total_norm"]["value"],
            result["fp32_maximum_finite"],
        )

    def test_explicit_nonfinite_json_categories(self):
        torch = self.torch
        for name, value, kind in (
            ("nan", float("nan"), "nan"),
            ("infinity", float("inf"), "positive_infinity"),
        ):
            with self.subTest(name=name):
                gradient = torch.tensor([value, 1.0], device=self.device)
                result = diagnose_gradients([(name, gradient)])
                self.assertEqual(result["category"], "EXPLICIT_NONFINITE_GRADIENT_ELEMENTS")
                self.assertEqual(result["fp64_reference_total_norm"]["kind"], kind)
                json.dumps(result, allow_nan=False)
        self.assertEqual(
            classify_norms(True, float("nan"), 2.0, 3e38),
            "FP32_REDUCTION_NONFINITE_MECHANISM_UNRESOLVED",
        )

    def test_representability_boundary_band(self):
        maximum = float(self.torch.finfo(self.torch.float32).max)
        for factor in (1 - FP32_RANGE_MARGIN / 2, 1, 1 + FP32_RANGE_MARGIN / 2):
            self.assertEqual(
                classify_norms(True, float("inf"), maximum * factor, maximum),
                "FP32_REPRESENTABILITY_BOUNDARY_UNRESOLVED",
            )
        self.assertEqual(
            classify_norms(True, float("inf"), maximum * (1 - 2 * FP32_RANGE_MARGIN), maximum),
            "INTERMEDIATE_FP32_REDUCTION_OVERFLOW",
        )
        self.assertEqual(
            classify_norms(True, float("inf"), maximum * (1 + 2 * FP32_RANGE_MARGIN), maximum),
            "REFERENCE_NORM_OUTSIDE_FP32_RANGE",
        )

    def test_diagnostic_does_not_mutate_gradients(self):
        torch = self.torch
        rng_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(0).clone() if self.device == "cuda:0" else None
        for size in (1, 4):
            gradient = torch.ones((), device=self.device).expand(size)
            before = gradient.clone(memory_format=torch.contiguous_format)
            stride = gradient.stride()
            result = diagnose_gradients([("expanded", gradient), ("unused", None)])
            self.assertTrue(torch.equal(gradient, before))
            self.assertEqual(gradient.stride(), stride)
            self.assertEqual(result["absent_gradient_names"], ["unused"])
            self.assertFalse(result["gradients_modified"])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        if cuda_before is not None:
            self.assertTrue(torch.equal(torch.cuda.get_rng_state(0), cuda_before))

    def test_rejects_unqualified_gradients(self):
        torch = self.torch
        value = torch.ones(2, dtype=torch.float32, device=self.device)
        bad_collections = [
            [],
            [("none_only", None)],
            [("duplicate", value), ("duplicate", value)],
            [("wrong_dtype", value.to(torch.float64))],
            [("sparse", value.to_sparse())],
        ]
        for collection in bad_collections:
            with self.assertRaises((ValueError, TypeError)):
                diagnose_gradients(collection)
        with self.assertRaises(ValueError):
            classify_norms(True, -1.0, 1.0, 3e38)
