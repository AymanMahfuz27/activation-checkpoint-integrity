"""Probe whether a native at-ones VJP identifies every FP32 sigmoid VJP.

This is a scientific probe, not an admission fixture. It compares two exactly
specified saved sigmoid outputs using unit and nonunit cotangents. A native
counterexample and a failure to reproduce the literal source-tree prediction
are both valid outcomes. Import and execute only after real allocated admission.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import selector as core

import torch
from torch.utils._python_dispatch import _disable_current_modes


SOURCE_URL = (
    "https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/"
    "BinaryMiscBackwardOpsKernels.cu#L45-L48"
)
SAVED_BITS = [0x3EC00000, 0x3F200000]  # 3/8 and 5/8.
UNIT_BITS = [0x3F800000, 0x3F800000]
NONUNIT_BITS = [0x3F800002, 0x3F800002]  # 1 + 2**-22.
EXPECTED_UNIT_BITS = [0x3E700000, 0x3E700000]  # 15/64.
EXPECTED_LITERAL_NONUNIT_BITS = [0x3E700003, 0x3E700004]


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _from_bits(words):
    return torch.tensor(words, dtype=torch.int32).view(torch.float32).to("cuda:0")


def _capture(value):
    _require(
        value.dtype == torch.float32
        and value.device == torch.device("cuda:0")
        and value.shape == (2,)
        and value.is_contiguous(),
        "Projection probe returned an unsupported tensor representation",
    )
    _require(bool(torch.isfinite(value).all()), "Projection probe returned nonfinite bytes")
    bits = value.view(torch.int32).cpu().tolist()
    return {
        "dtype": str(value.dtype),
        "device": str(value.device),
        "shape": list(value.shape),
        "uint32_hex": [f"{word & 0xFFFFFFFF:08x}" for word in bits],
        "logical_bytes_hex": bytes(value.view(torch.uint8).cpu().tolist()).hex(),
    }


def run_probe(output_path):
    """Save exact native outcomes and a separately evaluated literal FP tree.

    Inputs are a new JSON output path and the fixed bit patterns above. Native
    calls use aten.sigmoid_backward directly. Separate eager arithmetic kernels
    provide the round-after-each-operation reference; no reference result is
    substituted for native output. The returned classification is descriptive.
    """
    core.require_allocated()
    _require(os.environ.get("BB_TEST_DEVICE") == "cuda:0", "CUDA0 qualification required")
    _require(torch.cuda.is_available(), "Projection probe requires actual CUDA")
    _require(not torch.compiler.is_compiling(), "Compiled probe execution is unsupported")
    output_path = Path(output_path)
    _require(output_path.parent.is_dir(), "Projection output parent must already exist")
    _require(not output_path.exists(), "Projection output must be a new retained member")

    with torch.no_grad(), _disable_current_modes():
        saved = _from_bits(SAVED_BITS)
        unit = _from_bits(UNIT_BITS)
        nonunit = _from_bits(NONUNIT_BITS)
        original_inputs = {
            "saved_sigmoid_outputs": _capture(saved),
            "unit_cotangents": _capture(unit),
            "nonunit_cotangents": _capture(nonunit),
        }
        native_unit = torch.ops.aten.sigmoid_backward.default(unit, saved)
        native_nonunit = torch.ops.aten.sigmoid_backward.default(nonunit, saved)

        # Each eager call is a separate arithmetic kernel. This deliberately
        # avoids treating compiler reassociation of the native expression as an
        # observed property of the hand-evaluated reference tree.
        complement = torch.sub(torch.ones_like(saved), saved)
        first_unit = torch.mul(unit, complement)
        first_nonunit = torch.mul(nonunit, complement)
        literal_unit = torch.mul(first_unit, saved)
        literal_nonunit = torch.mul(first_nonunit, saved)
        torch.cuda.synchronize(0)

        captured = {
            "native_unit": _capture(native_unit),
            "native_nonunit": _capture(native_nonunit),
            "literal_complement": _capture(complement),
            "literal_first_unit_product": _capture(first_unit),
            "literal_first_nonunit_product": _capture(first_nonunit),
            "literal_unit": _capture(literal_unit),
            "literal_nonunit": _capture(literal_nonunit),
        }
        _require(
            captured["literal_unit"]["uint32_hex"] == [f"{word:08x}" for word in EXPECTED_UNIT_BITS]
            and captured["literal_nonunit"]["uint32_hex"]
            == [f"{word:08x}" for word in EXPECTED_LITERAL_NONUNIT_BITS],
            "Literal eager FP32 reference did not match the registered hand derivation",
        )
        _require(
            original_inputs == {
                "saved_sigmoid_outputs": _capture(saved),
                "unit_cotangents": _capture(unit),
                "nonunit_cotangents": _capture(nonunit),
            },
            "Native or reference probe mutated its inputs",
        )
        unit_equal = captured["native_unit"]["uint32_hex"][0] == captured["native_unit"]["uint32_hex"][1]
        nonunit_equal = captured["native_nonunit"]["uint32_hex"][0] == captured["native_nonunit"]["uint32_hex"][1]

    if unit_equal and not nonunit_equal:
        classification = "AT_ONES_CERTIFICATE_FALSIFIED_BY_NATIVE_COUNTEREXAMPLE"
    elif unit_equal:
        classification = "HAND_TREE_COUNTEREXAMPLE_NOT_REPRODUCED_BY_NATIVE"
    else:
        classification = "AT_ONES_EQUIVALENCE_PREMISE_NOT_REPRODUCED"
    record = {
        "probe": "FP32_SIGMOID_AT_ONES_SUFFICIENCY_FALSIFICATION_V1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "native_operator": "aten.sigmoid_backward.default",
        "primary_source_url": SOURCE_URL,
        "literal_source_expression": "(g * (1 - y)) * y",
        "inputs": original_inputs,
        "outputs": captured,
        "native_at_ones_equal": unit_equal,
        "native_nonunit_equal": nonunit_equal,
        "classification": classification,
        "hand_derivation": {
            "assumptions": "IEEE binary32, round-to-nearest ties-to-even, separate source-tree operations; all operands/results finite normal",
            "at_ones_uint32_hex": [f"{word:08x}" for word in EXPECTED_UNIT_BITS],
            "nonunit_uint32_hex": [f"{word:08x}" for word in EXPECTED_LITERAL_NONUNIT_BITS],
            "reason": "At g=1 both products are15/64. At g=1+2^-22, first-product rounding differs; final results are15/64+3*2^-26 and15/64+4*2^-26.",
        },
        "scope": "One exact native FP32 kernel/input pair; compiler reassociation may invalidate the literal prediction. No GELU or general projection result follows.",
        "scientific_acceptance": False,
        "required_admission_fixture": False,
    }
    with output_path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record
