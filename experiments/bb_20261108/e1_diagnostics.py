"""Measure clipping-norm failure mechanisms without changing any gradients.

The allocated driver supplies named pre-clipping gradients after its physical
admission. We recompute the FP32 reduction structure and a separate FP64 norm,
then distinguish nonfinite elements, reduction overflow and range limitations.
This diagnostic never clips, steps an optimizer, changes old records or grants
scientific acceptance. The native clip outcome must be recorded by the driver.
"""

import math
import os
import sys


FP32_RANGE_MARGIN = 1e-6


def _allocated_only():
    """Refuse accidental numerical execution on the laptop or a login shell."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if sys.platform != "linux" or os.getuid() != 893489 or not job.isdecimal():
        raise RuntimeError("E1 numerical diagnostics require an owned Slurm job")
    if os.environ.get("BB_NUMERICAL_ADMISSION") != "PASS:" + job:
        raise RuntimeError("Fresh parent physical/runtime admission is required")


def _scalar(value):
    """Encode exceptional scalars explicitly in strict, portable JSON."""
    value = float(value)
    if math.isnan(value):
        return {"kind": "nan"}
    if math.isinf(value):
        return {"kind": "positive_infinity" if value > 0 else "negative_infinity"}
    return {"kind": "finite", "value": value}


def classify_norms(finite_elements, total32, total64, maximum32):
    """Classify supplied norm observations; do not assume a backend overflows.

    This metadata function also admits controlled arithmetic fixtures. The
    live probe below always supplies its independently measured native-shaped
    FP32 reduction, so a CPU backend that computes a finite norm stays finite.
    """
    if type(finite_elements) is not bool or not math.isfinite(maximum32):
        raise ValueError("Qualified element finiteness and a finite limit required")
    if maximum32 <= 0 or total32 < 0 or total64 < 0:
        raise ValueError("Norms and their representability limit must be nonnegative")
    if not finite_elements:
        return "EXPLICIT_NONFINITE_GRADIENT_ELEMENTS"
    if not math.isfinite(total64):
        return "FP64_REFERENCE_UNQUALIFIED"
    if math.isfinite(total32):
        return "FINITE_FP32_AND_FP64_NORMS"
    if math.isnan(total32):
        return "FP32_REDUCTION_NONFINITE_MECHANISM_UNRESOLVED"
    if total64 < maximum32 * (1 - FP32_RANGE_MARGIN):
        return "INTERMEDIATE_FP32_REDUCTION_OVERFLOW"
    if total64 > maximum32 * (1 + FP32_RANGE_MARGIN):
        return "REFERENCE_NORM_OUTSIDE_FP32_RANGE"
    return "FP32_REPRESENTABILITY_BOUNDARY_UNRESOLVED"


def diagnose_gradients(named_gradients):
    """Return a nonmutating diagnostic for one FP32, single-device gradient set.

    Inputs are (name, tensor-or-None) pairs from a physically admitted driver.
    Validate their identity and type, measure element finiteness and per-tensor
    norms, then reduce those norms in FP32 and FP64 independently. The wider
    reduction is a numerical reference, not exact real arithmetic. A margin
    around the FP32 limit preserves uncertainty near representability bounds.
    """
    _allocated_only()
    import torch
    from torch.utils._python_dispatch import _disable_current_modes

    gradients = []
    names = set()
    devices = set()
    absent = []
    for name, gradient in named_gradients:
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Gradient names must be nonempty and unique")
        names.add(name)
        if gradient is None:
            absent.append(name)
            continue
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("Gradient is not a tensor: " + name)
        if gradient.dtype != torch.float32 or gradient.layout != torch.strided:
            raise ValueError("Only dense FP32 gradients are qualified: " + name)
        gradients.append((name, gradient))
        devices.add(str(gradient.device))
    if not gradients or len(devices) != 1:
        raise ValueError("A nonempty single-device gradient set is required")

    rows, norms32, norms64 = [], [], []
    # Diagnostic arithmetic must not enter an enclosing observation plan.
    # Detach and read only: the original gradients, strides and RNG stay intact.
    with torch.no_grad(), _disable_current_modes():
        for name, gradient in gradients:
            detached = gradient.detach()
            finite = bool(torch.isfinite(detached).all().item())
            norm32 = torch.linalg.vector_norm(detached, ord=2)
            norm64 = torch.linalg.vector_norm(detached.to(torch.float64), ord=2)
            norms32.append(norm32)
            norms64.append(norm64)
            rows.append({
                "name": name,
                "dtype": str(gradient.dtype),
                "device": str(gradient.device),
                "shape": list(gradient.shape),
                "stride": list(gradient.stride()),
                "elements": gradient.numel(),
                "all_elements_finite": finite,
                "recomputed_fp32_norm": _scalar(norm32.item()),
                "fp64_reference_norm": _scalar(norm64.item()),
            })
        total32 = float(torch.linalg.vector_norm(torch.stack(norms32), ord=2).item())
        total64 = float(torch.linalg.vector_norm(torch.stack(norms64), ord=2).item())
        maximum32 = float(torch.finfo(torch.float32).max)

    finite_elements = all(row["all_elements_finite"] for row in rows)
    category = classify_norms(finite_elements, total32, total64, maximum32)

    return {
        "category": category,
        "all_gradient_elements_finite": finite_elements,
        "recomputed_fp32_total_norm": _scalar(total32),
        "fp64_reference_total_norm": _scalar(total64),
        "fp32_maximum_finite": maximum32,
        "representability_relative_margin": FP32_RANGE_MARGIN,
        "gradient_tensor_count": len(rows),
        "absent_gradient_names": absent,
        "per_tensor": rows,
        "actual_native_clip_returned_norm": None,
        "native_outcome_owner": "Allocated driver must record the unchanged native clip outcome",
        "gradients_modified": False,
        "historical_validation_changed": False,
        "scientific_acceptance": False,
    }
