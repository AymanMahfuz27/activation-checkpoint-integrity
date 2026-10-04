"""Compile and launch the fused fingerprint kernel with CUDA's runtime compiler.

PyTorch's CUDA wheels already include NVRTC and NVIDIA's Python driver bindings.
Using those directly keeps the fast path self-contained: no system CUDA
compiler, C++ ABI, Ninja build, or generated shared library is required.
"""

from dataclasses import dataclass
from functools import lru_cache
import ctypes
import hashlib
from pathlib import Path
import time

import torch


_SOURCE_PATH = Path(__file__).with_name("csrc") / "fingerprint_cuda.cu"
_THREADS = 256


@dataclass
class _CompiledKernel:
    """Driver objects and setup evidence retained for one CUDA context."""

    module: object
    function: object
    parent_function: object
    compare_function: object
    architecture: str
    cubin_bytes: int
    compile_seconds: float
    source_sha256: str


def _check(result, operation):
    """Turn a nonzero CUDA/NVRTC status into a localized Python failure."""
    status = result[0]
    if int(status) != 0:
        raise RuntimeError(f"{operation} failed with {status}")
    return result[1:]


def _compile_cubin(source, architecture):
    """Compile one header-free CUDA source string to a device-native CUBIN."""
    from cuda.bindings import nvrtc

    (program,) = _check(
        nvrtc.nvrtcCreateProgram(
            source,
            b"fingerprint_cuda.cu",
            0,
            [],
            [],
        ),
        "nvrtcCreateProgram",
    )
    options = [
        f"--gpu-architecture={architecture}".encode(),
        b"--std=c++11",
    ]
    try:
        compile_result = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if int(compile_result[0]) != 0:
            log_size_result = nvrtc.nvrtcGetProgramLogSize(program)
            log = ""
            if int(log_size_result[0]) == 0 and log_size_result[1]:
                log_buffer = bytearray(log_size_result[1])
                nvrtc.nvrtcGetProgramLog(program, log_buffer)
                log = bytes(log_buffer).rstrip(b"\0").decode(errors="replace")
            raise RuntimeError(
                f"nvrtcCompileProgram failed with {compile_result[0]}: {log}"
            )
        (cubin_size,) = _check(
            nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize"
        )
        cubin = bytearray(cubin_size)
        _check(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
        return cubin
    finally:
        _check(nvrtc.nvrtcDestroyProgram(program), "nvrtcDestroyProgram")


@lru_cache(maxsize=None)
def _load_kernel(device_index):
    """Compile and load one kernel for the current device's CUDA context."""
    if not torch.cuda.is_available():
        raise RuntimeError("The fused fingerprint kernel requires CUDA")

    from cuda.bindings import driver

    with torch.cuda.device(device_index):
        torch.cuda.init()
        # A real allocation retains PyTorch's primary context before the CUDA
        # Driver API loads native device code into that same context.
        torch.empty(1, dtype=torch.uint8, device=f"cuda:{device_index}")
        major, minor = torch.cuda.get_device_capability(device_index)
        architecture = f"sm_{major}{minor}"
        source = _SOURCE_PATH.read_bytes().replace(
            b'#include "blake3_core.h"',
            _SOURCE_PATH.with_name("blake3_core.h").read_bytes(),
        )
        started = time.monotonic()
        cubin = _compile_cubin(source, architecture)
        (module,) = _check(driver.cuModuleLoadData(cubin), "cuModuleLoadData")
        (function,) = _check(
            driver.cuModuleGetFunction(module, b"fingerprint_kernel"),
            "cuModuleGetFunction",
        )
        (compare_function,) = _check(
            driver.cuModuleGetFunction(module, b"compare_fingerprint_kernel"),
            "cuModuleGetFunction(compare)",
        )
        (parent_function,) = _check(
            driver.cuModuleGetFunction(module, b"parent_fingerprint_kernel"),
            "cuModuleGetFunction(parent)",
        )
        return _CompiledKernel(
            module=module,
            function=function,
            parent_function=parent_function,
            compare_function=compare_function,
            architecture=architecture,
            cubin_bytes=len(cubin),
            compile_seconds=time.monotonic() - started,
            source_sha256=hashlib.sha256(source).hexdigest(),
        )


def preload_cuda_fingerprint(device=None):
    """Compile/load the kernel before a measured step and return setup evidence."""
    device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
    if device.type != "cuda":
        raise ValueError("preload_cuda_fingerprint requires a CUDA device")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    started = time.monotonic()
    kernel = _load_kernel(device_index)
    return {
        "backend": "nvrtc-driver",
        "architecture": kernel.architecture,
        "compile_seconds": kernel.compile_seconds,
        "load_call_seconds": time.monotonic() - started,
        "cubin_bytes": kernel.cubin_bytes,
        "source": str(_SOURCE_PATH),
        "source_sha256": kernel.source_sha256,
    }


def fingerprint_cuda_into(tensor, output, scan_payload=True):
    """Write one contiguous CUDA tensor's standard BLAKE3-256 digest.

    ``scan_payload=False`` is a diagnostic ablation. It preserves the production
    native launch path but gives the kernel a zero-byte payload, so
    it cannot be used to authorize an optimizer update.

    Callers using raw helpers must order producers before the current stream
    and retain/record input and output storage until queued work completes.
    FingerprintSession additionally enforces one owning stream per device.
    """
    if (not tensor.is_cuda or tensor.is_conj() or tensor.is_neg()
            or (scan_payload and not tensor.is_contiguous())):
        raise ValueError(
            "CUDA fingerprint input must be on CUDA and contiguous when scanned"
        )
    if (not output.is_cuda or output.device != tensor.device
            or output.dtype != torch.int64 or output.shape != (4,)
            or not output.is_contiguous()):
        raise ValueError("CUDA fingerprint output must be a contiguous int64[4] row")

    from cuda.bindings import driver

    device_index = tensor.device.index
    with torch.cuda.device(device_index):
        kernel = _load_kernel(device_index)
        stream_pointer = torch.cuda.current_stream(device_index).cuda_stream
        payload_bytes = (
            tensor.numel() * tensor.element_size() if scan_payload else 0
        )
        chunks = max(1, (payload_bytes + 1023) // 1024)
        blocks = (chunks + _THREADS - 1) // _THREADS
        # Large tensors retain only one 32-byte CV per 256 KiB subtree.
        # Each level shrinks by 256; no original payload goes to the CPU.
        target = output if blocks == 1 else torch.empty(
            (blocks, 4), dtype=torch.int64, device=tensor.device
        )
        parameters = (
            (tensor.data_ptr(), payload_bytes, target.data_ptr()),
            (ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_void_p),
        )
        _check(
            driver.cuLaunchKernel(
                kernel.function,
                blocks, 1, 1,
                _THREADS, 1, 1,
                0,
                stream_pointer,
                parameters,
                0,
            ),
            "cuLaunchKernel",
        )
        while blocks > 1:
            source = target
            node_count = blocks
            blocks = (node_count + _THREADS - 1) // _THREADS
            target = output if blocks == 1 else torch.empty(
                (blocks, 4), dtype=torch.int64, device=tensor.device
            )
            parameters = (
                (source.data_ptr(), node_count, target.data_ptr()),
                (ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_void_p),
            )
            _check(driver.cuLaunchKernel(
                kernel.parent_function, blocks, 1, 1, _THREADS, 1, 1,
                0, stream_pointer, parameters, 0,
            ), "cuLaunchKernel(parent)")
    return output


def compare_cuda_signatures_into(
    original, recomputed, mismatch, structural_failure=False
):
    """Compare two CUDA signatures and write one int64 mismatch flag.

    The comparison stays on PyTorch's current stream. This replaces several
    small PyTorch operations with one native launch and never reads a value on
    the host; the batched host decision still happens once after backward.
    """
    for name, signature in (("original", original), ("recomputed", recomputed)):
        if (not signature.is_cuda or signature.dtype != torch.int64
                or signature.shape != (4,) or not signature.is_contiguous()):
            raise ValueError(
                f"CUDA {name} signature must be a contiguous int64[4] row"
            )
    if original.device != recomputed.device:
        raise ValueError("CUDA signatures must be on the same device")
    if (not mismatch.is_cuda or mismatch.device != original.device
            or mismatch.dtype != torch.int64 or mismatch.shape != ()
            or not mismatch.is_contiguous()):
        raise ValueError("CUDA mismatch output must be a contiguous int64 scalar")

    from cuda.bindings import driver

    device_index = original.device.index
    with torch.cuda.device(device_index):
        kernel = _load_kernel(device_index)
        stream_pointer = torch.cuda.current_stream(device_index).cuda_stream
        parameters = (
            (
                original.data_ptr(),
                recomputed.data_ptr(),
                mismatch.data_ptr(),
                int(structural_failure),
            ),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint),
        )
        _check(
            driver.cuLaunchKernel(
                kernel.compare_function,
                1, 1, 1,
                1, 1, 1,
                0,
                stream_pointer,
                parameters,
                0,
            ),
            "cuLaunchKernel(compare)",
        )
    return mismatch
