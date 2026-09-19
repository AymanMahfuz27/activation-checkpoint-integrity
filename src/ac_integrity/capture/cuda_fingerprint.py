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
_MAXIMUM_BLOCKS = 256


@dataclass
class _CompiledKernel:
    """Driver objects and setup evidence retained for one CUDA context."""

    module: object
    function: object
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
        source = _SOURCE_PATH.read_bytes()
        started = time.monotonic()
        cubin = _compile_cubin(source, architecture)
        (module,) = _check(driver.cuModuleLoadData(cubin), "cuModuleLoadData")
        (function,) = _check(
            driver.cuModuleGetFunction(module, b"fingerprint_kernel"),
            "cuModuleGetFunction",
        )
        return _CompiledKernel(
            module=module,
            function=function,
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
    """Write one contiguous CUDA tensor's 128-bit signature into ``output``.

    ``scan_payload=False`` is a diagnostic ablation. It preserves the production
    memset and native launch path but gives the kernel a zero-byte payload, so
    it cannot be used to authorize an optimizer update.
    """
    if not tensor.is_cuda or (scan_payload and not tensor.is_contiguous()):
        raise ValueError(
            "CUDA fingerprint input must be on CUDA and contiguous when scanned"
        )
    if (not output.is_cuda or output.device != tensor.device
            or output.dtype != torch.int64 or output.shape != (2,)
            or not output.is_contiguous()):
        raise ValueError("CUDA fingerprint output must be a contiguous int64[2] row")

    from cuda.bindings import driver, runtime

    device_index = tensor.device.index
    with torch.cuda.device(device_index):
        kernel = _load_kernel(device_index)
        stream_pointer = torch.cuda.current_stream(device_index).cuda_stream
        _check(
            runtime.cudaMemsetAsync(
                output.data_ptr(), 0, output.numel() * output.element_size(),
                stream_pointer,
            ),
            "cudaMemsetAsync",
        )
        payload_bytes = (
            tensor.numel() * tensor.element_size() if scan_payload else 0
        )
        word_count = (payload_bytes + 3) // 4
        blocks = max(1, min(_MAXIMUM_BLOCKS, (word_count + _THREADS - 1) // _THREADS))
        parameters = (
            (tensor.data_ptr(), payload_bytes, output.data_ptr()),
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
    return output
