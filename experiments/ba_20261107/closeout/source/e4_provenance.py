"""Bind an E4 runtime source hash to the frozen files consumed by its loader."""

import hashlib
from pathlib import Path


CUDA = "src/ac_integrity/capture/csrc/fingerprint_cuda.cu"
HEADER = "src/ac_integrity/capture/csrc/blake3_core.h"
LOADER = "src/ac_integrity/capture/cuda_fingerprint.py"
INCLUDE = b'#include "blake3_core.h"'


def reconstruct_extension_source(original):
    """Verify all frozen inputs, then repeat the pinned loader's byte expansion.

    The runtime records the expanded NVRTC input, not the raw CUDA file. No
    source is executed or compiled here. The caller still checks architecture,
    original scientific revision, the rest of the source manifest and run scope.
    """
    source_root = Path(original["source_path"])
    if source_root.is_symlink() or source_root.resolve() != source_root:
        raise ValueError("E4 frozen source root must be canonical")
    raw = {}
    pins = original["source_files"]
    for member in (CUDA, HEADER, LOADER):
        path = source_root / member
        if (
            not path.is_file()
            or path.is_symlink()
            or any(parent.is_symlink() for parent in path.parents)
        ):
            raise ValueError(
                "E4 frozen source member must be a regular file: " + member
            )
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != pins.get(member):
            raise ValueError("E4 frozen source byte hash differs: " + member)
        raw[member] = content
    if raw[CUDA].count(INCLUDE) != 1:
        raise ValueError("E4 CUDA header include must occur exactly once")
    expanded = raw[CUDA].replace(INCLUDE, raw[HEADER])
    return {
        "namespace": "header-expanded-NVRTC-input",
        "raw_cuda_source_sha256": pins[CUDA],
        "header_sha256": pins[HEADER],
        "loader_sha256": pins[LOADER],
        "assembled_nvrtc_source_sha256": hashlib.sha256(expanded).hexdigest(),
        "transformation": "raw CUDA bytes with the sole literal blake3_core.h include replaced by frozen header bytes",
    }
