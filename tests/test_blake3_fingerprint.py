"""Independent known answers, native conformance and adversarial regressions.

Hash security comes from the standard construction, not from this finite test
set. These tests establish implementation conformance and guard known failures.
"""

import ctypes
import itertools
import json
from pathlib import Path
import random
import shutil
import struct
import subprocess

from blake3 import blake3
import pytest
import torch

from ac_integrity.capture.fingerprint import (
    FingerprintSession, FingerprintStreamError, fingerprint_tensor,
)

ROOT = Path(__file__).parents[1]
VECTORS_PATH = Path(__file__).with_name("fixtures") / "blake3_unkeyed_vectors.json"


def digest_bytes(tensor):
    digest = fingerprint_tensor(tensor, include_sketch=False).signatures.cpu()
    return struct.pack("<4q", *digest.tolist())


def byte_tensor(payload):
    # Empty input must stay materialized without retaining a Python buffer.
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone() if payload else torch.empty(0, dtype=torch.uint8)


@pytest.fixture(scope="session")
def native_hash(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for native compression conformance")
    output = tmp_path_factory.mktemp("native-blake3") / "reference.so"
    subprocess.run([
        compiler, "-std=c++11", "-O2", "-shared", "-fPIC",
        "-I", str(ROOT / "src/ac_integrity/capture/csrc"),
        str(ROOT / "tests/native_blake3_reference.cpp"), "-o", str(output),
    ], check=True, capture_output=True)
    library = ctypes.CDLL(str(output))
    function = library.aci_blake3_cpu
    function.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_void_p]
    function.restype = None

    def hash_payload(payload):
        source = ctypes.create_string_buffer(payload)
        destination = ctypes.create_string_buffer(32)
        function(source, len(payload), destination)
        return destination.raw

    return hash_payload


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_published_unkeyed_vectors(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    fixture = json.loads(VECTORS_PATH.read_text())
    for case in fixture["cases"]:
        size = case["input_len"]
        payload = bytes(i % 251 for i in range(size))
        expected = bytes.fromhex(case["hash"])
        assert blake3(payload).digest() == expected, size
        assert digest_bytes(byte_tensor(payload).to(device)) == expected, size


def test_native_primitives_match_published_vectors(native_hash):
    for case in json.loads(VECTORS_PATH.read_text())["cases"]:
        size = case["input_len"]
        payload = bytes(i % 251 for i in range(size))
        assert native_hash(payload).hex() == case["hash"], size


@pytest.mark.parametrize("size", [
    0, 1, 3, 4, 63, 64, 65, 1023, 1024, 1025, 2048, 2049,
    3072, 3073, 4095, 4096, 4097, 255 * 1024, 256 * 1024,
    256 * 1024 + 1, 257 * 1024, 511 * 1024, 512 * 1024,
    512 * 1024 + 1, 600 * 1024, 800 * 1024, 1024 * 1024 + 3,
    # Cross a second GPU reduction level (256**2 standard chunks).
    256 * 256 * 1024 + 1,
])
def test_native_tree_boundaries_against_independent_library(native_hash, size):
    payload = (bytes(range(251)) * ((size + 250) // 251))[:size]
    assert native_hash(payload) == blake3(payload).digest(), size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tree_boundaries_and_nondefault_stream():
    lengths = [0, 64, 65, 1024, 1025, 3073, 262144, 262145,
               524288, 524289, 800 * 1024, 1024 * 1024 + 3,
               256 * 256 * 1024 + 1]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pending = []
        for size in lengths:
            payload = (bytes(range(251)) * ((size + 250) // 251))[:size]
            tensor = byte_tensor(payload).cuda()
            signature = fingerprint_tensor(tensor, include_sketch=False).signatures
            pending.append((signature, blake3(payload).digest(), size))
    stream.synchronize()
    for signature, expected, size in pending:
        assert struct.pack("<4q", *signature.cpu().tolist()) == expected, size


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_two_sign_flip_collision_family_is_rejected(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    original = torch.arange(1, 17, dtype=torch.float32).to(device)
    expected = digest_bytes(original)
    for first, second in itertools.combinations(range(16), 2):
        changed = original.clone()
        changed[first] = -changed[first]
        changed[second] = -changed[second]
        assert digest_bytes(changed) != expected, (first, second)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_session_rejects_historical_collision(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    def event(role, sequence):
        return {
            "pair_id": "collision-regression", "event_id": role,
            "checkpoint_role": role, "operator": "checkpoint_region_output",
            "output_schema": "Tensor", "shape": [2], "dtype": "torch.float32",
            "device": str(torch.device(device if device == "cpu" else "cuda:0")),
            "layout": "torch.strided", "stride": [1], "storage_offset": 0,
            "region": "block-0", "microbatch": 0, "op_ordinal": 0,
            "module": "block-0", "sequence": sequence,
        }
    session = FingerprintSession(tmp_path, include_sketches=False,
                                 scope="checkpoint_boundaries")
    session.observe(event("original", 0), torch.tensor([1.0, 2.0], device=device))
    session.observe(event("recompute", 1), torch.tensor([-1.0, -2.0], device=device))
    result = session.finalize()
    assert result["failed"]
    assert result["counts"]["value_mismatch"] == 1
    assert result["exact_pairs"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_deterministic_mutation_corpus_against_exact_bytes(native_hash, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = random.Random(17)
    for size in (4, 8, 31, 64, 1024, 1025, 4097):
        original = bytes(rng.randrange(256) for _ in range(size))
        baseline = digest_bytes(byte_tensor(original).to(device))
        assert baseline == blake3(original).digest()
        for _ in range(128):
            changed = bytearray(original)
            for index in rng.sample(range(size), min(size, rng.randint(1, 4))):
                changed[index] ^= 1 << rng.randrange(8)
            assert bytes(changed) != original
            actual = digest_bytes(byte_tensor(changed).to(device))
            assert actual == blake3(changed).digest()
            assert native_hash(bytes(changed)) == actual
            assert actual != baseline


def test_special_bits_scalars_and_complex_inputs_preserve_exact_bytes():
    nan_a = torch.tensor([0x7fc00001], dtype=torch.int32).view(torch.float32)
    nan_b = torch.tensor([0x7fc00002], dtype=torch.int32).view(torch.float32)
    assert torch.isnan(nan_a).all() and torch.isnan(nan_b).all()
    assert digest_bytes(nan_a) != digest_bytes(nan_b)
    samples = [torch.tensor(1.0), torch.tensor([0.0, -0.0]),
               torch.tensor([float("inf"), -float("inf")]),
               torch.tensor([1 + 2j, -3 + 4j]), torch.empty((2, 0)),
               torch.arange(30).reshape(5, 6).T]
    for sample in samples:
        payload = sample.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        assert digest_bytes(sample) == blake3(payload).digest()


def test_lazy_conjugate_and_negative_views_fail_explicitly():
    for tensor in (torch.tensor([1 + 2j]).conj(), torch.tensor([1.0])._neg_view()):
        with pytest.raises(TypeError, match="resolved conjugate/negative"):
            fingerprint_tensor(tensor, include_sketch=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("switch_at", ["observe", "finalize", "none"])
def test_session_stream_ownership_is_enforced_before_host_decision(tmp_path, switch_at):
    def event(role, sequence):
        return {
            "pair_id": "stream-regression", "event_id": role,
            "checkpoint_role": role, "operator": "checkpoint_region_output",
            "output_schema": "Tensor", "shape": [2], "dtype": "torch.float32",
            "device": "cuda:0", "layout": "torch.strided", "stride": [1],
            "storage_offset": 0, "region": "block-0", "microbatch": 0,
            "op_ordinal": 0, "module": "block-0", "sequence": sequence,
        }
    owner = torch.cuda.Stream()
    session = FingerprintSession(tmp_path, include_sketches=False,
                                 scope="checkpoint_boundaries")
    with torch.cuda.stream(owner):
        session.observe(event("original", 0), torch.tensor([1.0, 2.0], device="cuda"))
        if switch_at != "observe":
            session.observe(event("recompute", 1), torch.tensor([-1.0, -2.0], device="cuda"))
        if switch_at == "none":
            # No caller synchronization: finalize must wait its own comparison.
            assert session.finalize()["failed"]
    if switch_at == "observe":
        with pytest.raises(FingerprintStreamError, match="one CUDA stream"):
            session.observe(event("recompute", 1), torch.tensor([-1.0, -2.0], device="cuda"))
    elif switch_at == "finalize":
        with pytest.raises(FingerprintStreamError, match="one CUDA stream"):
            session.finalize()
    if switch_at != "none":
        with torch.cuda.stream(owner):
            result = session.finalize()
            assert result["failed"]
            assert result["counts"]["unsupported_cuda_stream_change"] >= 1
