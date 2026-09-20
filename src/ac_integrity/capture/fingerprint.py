"""Compact, position-sensitive fingerprints for checkpoint recomputation.

The exact recorder writes complete tensors to disk.  This module implements the
normal-path alternative: reduce each tensor to four position-weighted 32-bit
lanes packed into two 64-bit storage slots and, when requested, a small
numerical sketch on the tensor's own device.
Original-forward and recomputation records are joined with the runtime's existing
exact ``pair_id``.

Only the equality of the two signatures is used for enforcement.  The numerical
sketch is diagnostic evidence for future calibration; it never turns a signature
mismatch into permission to update the model.
"""

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import torch

from ac_integrity.state import write_json


ALGORITHM = "mixsum128-positioned-u32-v2"
SIGNATURE_LANES = 2
SIGNATURE_BYTES = SIGNATURE_LANES * 8

_U32_MASK = (1 << 32) - 1
_U32_POSITION_SEEDS = (0x243F6A89, 0xA4093823, 0x13198A2F, 0x082EFA99)
_U32_LENGTH_SEEDS_LOW = (0x452821E7, 0x38D01377, 0xBE5466CF, 0x34E90C6D)
_U32_LENGTH_SEEDS_HIGH = (0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917)

# Signed representations of fixed 64-bit constants.  Torch performs int64
# arithmetic modulo 2**64, which gives the wraparound required by SplitMix64.
_GOLDEN_GAMMA = -7046029254386353131       # 0x9e3779b97f4a7c15
_MIX_MULTIPLIER_1 = -4658895280553007687   # 0xbf58476d1ce4e5b9
_MIX_MULTIPLIER_2 = -7723592293110705685   # 0x94d049bb133111eb

_POSITION_SEEDS = (2611923443488327891, -6626703657320631856)

_METADATA = ("shape", "dtype", "device", "layout", "stride", "storage_offset")
_STAT_NAMES = ("sum", "l1", "l2_squared", "max_abs", "projection_1", "projection_2")
_COUNT_NAMES = ("finite", "nan", "positive_infinity", "negative_infinity")


class FingerprintCapacityError(RuntimeError):
    """The configured bounded device buffers cannot represent this audit."""


def _logical_right_shift(values, bits):
    """Return an unsigned right shift while storing values in signed int64."""
    mask = (1 << (64 - bits)) - 1
    return (values >> bits) & mask


def _splitmix64(values):
    """Avalanche every input bit with deterministic wrapping int64 operations."""
    mixed = values + _GOLDEN_GAMMA
    mixed = (mixed ^ _logical_right_shift(mixed, 30)) * _MIX_MULTIPLIER_1
    mixed = (mixed ^ _logical_right_shift(mixed, 27)) * _MIX_MULTIPLIER_2
    return mixed ^ _logical_right_shift(mixed, 31)


def _hex64(value):
    """Render a signed Python integer as its underlying unsigned 64-bit word."""
    return f"0x{(int(value) & ((1 << 64) - 1)):016x}"


def _finite_json(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


@dataclass
class TensorDigest:
    """Device-resident compact representation of one tensor value."""

    signatures: torch.Tensor
    stats: torch.Tensor | None
    counts: torch.Tensor | None
    payload_bytes: int
    word_count: int
    sketch_values: int


def _fingerprint_reference_into(raw, chunk_bytes, output):
    """Readable Torch reference for the four packed uint32 signature lanes.

    Each 32-bit payload word is multiplied by four odd weights derived from two
    independently mixed versions of its absolute position.  The sums wrap
    modulo 2**32.  Length is included in a separate domain so a zero tail cannot
    alias a shorter payload.
    """
    def mix32(values):
        values = values & _U32_MASK
        values = values ^ (values >> 16)
        values = (values * 0x7FEB352D) & _U32_MASK
        values = values ^ (values >> 15)
        values = (values * 0x846CA68B) & _U32_MASK
        return values ^ (values >> 16)

    def rotate_left32(values, bits):
        return ((values << bits) | (values >> (32 - bits))) & _U32_MASK

    def word_weights(positions):
        first = mix32(positions ^ _U32_POSITION_SEEDS[0])
        second = mix32(positions ^ _U32_POSITION_SEEDS[1])
        return (
            first | 1,
            (rotate_left32(first, 11) ^ _U32_POSITION_SEEDS[2]) | 1,
            second | 1,
            (rotate_left32(second, 17) ^ _U32_POSITION_SEEDS[3]) | 1,
        )

    payload_bytes = raw.numel()
    full_word_count = payload_bytes // 4
    tail_size = payload_bytes - full_word_count * 4
    words_per_chunk = max(1, chunk_bytes // 4)
    lane_sums = torch.zeros(4, dtype=torch.int64, device=raw.device)

    for begin in range(0, full_word_count, words_per_chunk):
        end = min(full_word_count, begin + words_per_chunk)
        words = raw[begin * 4:end * 4].view(torch.int32).to(torch.int64)
        words = words & _U32_MASK
        positions = torch.arange(begin, end, dtype=torch.int64, device=raw.device)
        for lane, weights in enumerate(word_weights(positions)):
            lane_sums[lane] += (words * weights).sum()

    if tail_size:
        tail = raw[full_word_count * 4:].to(torch.int64)
        shifts = torch.arange(tail_size, dtype=torch.int64, device=raw.device) * 8
        tail_word = (tail << shifts).sum()
        position = torch.tensor(
            [full_word_count], dtype=torch.int64, device=raw.device
        )
        for lane, weight in enumerate(word_weights(position)):
            lane_sums[lane] += tail_word * weight[0]

    length_low = payload_bytes & _U32_MASK
    length_high = payload_bytes >> 32
    for lane in range(4):
        lane_sums[lane] += (
            (length_low + 1) * _U32_LENGTH_SEEDS_LOW[lane]
            + (length_high + 1) * _U32_LENGTH_SEEDS_HIGH[lane]
        )
    lane_sums &= _U32_MASK

    packed = torch.stack((
        lane_sums[0] | (lane_sums[1] << 32),
        lane_sums[2] | (lane_sums[3] << 32),
    ))
    output.copy_(packed)
    return output


def fingerprint_tensor(
    tensor,
    chunk_bytes=2 * 1024 * 1024,
    include_sketch=True,
    signature_output=None,
    backend="full",
):
    """Fingerprint one tensor without transferring its payload to the host.

    Input: any dense ordinary tensor accepted by the capture runtime.
    Output: two int64 signatures plus an optional numerical sketch, all left on
    the input device.  Non-contiguous tensors are fingerprinted in logical
    contiguous order, matching the exact recorder's comparison semantics.

    The readable reference views bytes as 32-bit words and computes four
    independent position-weighted modular sums, packed into two int64 values.
    CUDA uses the same arithmetic in one native kernel, writing directly into a
    caller-provided buffer row when available.  The final word packs a 1--3 byte
    tail and length is mixed separately.
    """
    if chunk_bytes < 8:
        raise ValueError("fingerprint chunk_bytes must be at least 8")
    if backend not in {"full", "bookkeeping", "launch"}:
        raise ValueError("Unknown fingerprint backend")
    if backend != "full" and include_sketch:
        raise ValueError("Diagnostic fingerprint backends cannot collect sketches")
    if (tensor.layout != torch.strided or tensor.is_quantized
            or tensor.device.type == "meta"):
        raise TypeError(
            "Fingerprinting requires a dense, non-quantized, materialized tensor"
        )

    source = tensor.detach()
    payload_bytes = source.numel() * source.element_size()
    if backend == "full":
        value = source.contiguous()
        raw = value.view(torch.uint8).reshape(-1)
    else:
        # Diagnostic backends must not copy or scan payload data. Keeping the
        # original layout cleanly separates those costs from observer work.
        value = source
        raw = None
    device = tensor.device
    if signature_output is None:
        factory = torch.zeros if backend == "bookkeeping" else torch.empty
        signatures = factory(2, dtype=torch.int64, device=device)
    else:
        signatures = signature_output
        if (signatures.shape != (2,) or signatures.dtype != torch.int64
                or signatures.device != device or not signatures.is_contiguous()):
            raise ValueError("signature_output must be a contiguous int64[2] row")

    if backend == "bookkeeping":
        # FingerprintSession pre-zeros its bounded signature buffers once. This
        # path deliberately performs no per-observation device operation.
        pass
    elif device.type == "cuda":
        from ac_integrity.capture.cuda_fingerprint import fingerprint_cuda_into
        fingerprint_cuda_into(value, signatures, scan_payload=backend == "full")
    elif backend == "launch":
        signatures.zero_()
    else:
        _fingerprint_reference_into(raw, chunk_bytes, signatures)

    stats, counts, sketch_values = (None, None, 0)
    if include_sketch:
        stats, counts, sketch_values = numerical_sketch(value, chunk_bytes)
    return TensorDigest(signatures, stats, counts, payload_bytes,
                        math.ceil(payload_bytes / 4), sketch_values)


def numerical_sketch(tensor, chunk_bytes=2 * 1024 * 1024):
    """Return compact scale, special-value and projection evidence on-device.

    Complex tensors are represented by their interleaved real and imaginary
    components.  The projections use deterministic position-derived signs.
    These aggregates help characterize a mismatch, but cancellation is possible,
    so they are never an equality oracle and never authorize a tolerant update.
    """
    value = tensor.detach().contiguous()
    if value.is_complex():
        value = torch.view_as_real(value).reshape(-1)
    else:
        value = value.reshape(-1)

    # Float64 preserves CPU diagnostics.  CUDA uses float32 unless the source is
    # already float64; sketches are evidence rather than the enforcement signal.
    use_float64 = value.device.type == "cpu" or value.dtype == torch.float64
    accumulator_dtype = torch.float64 if use_float64 else torch.float32
    elements_per_chunk = max(1, chunk_bytes // max(value.element_size(), 4))
    device = value.device

    sum_value = torch.zeros((), dtype=accumulator_dtype, device=device)
    l1 = torch.zeros_like(sum_value)
    l2_squared = torch.zeros_like(sum_value)
    max_abs = torch.zeros_like(sum_value)
    projection_1 = torch.zeros_like(sum_value)
    projection_2 = torch.zeros_like(sum_value)
    finite_count = torch.zeros((), dtype=torch.int64, device=device)
    nan_count = torch.zeros_like(finite_count)
    positive_infinity = torch.zeros_like(finite_count)
    negative_infinity = torch.zeros_like(finite_count)

    for begin in range(0, value.numel(), elements_per_chunk):
        end = min(value.numel(), begin + elements_per_chunk)
        numeric = value[begin:end].to(accumulator_dtype)
        finite = torch.isfinite(numeric)
        nan_count = nan_count + torch.isnan(numeric).sum(dtype=torch.int64)
        positive_infinity = positive_infinity + torch.isposinf(numeric).sum(dtype=torch.int64)
        negative_infinity = negative_infinity + torch.isneginf(numeric).sum(dtype=torch.int64)
        finite_count = finite_count + finite.sum(dtype=torch.int64)

        clean = torch.where(finite, numeric, torch.zeros_like(numeric))
        absolute = clean.abs()
        sum_value = sum_value + clean.sum()
        l1 = l1 + absolute.sum()
        l2_squared = l2_squared + (clean * clean).sum()
        if clean.numel():
            max_abs = torch.maximum(max_abs, absolute.max())

        positions = torch.arange(begin, end, dtype=torch.int64, device=device)
        bit_1 = _splitmix64(positions + _POSITION_SEEDS[0]) & 1
        bit_2 = _splitmix64(positions + _POSITION_SEEDS[1]) & 1
        sign_1 = (bit_1 * 2 - 1).to(accumulator_dtype)
        sign_2 = (bit_2 * 2 - 1).to(accumulator_dtype)
        projection_1 = projection_1 + (clean * sign_1).sum()
        projection_2 = projection_2 + (clean * sign_2).sum()

    stats = torch.stack((sum_value, l1, l2_squared, max_abs,
                         projection_1, projection_2))
    counts = torch.stack((finite_count, nan_count,
                          positive_infinity, negative_infinity))
    return stats, counts, value.numel()


class _DeviceBuffers:
    """Fixed-capacity device storage; overflow is an integrity failure."""

    def __init__(self, device, capacity, chunk_bytes, include_sketches, backend):
        self.device = torch.device(device)
        self.capacity = capacity
        self.chunk_bytes = chunk_bytes
        self.include_sketches = include_sketches
        self.backend = backend
        self.original_count = 0
        self.comparison_count = 0
        signature_factory = torch.zeros if backend == "bookkeeping" else torch.empty
        self.original_signatures = signature_factory(
            (capacity, 2), dtype=torch.int64, device=device
        )
        self.recompute_signatures = signature_factory(
            (capacity, 2), dtype=torch.int64, device=device
        )
        self.flags = torch.zeros(capacity, dtype=torch.int64, device=device)

        self.original_stats = None
        self.recompute_stats = None
        self.original_counts = None
        self.recompute_counts = None
        if include_sketches:
            stat_dtype = torch.float64 if self.device.type == "cpu" else torch.float32
            self.original_stats = torch.empty(
                (capacity, len(_STAT_NAMES)), dtype=stat_dtype, device=device
            )
            self.recompute_stats = torch.empty_like(self.original_stats)
            self.original_counts = torch.empty(
                (capacity, len(_COUNT_NAMES)), dtype=torch.int64, device=device
            )
            self.recompute_counts = torch.empty_like(self.original_counts)

    def reserve_original(self):
        if self.original_count >= self.capacity:
            raise FingerprintCapacityError(
                f"fingerprint original capacity {self.capacity} exceeded on {self.device}"
            )
        index = self.original_count
        self.original_count += 1
        return index

    def reserve_comparison(self):
        if self.comparison_count >= self.capacity:
            raise FingerprintCapacityError(
                f"fingerprint comparison capacity {self.capacity} exceeded on {self.device}"
            )
        index = self.comparison_count
        self.comparison_count += 1
        return index

    def store_original(self, index, digest):
        target = self.original_signatures[index]
        if digest.signatures.data_ptr() != target.data_ptr():
            target.copy_(digest.signatures)
        if self.include_sketches:
            self.original_stats[index].copy_(digest.stats)
            self.original_counts[index].copy_(digest.counts)

    def store_comparison(self, result_index, original_index, digest, structural_failure):
        target = self.recompute_signatures[result_index]
        if digest.signatures.data_ptr() != target.data_ptr():
            target.copy_(digest.signatures)
        if self.backend == "bookkeeping" and not structural_failure:
            # The buffer was pre-zeroed. Avoid every per-output device op so
            # this diagnostic arm measures observer bookkeeping by itself.
            return
        if self.device.type == "cuda":
            from ac_integrity.capture.cuda_fingerprint import (
                compare_cuda_signatures_into,
            )
            compare_cuda_signatures_into(
                self.original_signatures[original_index],
                digest.signatures,
                self.flags[result_index],
                structural_failure=structural_failure,
            )
        else:
            mismatch = (
                self.original_signatures[original_index] != digest.signatures
            ).any()
            if structural_failure:
                mismatch = torch.ones((), dtype=torch.bool, device=self.device)
            self.flags[result_index].copy_(mismatch.to(torch.int64))
        if self.include_sketches:
            self.recompute_stats[result_index].copy_(digest.stats)
            self.recompute_counts[result_index].copy_(digest.counts)

    def allocated_bytes(self):
        tensors = [self.original_signatures, self.recompute_signatures, self.flags,
                   self.original_stats, self.recompute_stats,
                   self.original_counts, self.recompute_counts]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


@dataclass
class _OriginalRecord:
    event: dict
    device_key: str
    buffer_index: int


@dataclass
class _ComparisonRecord:
    original: _OriginalRecord | None
    recompute: dict
    device_key: str | None
    result_index: int | None
    structural_status: str | None


class FingerprintSession:
    """Pair, compare and summarize fingerprints without retaining tensor payloads."""

    def __init__(self, root, capacity=16384, chunk_bytes=2 * 1024 * 1024,
                 include_sketches=True, require_pairs=True, backend="full",
                 scope="all_operators"):
        if capacity <= 0:
            raise ValueError("fingerprint capacity must be positive")
        if chunk_bytes < 8:
            raise ValueError("fingerprint chunk size must be at least 8 bytes")
        if backend not in {"full", "bookkeeping", "launch"}:
            raise ValueError("Unknown fingerprint backend")
        if backend != "full" and include_sketches:
            raise ValueError("Diagnostic fingerprint backends cannot collect sketches")
        if scope not in {"all_operators", "checkpoint_boundaries"}:
            raise ValueError("Unknown fingerprint scope")
        self.root = Path(root)
        self.capacity = capacity
        self.chunk_bytes = chunk_bytes
        self.include_sketches = include_sketches
        self.require_pairs = require_pairs
        self.backend = backend
        self.scope = scope
        self.devices = {}
        self.originals = {}
        self.comparisons = []
        self.comparison_keys = set()
        self.anomalies = []
        self.counts = Counter()
        self.payload_bytes_seen = 0
        self.result = None

    def _buffers(self, device):
        key = str(device)
        if key not in self.devices:
            self.devices[key] = _DeviceBuffers(
                device, self.capacity, self.chunk_bytes, self.include_sketches,
                self.backend,
            )
        return key, self.devices[key]

    def observe(self, event, tensor):
        """Consume one original or recomputed output while it is still available."""
        if self.result is not None:
            raise RuntimeError("fingerprint session already finalized")
        pair_id = event["pair_id"]
        role = event["checkpoint_role"]
        if pair_id is None or role not in {"original", "recompute", "recompute_parent"}:
            self.counts["intentionally_unpaired"] += 1
            return

        if role == "original":
            if pair_id in self.originals:
                self.anomalies.append({"status": "duplicate_original", "event": event})
                self.counts["duplicate_original"] += 1
                return
            try:
                device_key, buffers = self._buffers(tensor.device)
                index = buffers.reserve_original()
                digest = fingerprint_tensor(
                    tensor, self.chunk_bytes, self.include_sketches,
                    signature_output=buffers.original_signatures[index],
                    backend=self.backend,
                )
                buffers.store_original(index, digest)
            except FingerprintCapacityError:
                self.counts["capacity_exceeded"] += 1
                self.anomalies.append({"status": "capacity_exceeded", "event": event})
                raise
            self.originals[pair_id] = _OriginalRecord(event, device_key, index)
            self.payload_bytes_seen += digest.payload_bytes
            self.counts["original"] += 1
            return

        comparison_key = (pair_id, role)
        if comparison_key in self.comparison_keys:
            self.anomalies.append({"status": "duplicate_recompute", "event": event})
            self.counts["duplicate_recompute"] += 1
            return
        self.comparison_keys.add(comparison_key)
        original = self.originals.get(pair_id)
        if original is None:
            self.comparisons.append(_ComparisonRecord(None, event, None, None, "missing_original"))
            self.counts["missing_original"] += 1
            return

        original_event = original.event
        if (original_event["operator"] != event["operator"]
                or original_event["output_schema"] != event["output_schema"]):
            structural_status = "structural_mismatch"
        elif any(original_event[name] != event[name] for name in _METADATA):
            structural_status = "metadata_mismatch"
        else:
            structural_status = None

        if original.device_key != str(tensor.device):
            self.comparisons.append(_ComparisonRecord(
                original, event, None, None, structural_status or "metadata_mismatch"
            ))
            self.counts["metadata_mismatch"] += 1
            return

        try:
            buffers = self.devices[original.device_key]
            result_index = buffers.reserve_comparison()
            digest = fingerprint_tensor(
                tensor, self.chunk_bytes, self.include_sketches,
                signature_output=buffers.recompute_signatures[result_index],
                backend=self.backend,
            )
            buffers.store_comparison(result_index, original.buffer_index, digest,
                                     structural_status is not None)
        except FingerprintCapacityError:
            self.counts["capacity_exceeded"] += 1
            self.anomalies.append({"status": "capacity_exceeded", "event": event})
            raise
        self.payload_bytes_seen += digest.payload_bytes
        self.comparisons.append(_ComparisonRecord(
            original, event, original.device_key, result_index, structural_status
        ))
        self.counts["recompute"] += 1

    def _base_row(self, record):
        original = record.original.event if record.original else None
        recompute = record.recompute
        source = original or recompute
        row = {
            "pair_id": source["pair_id"],
            "original": original["event_id"] if original else None,
            "recompute": recompute["event_id"],
            "recompute_role": recompute["checkpoint_role"],
            "operator": source["operator"],
            "region": source["region"],
            "microbatch": source["microbatch"],
            "op_ordinal": source["op_ordinal"],
            "module": source["module"],
            "original_sequence": original["sequence"] if original else None,
            "recompute_sequence": recompute["sequence"],
        }
        if original:
            row["metadata_equal"] = all(original[name] == recompute[name] for name in _METADATA)
            row["default_metadata_would_detect"] = any(
                original[name] != recompute[name] for name in ("shape", "dtype", "device")
            )
            row["original_metadata"] = {name: original[name] for name in _METADATA}
            row["recompute_metadata"] = {name: recompute[name] for name in _METADATA}
        return row

    def _add_device_evidence(self, row, record, host_buffers):
        values = host_buffers[record.device_key]
        original_index = record.original.buffer_index
        result_index = record.result_index
        original_signature = values["original_signatures"][original_index]
        recompute_signature = values["recompute_signatures"][result_index]
        row["signatures"] = {
            "algorithm": (
                ALGORITHM if self.backend == "full"
                else f"diagnostic-{self.backend}-only"
            ),
            "original": [_hex64(value) for value in original_signature],
            "recompute": [_hex64(value) for value in recompute_signature],
        }
        if self.include_sketches:
            original_stats = values["original_stats"][original_index]
            recompute_stats = values["recompute_stats"][result_index]
            row["numerical_sketch"] = {
                "policy": "diagnostic_only",
                "original": {
                    name: _finite_json(value)
                    for name, value in zip(_STAT_NAMES, original_stats)
                },
                "recompute": {
                    name: _finite_json(value)
                    for name, value in zip(_STAT_NAMES, recompute_stats)
                },
                "delta": {name: _finite_json(b - a) for name, a, b in
                          zip(_STAT_NAMES, original_stats, recompute_stats)},
                "original_counts": {name: int(value) for name, value in
                                    zip(_COUNT_NAMES, values["original_counts"][original_index])},
                "recompute_counts": {name: int(value) for name, value in
                                     zip(_COUNT_NAMES, values["recompute_counts"][result_index])},
            }

    def finalize(self):
        """Make one normal-path host decision per tensor device after backward."""
        if self.result is not None:
            return self.result

        compared_pair_ids = {record.original.event["pair_id"] for record in self.comparisons
                             if record.original is not None}
        missing_rows = []
        for pair_id, original in self.originals.items():
            if pair_id not in compared_pair_ids:
                event = original.event
                row = {
                    "pair_id": pair_id, "original": event["event_id"], "recompute": None,
                    "recompute_role": "recompute", "operator": event["operator"],
                    "region": event["region"], "microbatch": event["microbatch"],
                    "op_ordinal": event["op_ordinal"], "module": event["module"],
                    "original_sequence": event["sequence"], "recompute_sequence": None,
                    "status": "missing_recompute",
                }
                missing_rows.append(row)

        device_has_mismatch = {}
        decision_host_checks = 0
        for key, buffers in self.devices.items():
            if buffers.comparison_count:
                # This is the only device-to-host scalar read on a clean,
                # single-device audited step.
                device_has_mismatch[key] = bool(
                    buffers.flags[:buffers.comparison_count].any().item()
                )
                decision_host_checks += 1
            else:
                device_has_mismatch[key] = False

        host_buffers = {}
        diagnostic_transfers = 0
        for key, buffers in self.devices.items():
            if not device_has_mismatch[key]:
                continue
            used_originals = buffers.original_count
            used_comparisons = buffers.comparison_count
            host_buffers[key] = {
                "flags": buffers.flags[:used_comparisons].cpu().tolist(),
                "original_signatures": buffers.original_signatures[:used_originals].cpu().tolist(),
                "recompute_signatures": (
                    buffers.recompute_signatures[:used_comparisons].cpu().tolist()
                ),
            }
            diagnostic_transfers += 3
            if self.include_sketches:
                host_buffers[key].update({
                    "original_stats": buffers.original_stats[:used_originals].cpu().tolist(),
                    "recompute_stats": buffers.recompute_stats[:used_comparisons].cpu().tolist(),
                    "original_counts": buffers.original_counts[:used_originals].cpu().tolist(),
                    "recompute_counts": buffers.recompute_counts[:used_comparisons].cpu().tolist(),
                })
                diagnostic_transfers += 4

        rows = []
        final_counts = Counter()
        for record in self.comparisons:
            row = self._base_row(record)
            if record.structural_status:
                status = record.structural_status
            elif record.device_key is None or record.result_index is None:
                status = "missing_original"
            elif not device_has_mismatch[record.device_key]:
                status = "exact_match"
            else:
                flag = host_buffers[record.device_key]["flags"][record.result_index]
                status = "value_mismatch" if flag else "exact_match"
                if flag:
                    self._add_device_evidence(row, record, host_buffers)
            row["status"] = status
            final_counts[status] += 1
            rows.append(row)

        rows.extend(missing_rows)
        final_counts["missing_recompute"] += len(missing_rows)
        if self.require_pairs and not self.originals:
            rows.append({
                "pair_id": None, "original": None, "recompute": None,
                "recompute_role": None, "operator": None, "region": None,
                "microbatch": None, "op_ordinal": None, "module": None,
                "original_sequence": None, "recompute_sequence": None,
                "status": "no_eligible_pairs",
            })
            final_counts["no_eligible_pairs"] += 1
        for anomaly in self.anomalies:
            event = anomaly["event"]
            row = {
                "pair_id": event.get("pair_id"), "original": None,
                "recompute": event.get("event_id"),
                "recompute_role": event.get("checkpoint_role"), "operator": event.get("operator"),
                "region": event.get("region"), "microbatch": event.get("microbatch"),
                "op_ordinal": event.get("op_ordinal"), "module": event.get("module"),
                "original_sequence": None, "recompute_sequence": event.get("sequence"),
                "status": anomaly["status"],
            }
            rows.append(row)
            final_counts[anomaly["status"]] += 1

        mismatches = [row for row in rows if row["status"] != "exact_match"]
        ordered = [row for row in mismatches if row.get("original_sequence") is not None]
        first = (
            min(ordered, key=lambda row: row["original_sequence"])
            if ordered else (mismatches[0] if mismatches else None)
        )
        observed = [row for row in mismatches if row.get("recompute_sequence") is not None]
        first_observed = (
            min(observed, key=lambda row: row["recompute_sequence"])
            if observed else None
        )
        originals_with_recompute = len(compared_pair_ids)
        allocated_bytes = sum(buffers.allocated_bytes() for buffers in self.devices.values())
        if self.backend == "bookkeeping":
            comparison_backend = "diagnostic_prezeroed"
        elif any(buffers.device.type == "cuda" for buffers in self.devices.values()):
            comparison_backend = "native_cuda_single_launch"
        else:
            comparison_backend = "pytorch"

        if mismatches:
            folder = self.root / "mismatches" / "fingerprint"
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "first_divergence.json", first)
            with (folder / "all_mismatches.jsonl").open("w") as output:
                for row in mismatches:
                    output.write(json.dumps(row, allow_nan=False) + "\n")

        self.result = {
            "mode": "fingerprint",
            "observation_scope": self.scope,
            "algorithm": (
                ALGORITHM if self.backend == "full"
                else f"diagnostic-{self.backend}-only"
            ),
            "comparison_backend": comparison_backend,
            "byte_order": sys.byteorder,
            "policy": (
                "bit_exact_fail_closed" if self.backend == "full"
                else "diagnostic_timing_only"
            ),
            "sketch_policy": "diagnostic_only" if self.include_sketches else "disabled",
            "counts": dict(final_counts),
            "eligible_pairs": len(self.comparisons) + len(missing_rows),
            "exact_pairs": final_counts["exact_match"],
            "failed": bool(mismatches),
            "first_divergence": first,
            "first_observed_in_backward": first_observed,
            "pair_coverage": (
                originals_with_recompute / len(self.originals)
                if self.originals else None
            ),
            "original_tensors": len(self.originals),
            "recomputed_tensors": len(self.comparisons),
            "full_payload_bytes_seen": self.payload_bytes_seen,
            "signature_bytes_per_tensor": SIGNATURE_BYTES,
            "configured_capacity_per_device": self.capacity,
            "allocated_device_bytes": allocated_bytes,
            "decision_host_checks": decision_host_checks,
            "diagnostic_transfers_after_failure": diagnostic_transfers,
        }
        return self.result
