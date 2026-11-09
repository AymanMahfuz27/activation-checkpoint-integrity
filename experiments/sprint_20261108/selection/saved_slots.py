"""Verify checkpoint-owned saved slots without nesting saved-tensor hooks.

Only the existing non-reentrant checkpoint hook is adapted. Its own pack and
unpack callbacks still create holders, recompute tensors, preserve early stop,
and return the original objects. Hashes compare original pack-time bytes with
unpack-time bytes; no claim covers a later unpack-to-kernel mutation.
"""

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import threading
import weakref

import selector as core

import torch
import torch.utils.checkpoint as checkpoint_module
from torch.utils._python_dispatch import _disable_current_modes
from ac_integrity.capture.fingerprint import ALGORITHM, fingerprint_tensor


# Byte-exact PyTorch2.13.0 source retained by BA3565254, independently verifiable
# before import on TACC. Another backend requires a new explicit qualification.
QUALIFIED_CHECKPOINT_SHA256 = "eab49639231cc989d65c04212923eef0ade4aa059b68bcabfe2819007f34e69f"
POLICIES = ("per_unpack", "full_copy", "stock_metadata_sync", "stock_metadata_deferred")
_PATCH_LOCK = threading.Lock()
_SOURCE_CACHE = {}


def qualify_checkpoint_source(expected_sha256=QUALIFIED_CHECKPOINT_SHA256):
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise core.UnsupportedPlan("An exact checkpoint source SHA256 is required")
    source = Path(inspect.getsourcefile(checkpoint_module)).resolve()
    key = (str(source), expected_sha256)
    if key not in _SOURCE_CACHE:
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise core.UnsupportedPlan("Unqualified torch.utils.checkpoint source: " + actual)
        native = checkpoint_module._checkpoint_hook
        if native.__module__ != "torch.utils.checkpoint" or native.__name__ != "_checkpoint_hook":
            raise core.UnsupportedPlan("Checkpoint hook has already been replaced")
        _SOURCE_CACHE[key] = {"source_sha256": actual, "source_path": str(source), "native_hook": native,
                              "native_metadata": checkpoint_module._allowed_determinism_checks_to_fns["default"]}
    return _SOURCE_CACHE[key]


def _metadata(value):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.is_quantized or value.is_conj() or value.is_neg()
            or value.device.type not in ("cpu", "cuda")):
        raise core.UnsupportedPlan("Saved-slot checker supports ordinary dense CPU/CUDA tensors only")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "stride": list(value.stride()), "storage_offset": value.storage_offset()}


class _SynchronousMetadata(dict):
    """Count actual stock comparisons while retaining immediate native errors."""

    def __init__(self, owner, metadata, words):
        super().__init__(metadata=metadata, blake3_words=words)
        self.owner = owner

    def __ne__(self, other):
        self.owner.metadata_comparisons += 1
        different = dict.__ne__(self, other)
        if different:
            if not isinstance(other, dict) or self.get("metadata") != other.get("metadata"):
                self.owner.unsupported = "Native saved-slot metadata changed"
            else:
                self.owner.native_value_mismatch = True
        return different


class _DeferredMetadata:
    """Use stock slot selection, with its value decision charged to final gate.

    Native metadata inequality still raises immediately. Value inequality is
    recorded on device, and finish() must run before optimizer mutation. This is
    a fair aggregate-decision baseline, not a modification of native selection.
    """

    def __init__(self, owner, metadata, signature):
        self.owner, self.metadata, self.signature = owner, metadata, signature

    def __ne__(self, other):
        if not isinstance(other, _DeferredMetadata) or other.owner is not self.owner:
            self.owner.unsupported = "Native saved-slot metadata callback ownership changed"
            return True
        if self.metadata != other.metadata:
            self.owner.unsupported = "Native saved-slot metadata changed"
            return True
        self.owner._record_comparison(self.signature, other.signature)
        self.owner.metadata_comparisons += 1
        return False

    def __repr__(self):
        return "DeferredValueMetadata(" + repr(self.metadata) + ")"


class SavedSlotCheck:
    """One scoped forward/backward checker, with a preoptimizer final gate.

    per_unpack and full_copy wrap checkpoint's original callbacks. The stock
    policies extend its native metadata callback and keep its original live-slot
    selection. All policies capture fresh original references and include
    copies, hashes, comparisons, transfers, and reference storage in their cost.
    """

    def __init__(self, policy="per_unpack", *, expected_checkpoint_sha256=QUALIFIED_CHECKPOINT_SHA256,
                 exclude_persistent=False):
        if policy not in POLICIES:
            raise ValueError("Unknown saved-slot policy: " + str(policy))
        self.policy = policy
        self.source = qualify_checkpoint_source(expected_checkpoint_sha256)
        self.exclude_persistent = bool(exclude_persistent)
        if self.exclude_persistent and policy.startswith("stock_metadata"):
            raise ValueError("Stock metadata baselines check every native live saved slot")
        self.records = weakref.WeakKeyDictionary()
        self.flags = []
        self.frames = 0
        self.packs = 0
        self.unpacks = 0
        self.skipped_persistent_packs = 0
        self.skipped_persistent_unpacks = 0
        self.metadata_calls = 0
        self.metadata_comparisons = 0
        self.fingerprints = 0
        self.fingerprint_payload_bytes = 0
        self.reference_bytes = 0
        self.max_reference_bytes = 0
        self.streams = {}
        self.persistent_storages = set()
        self.journal = []
        self.active = False
        self.gated = False
        self.closed = False
        self.unsupported = None
        self.result = None
        self.native_value_mismatch = False
        self.thread = None
        self.callback_thread_ids = set()
        self._replacement = None

    def _require_active(self):
        if not self.active or self.gated:
            raise core.UnsupportedPlan("Checkpoint callbacks escaped the checker interval")
        # Native autograd may invoke this frame's callbacks on its own engine
        # thread. Frame construction is caller-thread restricted separately;
        # bound callbacks remain owned by this checker across engine threads.
        self.callback_thread_ids.add(threading.get_ident())

    @contextmanager
    def track_model(self, model):
        """Optionally declare exact persistent allocations outside protection.

        Default is false: parameters/buffers saved by checkpoint are checked.
        Exclusion is an explicit fault-scope restriction, never free coverage.
        """
        if self.exclude_persistent:
            with _disable_current_modes():
                self.persistent_storages = {
                    (str(value.device), int(value.untyped_storage().data_ptr()), int(value.untyped_storage().nbytes()))
                    for value in list(model.parameters()) + list(model.buffers()) if value.numel()}
        try:
            yield self
        finally:
            self.persistent_storages.clear()

    def _persistent(self, value):
        if not self.exclude_persistent or value.numel() == 0:
            return False
        with _disable_current_modes():
            storage = value.untyped_storage()
            return (str(value.device), int(storage.data_ptr()), int(storage.nbytes())) in self.persistent_storages

    def _capture(self, value, *, full_copy=False):
        _metadata(value)
        with torch.no_grad(), _disable_current_modes():
            if value.device.type == "cuda":
                key = str(value.device)
                stream = torch.cuda.current_stream(value.device).cuda_stream
                if key in self.streams and self.streams[key] != stream:
                    raise core.UnsupportedPlan("Saved-slot fingerprint CUDA stream changed")
                self.streams[key] = stream
            canonical = value.detach().clone(memory_format=torch.contiguous_format)
            if full_copy:
                return canonical
            self.fingerprints += 1
            self.fingerprint_payload_bytes += value.numel() * value.element_size()
            return fingerprint_tensor(canonical, include_sketch=False, backend="full").signatures

    def _record_comparison(self, first, second):
        with torch.no_grad(), _disable_current_modes():
            self.flags.append(torch.ne(first, second).any())

    def _release_reference(self, count):
        self.reference_bytes -= count

    def _pack(self, original_pack, value):
        self._require_active()
        holder = original_pack(value)
        if holder in self.records:
            raise core.UnsupportedPlan("Native checkpoint returned a duplicate holder")
        metadata = _metadata(value)
        skipped = self._persistent(value)
        captured = None if skipped else self._capture(value, full_copy=self.policy == "full_copy")
        reference_bytes = 0 if captured is None else captured.numel() * captured.element_size()
        ordinal = self.packs
        self.packs += 1
        self.skipped_persistent_packs += int(skipped)
        self.records[holder] = {"ordinal": ordinal, "metadata": metadata, "reference": captured,
                                "skipped": skipped, "unpacked": False}
        self.reference_bytes += reference_bytes
        self.max_reference_bytes = max(self.max_reference_bytes, self.reference_bytes)
        weakref.finalize(holder, self._release_reference, reference_bytes)
        return holder

    def _unpack(self, original_unpack, holder):
        self._require_active()
        record = self.records.get(holder)
        if record is None:
            raise core.UnsupportedPlan("Unpack has no original holder-bound reference")
        # This exact native callback owns recomputation and native metadata
        # validation. Its original return object is passed on unchanged.
        value = original_unpack(holder)
        if record["unpacked"]:
            raise core.UnsupportedPlan("Repeated holder unpack is outside this one-backward scope")
        record["unpacked"] = True
        self.unpacks += 1
        if _metadata(value) != record["metadata"]:
            raise core.UnsupportedPlan("Unpacked slot metadata differs from original pack metadata")
        if record["skipped"]:
            self.skipped_persistent_unpacks += 1
        else:
            observed = self._capture(value, full_copy=self.policy == "full_copy")
            if self.policy == "full_copy":
                with torch.no_grad(), _disable_current_modes():
                    first = record["reference"].reshape(-1).view(torch.uint8)
                    second = observed.reshape(-1).view(torch.uint8)
                    self._record_comparison(first, second)
            else:
                self._record_comparison(record["reference"], observed)
        self.journal.append({"slot": record["ordinal"], "observation": "after_native_unpack_before_return",
                             "persistent_excluded": record["skipped"], "metadata": record["metadata"]})
        return value

    def _metadata_callback(self, value):
        self._require_active()
        self.metadata_calls += 1
        metadata = _metadata(value)
        signature = self._capture(value)
        if self.policy == "stock_metadata_sync":
            with torch.no_grad(), _disable_current_modes():
                words = tuple(int(item) for item in signature.detach().cpu().tolist())
            # Native metadata comparison owns immediate mismatch handling.
            return _SynchronousMetadata(self, metadata, words)
        return _DeferredMetadata(self, metadata, signature)

    def __enter__(self):
        if self.active or self.closed or not _PATCH_LOCK.acquire(blocking=False):
            raise core.UnsupportedPlan("Nested/concurrent saved-slot adaptation is unsupported")
        try:
            if checkpoint_module._checkpoint_hook is not self.source["native_hook"]:
                raise core.UnsupportedPlan("Native checkpoint hook identity changed")
            if checkpoint_module._allowed_determinism_checks_to_fns["default"] is not self.source["native_metadata"]:
                raise core.UnsupportedPlan("Native default metadata callback identity changed")
            self.thread = threading.get_ident()
            self.active = True
            owner = self
            native = self.source["native_hook"]

            class _ValueGuardCheckpointHook(native):
                def __init__(self, frame):
                    if threading.get_ident() != owner.thread:
                        raise core.UnsupportedPlan("Concurrent/nested-worker checkpoint frame creation is unsupported")
                    super().__init__(frame)
                    owner.frames += 1
                    if not owner.policy.startswith("stock_metadata"):
                        original_pack, original_unpack = self.pack_hook, self.unpack_hook
                        if not callable(original_pack) or not callable(original_unpack):
                            raise core.UnsupportedPlan("Native saved-tensor callback interface changed")
                        self.pack_hook = lambda value: owner._pack(original_pack, value)
                        self.unpack_hook = lambda holder: owner._unpack(original_unpack, holder)

            self._replacement = _ValueGuardCheckpointHook
            checkpoint_module._checkpoint_hook = self._replacement
            if self.policy.startswith("stock_metadata"):
                checkpoint_module._allowed_determinism_checks_to_fns["default"] = self._metadata_callback
            return self
        except Exception:
            self.active = False
            _PATCH_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, traceback):
        conflict = None
        try:
            if self.policy.startswith("stock_metadata"):
                if checkpoint_module._allowed_determinism_checks_to_fns["default"] != self._metadata_callback:
                    conflict = "Concurrent metadata callback replacement detected"
                else:
                    checkpoint_module._allowed_determinism_checks_to_fns["default"] = self.source["native_metadata"]
            if checkpoint_module._checkpoint_hook is not self._replacement:
                conflict = "Concurrent checkpoint hook replacement detected"
            else:
                checkpoint_module._checkpoint_hook = self.source["native_hook"]
        finally:
            self.active = False
            self.closed = True
            self.records.clear()
            _PATCH_LOCK.release()
        if conflict is not None:
            raise core.UnsupportedPlan(conflict)
        return False

    def finish(self):
        """Decide after backward and before clipping/optimizer mutation."""
        if self.result is not None:
            return self.result
        if self.unsupported:
            raise core.UnsupportedPlan(self.unsupported)
        if self.policy.startswith("stock_metadata"):
            if self.metadata_calls < 2 or self.metadata_comparisons < 1:
                raise core.UnsupportedPlan("No original/replay stock metadata observations")
            if self.policy == "stock_metadata_deferred" and not self.flags:
                raise core.UnsupportedPlan("No live saved-slot stock metadata comparisons")
        elif not self.frames or not self.packs or not self.unpacks:
            raise core.UnsupportedPlan("No checkpoint pack/unpack observations")
        with torch.no_grad(), _disable_current_modes():
            groups = {}
            for flag in self.flags:
                groups.setdefault(str(flag.device), []).append(flag)
            flags = []
            for group in groups.values():
                flags.extend(torch.stack(group).detach().cpu().tolist())
        self.gated = True
        self.result = {"status": "VALUE_MISMATCH" if any(flags) else "PASS", "failed": any(flags),
                       "mismatches": sum(bool(value) for value in flags), "comparison_count": len(flags),
                       "policy": self.policy, "packs": self.packs, "unpacks": self.unpacks,
                       "checkpoint_frames": self.frames, "metadata_calls": self.metadata_calls,
                       "metadata_comparisons": self.metadata_comparisons,
                       "fingerprints": self.fingerprints, "fingerprint_payload_bytes": self.fingerprint_payload_bytes,
                       "max_live_reference_bytes": None if self.policy.startswith("stock_metadata") else self.max_reference_bytes,
                       "stock_reference_memory_measurement": "not measured" if self.policy.startswith("stock_metadata") else "not applicable",
                       "callback_thread_ids": sorted(self.callback_thread_ids),
                       "skipped_persistent_packs": self.skipped_persistent_packs,
                       "skipped_persistent_unpacks": self.skipped_persistent_unpacks,
                       "checkpoint_source_sha256": self.source["source_sha256"], "algorithm": ALGORITHM,
                       "observation_scope": "original checkpoint pack vs native unpack return; later mutation excluded",
                       "no_nested_saved_tensor_hooks": True, "scientific_acceptance": False}
        return self.result

    def summary(self):
        return self.result or {"status": "UNSUPPORTED_PLAN" if self.unsupported else "VALUE_MISMATCH" if self.native_value_mismatch else "INCOMPLETE",
                               "policy": self.policy, "scientific_acceptance": False}
