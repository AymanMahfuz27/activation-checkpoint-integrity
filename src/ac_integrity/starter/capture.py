"""Phase labeling, exact recording, comparison, hook observation, and artifacts."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from ac_integrity.starter.fixture import TOKEN_TAGS


@dataclass
class CapturedTensor:
    tensor: torch.Tensor
    metadata: dict[str, Any]
    pair_id: str
    event_id: str
    phase: str
    tag: str
    occurrence: int


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "stride": list(tensor.stride()) if tensor.layout == torch.strided else None,
    }


class TensorRecorder:
    """Record full tensors under stable semantic identities for two checkpoint phases."""

    def __init__(self, *, run_id: str, experiment: str, case: str) -> None:
        self.run_id = run_id
        self.experiment = experiment
        self.case = case
        self.current_phase = "outside"
        self.phase_counts = {"original": 0, "recompute": 0}
        self.records: dict[str, dict[str, list[CapturedTensor]]] = {
            "original": {},
            "recompute": {},
        }
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def phase(self, phase: str) -> Iterator[None]:
        previous = self.current_phase
        self.current_phase = phase
        self.phase_counts[phase] += 1
        self.events.append({"kind": "phase_enter", "phase": phase})
        try:
            yield
        finally:
            self.events.append({"kind": "phase_exit", "phase": phase})
            self.current_phase = previous

    def context_fn(self) -> tuple[Any, Any]:
        return self.phase("original"), self.phase("recompute")

    def record(self, tag: str, tensor: torch.Tensor) -> None:
        if self.current_phase not in self.records:
            raise RuntimeError(f"tag {tag!r} recorded outside checkpoint phase")
        phase_records = self.records[self.current_phase].setdefault(tag, [])
        occurrence = len(phase_records)
        pair_id = (
            f"{self.experiment}/{self.case}/checkpoint-0/{tag}/{occurrence}"
        )
        event_id = f"{self.run_id}/{pair_id}/{self.current_phase}"
        captured = CapturedTensor(
            tensor=tensor.detach().cpu().clone(),
            metadata=tensor_metadata(tensor),
            pair_id=pair_id,
            event_id=event_id,
            phase=self.current_phase,
            tag=tag,
            occurrence=occurrence,
        )
        phase_records.append(captured)
        self.events.append(
            {
                "kind": "direct_tag",
                "event_id": event_id,
                "pair_id": pair_id,
                "phase": self.current_phase,
                "tag": tag,
                "occurrence": occurrence,
                "metadata": captured.metadata,
            }
        )


@dataclass
class _PackedTensor:
    pack_id: int
    tensor: torch.Tensor


class SavedTensorObserver:
    """Observe public pack/unpack events and bind TaggedSave tokens to tensors."""

    def __init__(self, recorder: TensorRecorder) -> None:
        self.recorder = recorder
        self.next_pack_id = 0
        self.last_value_pack: dict[str, int | None] = {
            "original": None,
            "recompute": None,
        }
        self.pack_phase: dict[int, str] = {}
        self.pack_tag: dict[int, str] = {}
        self.token_pack_tag: dict[int, str] = {}
        self.events: list[dict[str, Any]] = []

    def pack(self, tensor: torch.Tensor) -> _PackedTensor:
        phase = self.recorder.current_phase
        pack_id = self.next_pack_id
        self.next_pack_id += 1
        self.pack_phase[pack_id] = phase
        token = self._semantic_token(tensor)
        event: dict[str, Any] = {
            "kind": "hook_pack",
            "pack_id": pack_id,
            "phase": phase,
            "metadata": tensor_metadata(tensor),
        }
        if token is None:
            self.last_value_pack[phase] = pack_id
        else:
            tag = TOKEN_TAGS[token]
            value_pack_id = self.last_value_pack[phase]
            event.update({"semantic_token": token, "tag": tag, "value_pack_id": value_pack_id})
            self.token_pack_tag[pack_id] = tag
            if value_pack_id is not None:
                self.pack_tag[value_pack_id] = tag
        self.events.append(event)
        return _PackedTensor(pack_id=pack_id, tensor=tensor)

    def unpack(self, packed: _PackedTensor) -> torch.Tensor:
        pack_id = packed.pack_id
        tag = self.pack_tag.get(pack_id) or self.token_pack_tag.get(pack_id)
        event = {
            "kind": "hook_unpack",
            "pack_id": pack_id,
            "phase": self.pack_phase[pack_id],
            "tag": tag,
        }
        self.events.append(event)
        if tag is not None:
            self.events.append(
                {
                    "kind": "hook_access",
                    "pack_id": pack_id,
                    "phase": self.pack_phase[pack_id],
                    "tag": tag,
                }
            )
        return packed.tensor

    @contextmanager
    def phase_with_hooks(self, phase: str) -> Iterator[None]:
        with self.recorder.phase(phase):
            with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
                yield

    def context_fn(self) -> tuple[Any, Any]:
        return self.phase_with_hooks("original"), self.phase_with_hooks("recompute")

    @staticmethod
    def _semantic_token(tensor: torch.Tensor) -> int | None:
        if tensor.dtype != torch.int64 or tensor.numel() != 1:
            return None
        value = int(tensor.detach().cpu().item())
        return value if value in TOKEN_TAGS else None


def _elementwise_exact(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("element comparison requires matching shape and dtype")
    if left.dtype == torch.float64:
        return left.view(torch.int64) == right.view(torch.int64)
    if left.dtype == torch.float32:
        return left.view(torch.int32) == right.view(torch.int32)
    if left.dtype in (torch.float16, torch.bfloat16):
        return left.view(torch.int16) == right.view(torch.int16)
    if left.dtype == torch.complex64:
        return (left.view(torch.int32) == right.view(torch.int32)).reshape(*left.shape, 2).all(-1)
    if left.dtype == torch.complex128:
        return (left.view(torch.int64) == right.view(torch.int64)).reshape(*left.shape, 2).all(-1)
    return left == right


def _json_number(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def compare_tensors(left: CapturedTensor, right: CapturedTensor) -> dict[str, Any]:
    """Compare metadata and values exactly, including floating-point bit patterns."""

    metadata_equal = left.metadata == right.metadata
    result: dict[str, Any] = {
        "pair_id": left.pair_id,
        "tag": left.tag,
        "original_metadata": left.metadata,
        "recompute_metadata": right.metadata,
        "metadata_equal": metadata_equal,
        "exact_equal": False,
        "different_elements": None,
        "first_differing_index": None,
        "max_absolute_difference": None,
        "relative_l2_difference": None,
    }
    if not metadata_equal or left.tensor.shape != right.tensor.shape or left.tensor.dtype != right.tensor.dtype:
        return result

    equal_elements = _elementwise_exact(left.tensor, right.tensor)
    differing = ~equal_elements
    count = int(differing.sum().item())
    result["exact_equal"] = count == 0
    result["different_elements"] = count
    if count == 0:
        result["max_absolute_difference"] = 0.0
        result["relative_l2_difference"] = 0.0
        return result

    flat_position = int(torch.nonzero(differing.reshape(-1), as_tuple=False)[0, 0].item())
    if left.tensor.ndim == 0:
        first_index: list[int] = []
    else:
        first_index = []
        remainder = flat_position
        for size in reversed(left.tensor.shape):
            first_index.append(remainder % size)
            remainder //= size
        first_index.reverse()
    result["first_differing_index"] = first_index

    left64 = left.tensor.to(torch.float64)
    right64 = right.tensor.to(torch.float64)
    numeric_difference = left64 - right64
    max_abs = float(numeric_difference.abs().max().item())
    denominator = float(left64.norm().item())
    numerator = float(numeric_difference.norm().item())
    relative = numerator if denominator == 0.0 else numerator / denominator
    result["max_absolute_difference"] = _json_number(max_abs)
    result["relative_l2_difference"] = _json_number(relative)
    return result


def pair_records(
    recorder: TensorRecorder, expected_tags: tuple[str, ...]
) -> tuple[list[tuple[CapturedTensor, CapturedTensor]], dict[str, Any]]:
    pairs: list[tuple[CapturedTensor, CapturedTensor]] = []
    missing: list[dict[str, str]] = []
    duplicates: list[dict[str, Any]] = []
    for tag in expected_tags:
        originals = recorder.records["original"].get(tag, [])
        recomputes = recorder.records["recompute"].get(tag, [])
        if not originals:
            missing.append({"phase": "original", "tag": tag})
        if not recomputes:
            missing.append({"phase": "recompute", "tag": tag})
        if len(originals) > 1:
            duplicates.append({"phase": "original", "tag": tag, "count": len(originals)})
        if len(recomputes) > 1:
            duplicates.append({"phase": "recompute", "tag": tag, "count": len(recomputes)})
        if len(originals) == 1 and len(recomputes) == 1:
            pairs.append((originals[0], recomputes[0]))
    return pairs, {"missing_events": missing, "duplicate_events": duplicates}


class ArtifactWriter:
    """Persist one immutable run as valid JSON plus full CPU tensor files."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def write_json(self, name: str, value: Any) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return path

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> Path:
        path = self.run_dir / name
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        return path

    def save_tensor(self, case: str, tag: str, phase: str, tensor: torch.Tensor) -> Path:
        path = self.run_dir / "tensors" / case / f"{tag}.{phase}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)
        return path
