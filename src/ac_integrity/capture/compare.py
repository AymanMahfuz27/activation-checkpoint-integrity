"""Exact payload comparisons, strict structural pairing and artifact validation."""

from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
import torch
from ac_integrity.capture.writer import read_payload, read_tensor
from ac_integrity.state import write_json, sha256

METADATA = ("shape", "dtype", "device", "layout", "stride", "storage_offset")


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def compare_tensors(a, b):
    """Keep raw-byte equality separate from numeric equality (NaNs and signed zero)."""
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    result = {"torch_equal": torch.equal(a, b), "raw_equal": False}
    if a.shape != b.shape or a.dtype != b.dtype:
        return result
    raw_a = a.reshape(-1).view(torch.uint8).reshape(a.numel(), a.element_size())
    raw_b = b.reshape(-1).view(torch.uint8).reshape(b.numel(), b.element_size())
    changed = (raw_a != raw_b).any(dim=1)
    result["raw_equal"] = not bool(changed.any())
    result["different_elements"] = int(changed.sum())
    first = int(torch.nonzero(changed)[0]) if changed.any() else None
    result["first_flat_index"] = first
    result["first_index"] = list(np.unravel_index(first, a.shape)) if first is not None and a.ndim else ([] if first is not None else None)
    if result["first_index"] is not None:
        result["first_index"] = [int(i) for i in result["first_index"]]
    if a.is_complex():
        x, y = a.to(torch.complex128).flatten(), b.to(torch.complex128).flatten()
    else:
        x, y = a.to(torch.float64).flatten(), b.to(torch.float64).flatten()
    difference = (x - y).abs()
    if a.numel():
        result.update({"max_abs_error": finite(difference.max()), "mean_abs_error": finite(difference.mean()),
                       "max_relative_error": finite((difference / y.abs().clamp_min(1e-300)).max()),
                       "relative_l1": finite(difference.sum() / y.abs().sum().clamp_min(1e-300)),
                       "relative_l2": finite(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(y).clamp_min(1e-300))})
    if a.is_floating_point():
        for name, tensor in (("original", a), ("recompute", b)):
            result[name + "_special"] = {"nan": int(torch.isnan(tensor).sum()),
                "positive_infinity": int(torch.isposinf(tensor).sum()),
                "negative_infinity": int(torch.isneginf(tensor).sum()),
                "negative_zero": int(((tensor == 0) & torch.signbit(tensor)).sum()),
                "positive_zero": int(((tensor == 0) & ~torch.signbit(tensor)).sum())}
        integer_dtype = {torch.float16: np.uint16, torch.bfloat16: np.uint16,
                         torch.float32: np.uint32, torch.float64: np.uint64}.get(a.dtype)
        if integer_dtype and a.numel():
            bits_a = np.frombuffer(raw_a.numpy().tobytes(), dtype=integer_dtype)
            bits_b = np.frombuffer(raw_b.numpy().tobytes(), dtype=integer_dtype)
            sign = np.array(1 << (a.element_size() * 8 - 1), dtype=integer_dtype)
            ordered_a = np.where(bits_a & sign, ~bits_a, bits_a ^ sign)
            ordered_b = np.where(bits_b & sign, ~bits_b, bits_b ^ sign)
            distances = np.maximum(ordered_a, ordered_b) - np.minimum(ordered_a, ordered_b)
            valid = torch.isfinite(a).flatten().numpy() & torch.isfinite(b).flatten().numpy()
            result["max_ulp_finite"] = int(distances[valid].max()) if valid.any() else None
    if first is not None:
        begin, end = max(0, first - 3), min(a.numel(), first + 4)
        result["local_view"] = {"start": begin, "original": [str(v.item()) for v in a.flatten()[begin:end]],
                                "recompute": [str(v.item()) for v in b.flatten()[begin:end]]}
    return result


def events(root, rank=0):
    with (Path(root) / "events" / f"rank_{rank}.jsonl").open() as source:
        for line in source:
            yield json.loads(line)


def compare_events(root, step=None, rank=0):
    """Join exact pair IDs. Never align traces heuristically after a divergence."""
    pairs = {}
    counts = Counter()
    duplicates = []
    for event in events(root, rank):
        if step is not None and event["step"] != step:
            continue
        if event["pair_id"] is None:
            counts["intentionally_unpaired"] += 1
            continue
        pair = pairs.setdefault(event["pair_id"], {})
        role = event["checkpoint_role"]
        if role in pair:
            duplicates.append(event["pair_id"])
        pair[role] = event
    rows = []
    comparisons = []
    for pair_id, values in pairs.items():
        roles = [role for role in values if role.startswith("recompute")]
        for role in roles or ["recompute"]:
            comparisons.append((pair_id, {"original": values.get("original"),
                                          "recompute": values.get(role)}, role))
    for pair_id, pair, recompute_role in comparisons:
        a, b = pair.get("original"), pair.get("recompute")
        row = {"pair_id": pair_id, "original": a["event_id"] if a else None,
               "recompute": b["event_id"] if b else None, "recompute_role": recompute_role}
        if a is None or b is None:
            row["status"] = "missing_original" if a is None else "missing_recompute"
        elif a["operator"] != b["operator"] or a["output_schema"] != b["output_schema"]:
            row["status"] = "structural_mismatch"
            row["operators"] = [a["operator"], b["operator"]]
        else:
            row["metadata_equal"] = all(a[k] == b[k] for k in METADATA)
            row["default_metadata_would_detect"] = any(a[k] != b[k] for k in ("shape", "dtype", "device"))
            row["original_metadata"] = {k: a[k] for k in METADATA}
            row["recompute_metadata"] = {k: b[k] for k in METADATA}
            row.update(compare_tensors(read_tensor(root, a), read_tensor(root, b)))
            row["status"] = ("metadata_mismatch" if not row["metadata_equal"] else
                             "exact_match" if row["raw_equal"] else "value_mismatch")
            row["payloads"] = [a["payload"], b["payload"]]
        counts[row["status"]] += 1
        rows.append(row)
    counts["duplicate_pair_roles"] = len(duplicates)
    complete = counts["exact_match"]
    failures = sum(value for key, value in counts.items() if key not in {"exact_match", "intentionally_unpaired"})
    folder = Path(root) / "comparisons"
    folder.mkdir(exist_ok=True)
    destination = folder / f"rank_{rank}_step_{step if step is not None else 'all'}.jsonl"
    with destination.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + "\n")
        output.flush()
        import os
        os.fsync(output.fileno())
    first = next((row for row in rows if row["status"] != "exact_match"), None)
    if first:
        capsule = Path(root) / "mismatches" / f"rank_{rank}_step_{step}"
        capsule.mkdir(parents=True, exist_ok=True)
        write_json(capsule / "first_divergence.json", first)
    return {"counts": dict(counts), "eligible_pairs": len(comparisons), "exact_pairs": complete,
            "failed": bool(failures), "first_divergence": first,
            "pair_coverage": (sum(bool(p["original"] and p["recompute"]) for _, p, _ in comparisons) / len(comparisons)) if comparisons else None}


def validate_artifacts(root, allow_partial=False):
    root = Path(root)
    failures = []
    count = 0
    ids = set()
    for index in sorted((root / "events").glob("rank_*.jsonl")):
        lines = index.read_text().splitlines()
        for number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
                if event["event_id"] in ids:
                    raise ValueError("Duplicate event ID")
                ids.add(event["event_id"])
                read_payload(root, event, allow_partial)
                count += 1
            except (ValueError, OSError, KeyError) as error:
                failures.append({"index": str(index), "line": number, "error": str(error)})
    partials = [str(p.relative_to(root)) for p in root.rglob("*.partial")]
    if partials and not allow_partial:
        failures.append({"error": "Unsealed shards", "paths": partials})
    commits = sorted(root.glob("steps/*/STEP_COMMIT"))
    if not commits:
        failures.append({"error": "No STEP_COMMIT; artifact is incomplete/abandoned"})
    for path in commits:
        try:
            commit = json.loads(path.read_text())
            index = root / "events" / f"rank_{commit['rank']}.jsonl"
            if sha256(index) != commit["index_sha256"]:
                failures.append({"error": "Committed index checksum changed"})
            if commit["tensor_count"] != count:
                failures.append({"error": "Committed tensor count differs from index"})
        except (OSError, ValueError, KeyError) as error:
            failures.append({"error": f"Invalid step commit: {error}"})
    if not (root / "summary.json").exists():
        failures.append({"error": "Missing final summary"})
    return {"valid": not failures, "verified_tensors": count, "failures": failures,
            "partial_shards": partials, "committed_steps": len(commits)}
