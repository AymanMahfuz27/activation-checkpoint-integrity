"""Eight allocated arms distinguish producer observation from later mutation.

No historical code is edited. Each arm restores the same full state. Treatment
equivalence is established using complete gradient/state comparisons before the
observed checker decisions are interpreted. No performance claim is produced.
"""

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import torch

from ac_integrity.research import completion
from ac_integrity.research.m4m5 import FaultSpec
from placement.injector import EarlyResultFault, TARGET, intervention_class, require


ARMS = (
    ("clean-early-all", True, False, "all_operators", True, False),
    ("late-unobserved", False, True, "checkpoint_boundaries", False, False),
    ("early-unobserved", True, True, "checkpoint_boundaries", False, False),
    ("late-all-observe", False, True, "all_operators", True, False),
    ("early-all-observe", True, True, "all_operators", True, False),
    ("late-boundary-observe", False, True, "checkpoint_boundaries", True, False),
    ("early-boundary-observe", True, True, "checkpoint_boundaries", True, False),
    ("early-all-enforce", True, True, "all_operators", True, True),
)
PRIMARY = {
    "element": 6154933,
    "old_bits_u32": 3198802410,
    "new_bits_u32": 3198802411,
    "role": "recompute",
}


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def leaf_manifest(value):
    """Keep typed hashes of every logical leaf, including singleton zero strides."""
    result = {}
    for name, item in completion.leaves(value):
        require(name not in result, "Duplicate state leaf path")
        if isinstance(item, torch.Tensor):
            raw = (
                item.detach()
                .clone(memory_format=torch.contiguous_format)
                .reshape(-1)
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            record = {
                "kind": "tensor",
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "device": str(item.device),
                "stride": list(item.stride()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        elif isinstance(item, (np.ndarray, np.generic)):
            raw = item.tobytes()
            record = {
                "kind": type(item).__name__,
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        elif type(item) is float:
            record = {"kind": "float", "binary64_hex": struct.pack("!d", item).hex()}
        elif item is None or type(item) in (str, int, bool):
            record = {"kind": type(item).__name__, "value": item}
        else:
            raise TypeError(
                "Unsupported full-state leaf: " + name + " " + type(item).__name__
            )
        result[name] = record
    require(result, "Empty full-state manifest")
    return result


def strict_equal(record):
    return all(
        record.get(key) is True
        for key in ("valid", "all_finite", "byte_equal", "value_equal")
    )


def parity(actual, reference):
    return {
        name: completion.compare_trees(actual[name], reference[name])
        for name in ("state", "pre_clip", "post_clip")
    }


def require_parity(record, message):
    require(all(strict_equal(item) for item in record.values()), message)


def supported_comparison(row):
    comparison = row.get("comparison")
    require(
        isinstance(comparison, dict)
        and comparison.get("mode") == "fingerprint"
        and comparison.get("policy") == "bit_exact_fail_closed"
        and comparison.get("algorithm") == "blake3-256-v1",
        "Native strict checker result absent",
    )
    require(
        type(comparison.get("eligible_pairs")) is int
        and comparison["eligible_pairs"] > 0
        and comparison.get("pair_coverage") == 1.0
        and comparison.get("original_tensors") == comparison.get("recomputed_tensors"),
        "Native checker pairing is incomplete",
    )
    counts = comparison.get("counts") or {}
    require(
        all(
            name in ("exact_match", "value_mismatch") or count == 0
            for name, count in counts.items()
        ),
        "Non-value structural/coverage failure cannot count as detection",
    )
    require(
        sum(counts.values()) == comparison["eligible_pairs"],
        "Native pair status census differs",
    )
    require(
        bool(counts.get("value_mismatch", 0)) is comparison.get("failed"),
        "Native decision contradicts finite value statuses",
    )
    require(
        not (row.get("runtime_summary") or {}).get("unsupported"),
        "Unsupported native capture outcome",
    )
    return comparison


def run_arm(engine, stream, prestate, folder, definition):
    name, early, enabled, scope, observed, enforce = definition
    completion.cutoff()
    completion.restore(engine, prestate)
    current = completion.snapshot(engine)
    restored = completion.compare_trees(current, prestate)
    require(strict_equal(restored), "Arm did not restore exact immutable prestate")
    del current
    holder, contexts = {}, Counter()
    mode = EarlyResultFault(holder) if early else None
    spec = replace(
        FaultSpec(
            "recompute-L6-fraction-low-bit0", "recompute", 6, "bit", bit=0, seed=1729
        ),
        enabled=enabled,
    )
    with (
        intervention_class(completion, holder, contexts, early),
        mode if mode is not None else nullcontext(),
    ):
        row, evidence = completion.execute(
            engine,
            stream,
            folder,
            spec=spec,
            policy="every" if observed else "disabled",
            scope=scope,
            enforce=enforce,
            preserve_rng=True,
            branch_update=1,
            gradients=True,
        )
    # Preserve the original outcome before a derived validation gate can fail.
    # These records contain metadata only; no tensor serialization is introduced.
    write(
        folder / "raw-arm-execution.json",
        {
            "status": "RAW_EXECUTION_RETAINED_BEFORE_VALIDATION",
            "name": name,
            "early": early,
            "enabled": enabled,
            "scope": scope,
            "observed": observed,
            "enforce": enforce,
            "row": row,
            "restore_parity": restored,
            "contexts": [
                {"region": key[0], "role": key[1], "count": value}
                for key, value in sorted(contexts.items())
            ],
            "mode": (
                {
                    "candidates": mode.candidates,
                    "deliveries": mode.deliveries,
                    "order": mode.order,
                }
                if mode is not None
                else None
            ),
            "scientific_acceptance": False,
        },
    )
    require(
        all(contexts[(TARGET, role)] == 4 for role in ("original", "recompute")),
        "Actual checkpoint context count differs",
    )
    placement = (
        mode.validate(enabled)
        if mode is not None
        else {
            "placement": "historical module forward hook after native producer record"
        }
    )
    events = row.get("injections")
    if enabled:
        require(
            isinstance(events, list)
            and len(events) == 1
            and all(events[0].get(key) == value for key, value in PRIMARY.items()),
            "Exact historical logical event did not reproduce",
        )
    else:
        require(events == [], "Disabled early integration emitted an event")
    require(
        row.get("status") in ("ACCEPTED", "REJECTED_PRE_OPTIMIZER"),
        "Placement arm ended in error",
    )
    require(row["status"] == "ACCEPTED" or enforce, "Observation-only arm rejected")
    if observed:
        comparison = supported_comparison(row)
        if not enabled:
            require(
                comparison["failed"] is False and row["status"] == "ACCEPTED",
                "Clean early mode control failed",
            )
    require(
        evidence.get("phase") in ("ACCEPTED", "BACKWARD_COMPLETE"),
        "Unknown terminal arm phase",
    )
    pre_clip = evidence.get("gradients")
    require(
        isinstance(pre_clip, dict) and bool(pre_clip),
        "Complete pre-clipping gradients absent",
    )
    state = completion.snapshot(engine)
    after = {
        "state": state,
        "pre_clip": pre_clip,
        "post_clip": evidence.get("post_clip_gradients"),
    }
    if row["status"] == "ACCEPTED":
        require(
            isinstance(after["post_clip"], dict) and bool(after["post_clip"]),
            "Accepted arm lacks complete post-clipping gradients",
        )
    else:
        require(
            evidence["phase"] == "BACKWARD_COMPLETE"
            and after["post_clip"] == "NOT_REACHED"
            and row["comparison"]["failed"] is True,
            "Enforcing arm rejection phase is inconsistent",
        )
    record = {
        "name": name,
        "early": early,
        "enabled": enabled,
        "scope": scope,
        "observed": observed,
        "enforce": enforce,
        "row": row,
        "restore_parity": restored,
        "placement": placement,
        "contexts": [
            {"region": key[0], "role": key[1], "count": value}
            for key, value in sorted(contexts.items())
        ],
        "post_state_leaf_manifest": leaf_manifest(state),
        "pre_clip_leaf_manifest": leaf_manifest(pre_clip),
        "post_clip_leaf_manifest": leaf_manifest(after["post_clip"])
        if isinstance(after["post_clip"], dict)
        else after["post_clip"],
        "scientific_acceptance": False,
    }
    write(folder / "arm-result.json", record)
    return record, after


def run(out, contract):
    """Run fixed arms after parent admission and the exact allocated fixtures."""
    completion.source_allocation_guard()
    require(
        contract["experiment_id"] == "BA-PLACEMENT-20261107-v1",
        "Wrong placement experiment",
    )
    require(
        contract["execution_phase"] == "DEV_PLACEMENT_CONTROL",
        "Placement requires DEV-first phase",
    )
    require(
        contract["arm_ids"] == [row[0] for row in ARMS], "Placement arm census differs"
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    item = next(
        row for row in contract["inputs"] if row["role"] == "corrected210-history1234"
    )
    stream = completion.stream_from_inputs(contract["inputs"])
    engine = completion.load_engine(item)
    prestate = completion.snapshot(engine)
    write(out / "initial-state-leaf-manifest.json", leaf_manifest(prestate))
    records, equivalence = {}, {}
    reference = None
    for definition in ARMS:
        name = definition[0]
        folder = out / name
        folder.mkdir()
        record, actual = run_arm(engine, stream, prestate, folder, definition)
        records[name] = record
        if name == "late-unobserved":
            reference = actual
            continue
        if name == "clean-early-all":
            finite = completion.compare_trees(actual["pre_clip"], actual["pre_clip"])
            require(strict_equal(finite), "Clean gradient finite/value check failed")
            write(folder / "clean-finite-gradient-check.json", finite)
        elif (
            name == "early-all-enforce"
            and record["row"]["status"] == "REJECTED_PRE_OPTIMIZER"
        ):
            gradient = completion.compare_trees(
                actual["pre_clip"], reference["pre_clip"]
            )
            protected = completion.compare_trees(
                {key: value for key, value in actual["state"].items() if key != "rng"},
                {key: value for key, value in prestate.items() if key != "rng"},
            )
            require(
                strict_equal(gradient) and strict_equal(protected),
                "Rejected early arm changed pre-gate gradient or protected persistent state",
            )
            equivalence[name] = {
                "pre_clip": gradient,
                "protected_persistent_state": protected,
                "rng_restoration_required": False,
                "rollback_claim": False,
            }
        else:
            result = parity(actual, reference)
            write(folder / "vs-late-unobserved.json", result)
            require_parity(
                result,
                "Early/late or observed/unobserved fault delivery is not equivalent",
            )
            equivalence[name] = result
        del actual
    # Compare matched native capture census after all endpoint gates passed.
    for left, right in (
        ("late-all-observe", "early-all-observe"),
        ("late-boundary-observe", "early-boundary-observe"),
        ("early-all-observe", "early-all-enforce"),
    ):
        a, b = (records[name]["row"]["comparison"] for name in (left, right))
        require(
            all(
                a[key] == b[key]
                for key in (
                    "eligible_pairs",
                    "original_tensors",
                    "recomputed_tensors",
                    "pair_coverage",
                )
            ),
            "Matched early/late native observation census differs",
        )
    pattern = {
        "late_all_outputs_match": records["late-all-observe"]["row"]["comparison"][
            "failed"
        ]
        is False,
        "early_all_outputs_detect": records["early-all-observe"]["row"]["comparison"][
            "failed"
        ]
        is True,
        "late_boundary_matches": records["late-boundary-observe"]["row"]["comparison"][
            "failed"
        ]
        is False,
        "early_boundary_matches": records["early-boundary-observe"]["row"][
            "comparison"
        ]["failed"]
        is False,
        "early_enforces_before_optimizer": records["early-all-enforce"]["row"]["status"]
        == "REJECTED_PRE_OPTIMIZER",
    }
    result = {
        "experiment_id": contract["experiment_id"],
        "execution_complete": True,
        "status": "COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "arms_executed": len(records),
        "pattern": pattern,
        "predicted_pattern_observed": all(pattern.values()),
        "classification": "POST_PRODUCER_OBSERVATION_MUTATION_EXPLAINS_RETAINED_ALL_OUTPUT_MISS"
        if all(pattern.values())
        else "PREDICTED_PLACEMENT_SIGNATURE_NOT_ESTABLISHED",
        "equivalence": equivalence,
        "arms": records,
        "scientific_acceptance": False,
        "limits": [
            "One history/state and fixed software bit; no population or incidence claim",
            "Full gradient/accepted state endpoint equivalence; no new GELU saved-input trace installed",
            "Native and module return shapes differ intentionally; exact flat word identity is fixed",
            "Pre-optimizer rejection permits RNG consumption; no rollback guarantee",
            "Forensic arms provide no overhead or timing estimate",
        ],
    }
    write(out / "results.json", result)
    return result
